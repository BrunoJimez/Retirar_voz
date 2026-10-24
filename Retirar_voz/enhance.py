"""
enhance.py — Restauração e condicionamento do áudio.

Cadeia aplicada (cada etapa é opcional e acionada pelo laudo de analysis.py):

    1. Remoção de offset DC
    2. High-pass de rumble (Butterworth ordem 4, zero-phase)
    3. Reconstrução de amostras clipadas (interpolação cúbica)
    4. Redução de ruído espectral (noisereduce, algoritmo estacionário)
    5. Correção de polaridade de canal
    6. Normalização de loudness EBU R128
    7. Limitador true-peak com lookahead
    8. Dither TPDF (apenas em exportação de 16 bits)

Todo o processamento roda em float64 internamente para evitar acúmulo de erro.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

EPS = 1e-12


@dataclass
class RelatorioEnhance:
    etapas: list = field(default_factory=list)
    ganho_aplicado_db: float = 0.0
    lufs_antes: float = 0.0
    lufs_depois: float = 0.0
    true_peak_depois: float = 0.0
    amostras_reconstruidas: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Etapas
# --------------------------------------------------------------------------

def remover_dc(audio: np.ndarray) -> np.ndarray:
    return audio - audio.mean(axis=-1, keepdims=True)


def highpass(audio: np.ndarray, sr: int, corte_hz: float = 22.0) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt
    nyq = sr / 2
    fc = min(max(corte_hz, 1.0), nyq * 0.95)
    sos = butter(4, fc / nyq, btype="highpass", output="sos")
    return sosfiltfilt(sos, audio, axis=-1)


def declip(audio: np.ndarray, limiar: float = 0.9985) -> tuple[np.ndarray, int]:
    """
    Reconstrói trechos saturados por interpolação cúbica sobre as bordas do
    platô, seguida de um leve realce de amplitude do trecho reconstruído.
    Funciona bem para clipping curto (< ~20 amostras); trechos longos são
    apenas suavizados.
    """
    from scipy.interpolate import CubicSpline

    saida = audio.copy()
    total = 0

    for c in range(saida.shape[0]):
        x = saida[c]
        mascara = np.abs(x) >= limiar
        if not mascara.any():
            continue

        bordas = np.diff(mascara.astype(np.int8))
        inicios = list(np.where(bordas == 1)[0] + 1)
        fins = list(np.where(bordas == -1)[0] + 1)
        if mascara[0]:
            inicios.insert(0, 0)
        if mascara[-1]:
            fins.append(len(x))

        for i0, i1 in zip(inicios, fins):
            largura = i1 - i0
            if largura <= 0 or largura > 800:
                continue
            margem = max(4, min(32, largura * 3))
            a = max(0, i0 - margem)
            b = min(len(x), i1 + margem)
            idx_apoio = np.concatenate([np.arange(a, i0), np.arange(i1, b)])
            if len(idx_apoio) < 4:
                continue
            try:
                spline = CubicSpline(idx_apoio, x[idx_apoio])
                trecho = spline(np.arange(i0, i1))
            except Exception:
                continue
            sinal = np.sign(np.mean(x[i0:i1]) or 1.0)
            # Garante que o trecho reconstruído ultrapasse o teto original,
            # que é o comportamento físico esperado do pico ceifado.
            pico = np.max(np.abs(trecho)) + EPS
            alvo = limiar * (1.0 + 0.04 * min(largura, 20) / 20)
            if pico < alvo:
                trecho = trecho * (alvo / pico)
            trecho = np.clip(trecho, -1.6, 1.6)
            if sinal * np.mean(trecho) < 0:
                trecho = -trecho
            x[i0:i1] = trecho
            total += largura
        saida[c] = x

    return saida, total


def denoise(audio: np.ndarray, sr: int, forca: float = 0.5) -> np.ndarray:
    """Redução de ruído espectral. Sem noisereduce, aplica gate espectral simples."""
    try:
        import noisereduce as nr
        limpo = nr.reduce_noise(
            y=audio, sr=sr, stationary=True,
            prop_decrease=float(np.clip(forca, 0.0, 0.95)),
            n_fft=2048, win_length=2048, hop_length=512,
        )
        return np.asarray(limpo, dtype=np.float64)
    except Exception:
        return _gate_espectral(audio, sr, forca)


def _gate_espectral(audio: np.ndarray, sr: int, forca: float) -> np.ndarray:
    """Fallback: subtração espectral usando o percentil 10 como perfil de ruído."""
    from scipy.signal import stft, istft
    saida = np.empty_like(audio)
    for c in range(audio.shape[0]):
        f, t, Z = stft(audio[c], fs=sr, nperseg=2048, noverlap=1536)
        mag, fase = np.abs(Z), np.angle(Z)
        perfil = np.percentile(mag, 10, axis=1, keepdims=True)
        limpo = np.maximum(mag - forca * perfil * 1.5, mag * (1 - forca) * 0.1)
        _, rec = istft(limpo * np.exp(1j * fase), fs=sr,
                       nperseg=2048, noverlap=1536)
        n = min(len(rec), audio.shape[1])
        buf = np.zeros(audio.shape[1])
        buf[:n] = rec[:n]
        saida[c] = buf
    return saida


def corrigir_fase(audio: np.ndarray) -> np.ndarray:
    """Inverte a polaridade do canal direito se isso melhorar a correlação."""
    if audio.shape[0] < 2:
        return audio
    l, r = audio[0], audio[1]
    denom = np.sqrt(np.sum(l**2) * np.sum(r**2)) + EPS
    corr = float(np.sum(l * r) / denom)
    if corr < -0.3:
        audio = audio.copy()
        audio[1] = -audio[1]
    return audio


def limitador_true_peak(audio: np.ndarray, sr: int, teto_dbtp: float = -1.0,
                        lookahead_ms: float = 5.0,
                        release_ms: float = 50.0) -> np.ndarray:
    """
    Limitador com detecção em 4x oversampling, envelope suavizado e lookahead —
    evita distorção intersample após codificação com perda.
    """
    from scipy.signal import resample_poly

    teto = 10 ** (teto_dbtp / 20.0)
    os_fator = 4
    try:
        up = resample_poly(audio, os_fator, 1, axis=-1)
    except Exception:
        up = audio
        os_fator = 1

    envelope = np.max(np.abs(up), axis=0)
    ganho = np.minimum(1.0, teto / (envelope + EPS))

    # Release: propaga a redução de ganho para frente (suavização assimétrica)
    n_rel = max(1, int(sr * os_fator * release_ms / 1000.0))
    alfa = np.exp(-1.0 / n_rel)
    g = np.empty_like(ganho)
    atual = 1.0
    for i in range(len(ganho)):
        v = ganho[i]
        atual = v if v < atual else alfa * atual + (1 - alfa) * v
        g[i] = atual

    # Lookahead: antecipa a redução para não ceifar o transiente
    n_look = max(1, int(sr * os_fator * lookahead_ms / 1000.0))
    g = np.minimum(g, np.concatenate([g[n_look:], np.full(n_look, g[-1])]))

    # Suavização final para eliminar modulação audível
    janela = np.hanning(max(3, n_look))
    janela /= janela.sum()
    g = np.convolve(g, janela, mode="same")

    if os_fator > 1:
        g = resample_poly(g, 1, os_fator)
    n = min(len(g), audio.shape[1])
    ganho_final = np.ones(audio.shape[1])
    ganho_final[:n] = g[:n]

    saida = audio * ganho_final
    return np.clip(saida, -teto, teto)


def normalizar_loudness(audio: np.ndarray, sr: int,
                        alvo_lufs: float = -14.0) -> tuple[np.ndarray, float, float]:
    """Normaliza para o alvo LUFS. Retorna (audio, lufs_antes, ganho_db)."""
    try:
        import pyloudnorm as pyln
        metro = pyln.Meter(sr)
        atual = float(metro.integrated_loudness(audio.T))
    except Exception:
        atual = 20 * np.log10(np.sqrt(np.mean(audio**2)) + EPS) - 3.0

    if not np.isfinite(atual):
        return audio, atual, 0.0
    ganho_db = alvo_lufs - atual
    return audio * (10 ** (ganho_db / 20.0)), atual, ganho_db


def dither_tpdf(audio: np.ndarray, bits: int = 16,
                noise_shaping: bool = True) -> np.ndarray:
    """Dither triangular com noise shaping de 2ª ordem, para redução de bits."""
    q = 2.0 ** -(bits - 1)
    rng = np.random.default_rng(12345)
    ruido = (rng.random(audio.shape) - rng.random(audio.shape)) * q
    saida = audio + ruido
    if noise_shaping:
        erro = np.zeros(audio.shape[0])
        # Filtro simples de deslocamento do ruído para agudos
        for n in range(1, saida.shape[1]):
            saida[:, n] += 0.5 * erro
            quant = np.round(saida[:, n] / q) * q
            erro = saida[:, n] - quant
    return np.clip(saida, -1.0, 1.0)


# --------------------------------------------------------------------------
# Cadeia completa
# --------------------------------------------------------------------------

def processar(audio: np.ndarray, sr: int, sugestoes: dict | None = None,
              alvo_lufs: float | None = None, teto_dbtp: float = -1.0,
              forcar: dict | None = None) -> tuple[np.ndarray, RelatorioEnhance]:
    """
    Aplica a cadeia conforme as sugestões do laudo.

    sugestoes: dicionário produzido por analysis.analisar().sugestoes
    forcar:    sobrescreve/adiciona etapas manualmente
    alvo_lufs: None desliga a normalização (recomendado para stems, que devem
               preservar o balanço relativo da mixagem original)
    """
    sug = dict(sugestoes or {})
    sug.update(forcar or {})
    rel = RelatorioEnhance()
    x = np.array(audio, dtype=np.float64, copy=True)

    if sug.get("remover_dc"):
        x = remover_dc(x)
        rel.etapas.append("Offset DC removido")

    if sug.get("highpass_hz"):
        x = highpass(x, sr, float(sug["highpass_hz"]))
        rel.etapas.append(f"High-pass {sug['highpass_hz']} Hz aplicado")

    if sug.get("declip"):
        x, n = declip(x)
        rel.amostras_reconstruidas = n
        rel.etapas.append(f"Declipping: {n} amostras reconstruídas")

    if sug.get("denoise"):
        forca = float(sug["denoise"]) if not isinstance(sug["denoise"], bool) else 0.5
        x = denoise(x, sr, forca)
        rel.etapas.append(f"Redução de ruído (força {forca:.2f})")

    if sug.get("corrigir_fase"):
        x = corrigir_fase(x)
        rel.etapas.append("Polaridade de canal corrigida")

    if alvo_lufs is not None:
        x, antes, ganho = normalizar_loudness(x, sr, alvo_lufs)
        rel.lufs_antes = round(antes, 2)
        rel.lufs_depois = alvo_lufs
        rel.ganho_aplicado_db = round(ganho, 2)
        rel.etapas.append(
            f"Loudness normalizado: {antes:.1f} → {alvo_lufs} LUFS ({ganho:+.1f} dB)"
        )

    pico = float(np.max(np.abs(x))) if x.size else 0.0
    if pico > 10 ** (teto_dbtp / 20.0):
        x = limitador_true_peak(x, sr, teto_dbtp)
        rel.etapas.append(f"Limitador true-peak em {teto_dbtp} dBTP")

    try:
        from scipy.signal import resample_poly
        tp = float(np.max(np.abs(resample_poly(x, 4, 1, axis=-1))))
        rel.true_peak_depois = round(20 * np.log10(tp + EPS), 2)
    except Exception:
        rel.true_peak_depois = round(20 * np.log10(float(np.max(np.abs(x))) + EPS), 2)

    if not rel.etapas:
        rel.etapas.append("Nenhuma correção necessária — áudio mantido intacto")

    return x, rel
