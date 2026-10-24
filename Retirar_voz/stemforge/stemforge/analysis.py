"""
analysis.py — Medição técnica do áudio.

Produz um laudo objetivo (loudness EBU R128, true peak, clipping, ruído de
fundo, largura de banda efetiva, correlação estéreo, BPM, tonalidade) e, a
partir dele, uma lista de correções sugeridas para o módulo de enhancement.

Todas as medidas usam bibliotecas livres: numpy, scipy, soundfile, librosa,
pyloudnorm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np


EPS = 1e-12


# --------------------------------------------------------------------------
# Estruturas
# --------------------------------------------------------------------------

@dataclass
class Laudo:
    arquivo: str = ""
    sample_rate: int = 0
    canais: int = 0
    duracao_s: float = 0.0
    profundidade_origem: str = "desconhecida"

    pico_dbfs: float = 0.0
    true_peak_dbtp: float = 0.0
    rms_dbfs: float = 0.0
    lufs_integrado: float = 0.0
    lra: float = 0.0
    fator_crista_db: float = 0.0

    dc_offset: float = 0.0
    amostras_clipadas: int = 0
    taxa_clipping: float = 0.0
    piso_ruido_dbfs: float = -120.0
    snr_estimada_db: float = 0.0
    ruido_confiavel: bool = True

    banda_efetiva_hz: float = 0.0
    provavel_origem_lossy: bool = False
    centroide_espectral_hz: float = 0.0
    energia_sub_30hz_db: float = -120.0

    correlacao_estereo: float = 0.0
    largura_estereo: float = 0.0
    mono_compativel: bool = True

    bpm: float = 0.0
    tonalidade: str = "N/A"
    confianca_tonalidade: float = 0.0

    problemas: list = field(default_factory=list)
    sugestoes: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Leitura
# --------------------------------------------------------------------------

def carregar_audio(caminho: str, sr_alvo: Optional[int] = None):
    """
    Carrega o áquivo preservando os canais e retorna (audio, sr, subtipo).
    audio: ndarray float64 no formato (canais, amostras).
    """
    import soundfile as sf

    try:
        info = sf.info(caminho)
        subtipo = f"{info.subtype} @ {info.format}"
    except Exception:
        subtipo = "desconhecida"

    try:
        dados, sr = sf.read(caminho, always_2d=True, dtype="float64")
        audio = dados.T
    except Exception:
        # Formatos comprimidos que o libsndfile não abre (ex.: alguns m4a/wma)
        import librosa
        audio, sr = librosa.load(caminho, sr=None, mono=False)
        audio = np.atleast_2d(np.asarray(audio, dtype=np.float64))
        subtipo = "decodificado via ffmpeg/librosa"

    if sr_alvo and sr != sr_alvo:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(int(sr_alvo), int(sr))
        audio = resample_poly(audio, int(sr_alvo) // g, int(sr) // g, axis=-1)
        sr = sr_alvo

    return np.ascontiguousarray(audio), int(sr), subtipo


# --------------------------------------------------------------------------
# Medidas individuais
# --------------------------------------------------------------------------

def _db(x: float) -> float:
    return 20.0 * math.log10(max(abs(x), EPS))


def true_peak_dbtp(audio: np.ndarray, sr: int, oversample: int = 4) -> float:
    """True peak por sobreamostragem (aproximação do ITU-R BS.1770-4)."""
    from scipy.signal import resample_poly
    try:
        up = resample_poly(audio, oversample, 1, axis=-1)
    except Exception:
        up = audio
    return _db(float(np.max(np.abs(up))))


def medir_loudness(audio: np.ndarray, sr: int) -> tuple[float, float]:
    """LUFS integrado e Loudness Range (EBU R128 via pyloudnorm)."""
    try:
        import pyloudnorm as pyln
        metro = pyln.Meter(sr)
        lufs = float(metro.integrated_loudness(audio.T))
    except Exception:
        rms = float(np.sqrt(np.mean(audio**2) + EPS))
        lufs = _db(rms) - 3.0  # aproximação grosseira

    # LRA aproximado: dispersão dos blocos curtos (percentis 10/95)
    try:
        bloco = int(sr * 3.0)
        salto = int(sr * 1.0)
        mono = audio.mean(axis=0)
        niveis = []
        for i in range(0, max(1, len(mono) - bloco), salto):
            seg = mono[i:i + bloco]
            if seg.size:
                niveis.append(_db(float(np.sqrt(np.mean(seg**2) + EPS))))
        niveis = [n for n in niveis if n > -70]
        lra = float(np.percentile(niveis, 95) - np.percentile(niveis, 10)) if niveis else 0.0
    except Exception:
        lra = 0.0

    return round(lufs, 2), round(lra, 2)


def detectar_clipping(audio: np.ndarray, limiar: float = 0.9995) -> tuple[int, float]:
    """Conta amostras no teto digital (indício de clipping)."""
    excedem = np.abs(audio) >= limiar
    total = int(excedem.sum())
    return total, round(total / max(audio.size, 1), 8)


def piso_de_ruido(audio: np.ndarray, sr: int) -> tuple[float, float, bool]:
    """
    Estima o piso de ruído por estatística de mínimos espectrais: para cada
    bin de frequência, toma o percentil 5 ao longo do tempo. O sinal musical
    é intermitente em cada bin, o ruído estacionário não — logo o percentil
    baixo isola o ruído mesmo em faixas sem trechos silenciosos.

    A estimativa só é confiável quando o material é não-estacionário (música
    real). Sinais sustentados sem variação — tons de teste, drones — devolvem
    confiavel=False para não disparar redução de ruído indevida.

    Retorna (piso_dbfs, snr_db, confiavel).
    """
    from scipy.signal import stft

    mono = audio.mean(axis=0)
    if len(mono) < 8192:
        return -120.0, 0.0, False
    try:
        _, _, Z = stft(mono[: sr * 90], fs=sr, nperseg=2048, noverlap=1024)
    except Exception:
        return -120.0, 0.0, False

    pot = np.abs(Z) ** 2
    if pot.shape[1] < 8:
        return -120.0, 0.0, False

    # Estacionariedade: coeficiente de variação da energia por quadro.
    energia = np.sum(pot, axis=0)
    cv = float(np.std(energia) / (np.mean(energia) + EPS))
    confiavel = cv > 0.20

    piso_bin = np.percentile(pot, 5, axis=1)
    pot_ruido = float(np.sum(piso_bin))
    pot_total = float(np.mean(np.sum(pot, axis=0)))

    piso_db = 10 * math.log10(max(pot_ruido, EPS) /
                              max(pot.shape[0], 1) + EPS) / 1.0
    # Converte para escala dBFS comparável ao RMS do sinal.
    escala = _db(float(np.sqrt(np.mean(mono**2) + EPS))) - \
        10 * math.log10(max(pot_total, EPS) + EPS)
    piso_dbfs = piso_db + escala
    snr = 10 * math.log10(max(pot_total, EPS) / max(pot_ruido, EPS))
    return (round(float(np.clip(piso_dbfs, -140, 0)), 2),
            round(float(snr), 2), confiavel)


def banda_efetiva(audio: np.ndarray, sr: int) -> tuple[float, bool]:
    """
    Encontra a frequência acima da qual a energia despenca (corte típico de
    codecs com perda). Retorna (hz, provavelmente_lossy).
    """
    mono = audio.mean(axis=0)
    n = min(len(mono), sr * 60)
    if n < 4096:
        return float(sr / 2), False
    from scipy.signal import welch
    f, pxx = welch(mono[:n], fs=sr, nperseg=8192)
    pxx_db = 10 * np.log10(pxx + EPS)

    # Suaviza (~200 Hz) para não confundir vale entre parciais com corte real.
    passo = max(1, int(200 / (f[1] - f[0])))
    kernel = np.ones(passo) / passo
    suave = np.convolve(pxx_db, kernel, mode="same")

    # Referência: nível mediano da região de médios, imune a picos tonais.
    faixa = (f >= 1000) & (f <= 5000)
    ref = float(np.median(suave[faixa])) if faixa.any() else float(np.max(suave))

    acima = np.where(suave > ref - 40)[0]
    nyq = sr / 2
    corte = float(f[acima[-1]]) if acima.size else nyq

    # Codec lossy deixa uma parede: queda abrupta logo acima do corte.
    queda_abrupta = False
    if acima.size:
        i = acima[-1]
        j = min(len(suave) - 1, i + max(1, int(1000 / (f[1] - f[0]))))
        queda_abrupta = (suave[i] - suave[j]) > 20

    lossy = bool(corte < nyq * 0.92 and corte < 20500 and queda_abrupta)
    return round(corte, 1), lossy


def analise_estereo(audio: np.ndarray) -> tuple[float, float, bool]:
    if audio.shape[0] < 2:
        return 1.0, 0.0, True
    l, r = audio[0], audio[1]
    denom = math.sqrt(float(np.sum(l**2)) * float(np.sum(r**2))) + EPS
    corr = float(np.sum(l * r) / denom)
    mid = (l + r) / 2
    side = (l - r) / 2
    largura = float(np.sqrt(np.mean(side**2) + EPS) /
                    (np.sqrt(np.mean(mid**2) + EPS)))
    return round(corr, 4), round(largura, 4), bool(corr > -0.2)


def ritmo_e_tom(audio: np.ndarray, sr: int) -> tuple[float, str, float]:
    """BPM (librosa) e tonalidade por correlação com perfis Krumhansl-Schmuckler."""
    try:
        import librosa
    except Exception:
        return 0.0, "N/A", 0.0

    mono = audio.mean(axis=0).astype(np.float32)
    if sr != 22050:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=22050)
        sr_a = 22050
    else:
        sr_a = sr

    try:
        bpm, _ = librosa.beat.beat_track(y=mono, sr=sr_a)
        bpm = float(np.atleast_1d(bpm)[0])
    except Exception:
        bpm = 0.0

    try:
        chroma = librosa.feature.chroma_cqt(y=mono, sr=sr_a)
        perfil = chroma.mean(axis=1)
        perfil = (perfil - perfil.mean()) / (perfil.std() + EPS)

        maior = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                          2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        menor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                          2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        maior = (maior - maior.mean()) / maior.std()
        menor = (menor - menor.mean()) / menor.std()

        notas = ["C", "C#", "D", "D#", "E", "F",
                 "F#", "G", "G#", "A", "A#", "B"]
        melhor, pontuacao = "N/A", -9.0
        for i in range(12):
            rot = np.roll(perfil, -i)
            for modelo, sufixo in ((maior, "maior"), (menor, "menor")):
                s = float(np.corrcoef(rot, modelo)[0, 1])
                if s > pontuacao:
                    pontuacao, melhor = s, f"{notas[i]} {sufixo}"
        return round(bpm, 1), melhor, round(max(0.0, pontuacao), 3)
    except Exception:
        return round(bpm, 1), "N/A", 0.0


# --------------------------------------------------------------------------
# Laudo completo
# --------------------------------------------------------------------------

def analisar(caminho: str, audio: Optional[np.ndarray] = None,
             sr: Optional[int] = None, rapido: bool = False) -> Laudo:
    """Executa toda a bateria de medidas e monta o laudo com sugestões."""
    if audio is None or sr is None:
        audio, sr, subtipo = carregar_audio(caminho)
    else:
        subtipo = "em memória"

    ld = Laudo(
        arquivo=caminho,
        sample_rate=sr,
        canais=int(audio.shape[0]),
        duracao_s=round(audio.shape[1] / sr, 3),
        profundidade_origem=subtipo,
    )

    ld.pico_dbfs = round(_db(float(np.max(np.abs(audio)))), 2)
    ld.true_peak_dbtp = round(true_peak_dbtp(audio, sr), 2)
    ld.rms_dbfs = round(_db(float(np.sqrt(np.mean(audio**2) + EPS))), 2)
    ld.lufs_integrado, ld.lra = medir_loudness(audio, sr)
    ld.fator_crista_db = round(ld.pico_dbfs - ld.rms_dbfs, 2)

    ld.dc_offset = round(float(np.mean(audio)), 6)
    ld.amostras_clipadas, ld.taxa_clipping = detectar_clipping(audio)
    ld.piso_ruido_dbfs, ld.snr_estimada_db, ld.ruido_confiavel = \
        piso_de_ruido(audio, sr)
    ld.banda_efetiva_hz, ld.provavel_origem_lossy = banda_efetiva(audio, sr)
    ld.correlacao_estereo, ld.largura_estereo, ld.mono_compativel = analise_estereo(audio)

    try:
        from scipy.signal import welch
        mono = audio.mean(axis=0)
        f, pxx = welch(mono[: sr * 60], fs=sr, nperseg=8192)
        ld.centroide_espectral_hz = round(
            float(np.sum(f * pxx) / (np.sum(pxx) + EPS)), 1)
        sub = pxx[f < 30]
        ld.energia_sub_30hz_db = round(
            10 * math.log10(float(np.sum(sub)) + EPS) -
            10 * math.log10(float(np.sum(pxx)) + EPS), 2)
    except Exception:
        pass

    if not rapido:
        ld.bpm, ld.tonalidade, ld.confianca_tonalidade = ritmo_e_tom(audio, sr)

    _diagnosticar(ld)
    return ld


def _diagnosticar(ld: Laudo) -> None:
    """Preenche ld.problemas e ld.sugestoes (parâmetros para o enhancer)."""
    p, s = ld.problemas, ld.sugestoes

    if abs(ld.dc_offset) > 1e-4:
        p.append(f"Offset DC de {ld.dc_offset:+.5f} — desperdiça headroom.")
        s["remover_dc"] = True

    if ld.taxa_clipping > 1e-6:
        p.append(
            f"{ld.amostras_clipadas} amostras no teto digital "
            f"({ld.taxa_clipping*100:.4f}%) — clipping na origem."
        )
        s["declip"] = True

    if ld.true_peak_dbtp > -0.1:
        p.append(
            f"True peak de {ld.true_peak_dbtp} dBTP — risco de distorção "
            "intersample ao codificar em mp3/aac."
        )
        s["limitar_true_peak"] = -1.0

    if ld.ruido_confiavel and ld.snr_estimada_db < 40 and ld.piso_ruido_dbfs > -60:
        p.append(
            f"Piso de ruído alto ({ld.piso_ruido_dbfs} dBFS, "
            f"SNR ~{ld.snr_estimada_db} dB)."
        )
        s["denoise"] = min(0.85, max(0.25, (45 - ld.snr_estimada_db) / 45))

    if ld.provavel_origem_lossy:
        p.append(
            f"Banda efetiva de {ld.banda_efetiva_hz/1000:.1f} kHz — fonte "
            "provavelmente comprimida com perda. A separação herda esse limite."
        )
        s["origem_lossy"] = True

    if ld.energia_sub_30hz_db > -35:
        p.append("Energia excessiva abaixo de 30 Hz (rumble/DC inaudível).")
        s["highpass_hz"] = 22

    if ld.fator_crista_db < 6:
        p.append(
            f"Fator de crista de {ld.fator_crista_db} dB — masterização muito "
            "comprimida; a separação tende a gerar mais artefatos."
        )

    if not ld.mono_compativel:
        p.append(
            f"Correlação estéreo de {ld.correlacao_estereo} — cancelamento "
            "em soma mono."
        )
        s["corrigir_fase"] = True

    if ld.lufs_integrado > -6:
        p.append(f"Loudness de {ld.lufs_integrado} LUFS — acima do usual.")

    if not p:
        p.append("Nenhum problema técnico relevante detectado.")


def relatorio_laudo(ld: Laudo) -> str:
    linhas = [
        "═" * 62,
        " ANÁLISE DO ÁUDIO",
        "═" * 62,
        f"  Arquivo........: {ld.arquivo}",
        f"  Formato........: {ld.sample_rate} Hz · {ld.canais} canal(is) · "
        f"{ld.duracao_s:.1f} s · {ld.profundidade_origem}",
        "",
        f"  Pico...........: {ld.pico_dbfs} dBFS   True peak: {ld.true_peak_dbtp} dBTP",
        f"  RMS............: {ld.rms_dbfs} dBFS   Crista: {ld.fator_crista_db} dB",
        f"  Loudness.......: {ld.lufs_integrado} LUFS   LRA: {ld.lra} LU",
        f"  Ruído..........: piso {ld.piso_ruido_dbfs} dBFS   "
        f"SNR ~{ld.snr_estimada_db} dB"
        + ("" if ld.ruido_confiavel else "  (estimativa não confiável)"),
        f"  Clipping.......: {ld.amostras_clipadas} amostras "
        f"({ld.taxa_clipping*100:.4f}%)",
        f"  DC offset......: {ld.dc_offset:+.6f}",
        f"  Banda efetiva..: {ld.banda_efetiva_hz/1000:.2f} kHz"
        + ("  (fonte lossy)" if ld.provavel_origem_lossy else ""),
        f"  Estéreo........: correlação {ld.correlacao_estereo} · "
        f"largura {ld.largura_estereo}",
        f"  Musical........: {ld.bpm} BPM · tonalidade {ld.tonalidade} "
        f"(conf. {ld.confianca_tonalidade})",
        "",
        "  Diagnóstico:",
    ]
    linhas += [f"    · {x}" for x in ld.problemas]
    linhas.append("═" * 62)
    return "\n".join(linhas)
