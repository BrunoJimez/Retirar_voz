"""
separate.py — Separação de fontes com modelos abertos.

Backend principal: Demucs v4 (Meta AI, licença MIT, pesos livres).
    htdemucs      → vocals, drums, bass, other        (4 stems, rápido)
    htdemucs_ft   → idem, fine-tuned                  (4 stems, melhor SDR)
    htdemucs_6s   → + guitar, piano                   (6 stems)
    mdx_extra     → treinado com dados extras         (4 stems)

Backend opcional: audio-separator (modelos UVR/MDX-Net, também livres),
usado para refinar a faixa vocal quando o usuário pede qualidade máxima.

Além dos stems do modelo, o módulo deriva combinações úteis:
    instrumental / karaokê, seção rítmica, harmônicos, acapella,
    e a decomposição mid/side de qualquer stem.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


MODELOS = {
    "htdemucs":     ("vocals", "drums", "bass", "other"),
    "htdemucs_ft":  ("vocals", "drums", "bass", "other"),
    "htdemucs_6s":  ("vocals", "drums", "bass", "guitar", "piano", "other"),
    "mdx_extra":    ("vocals", "drums", "bass", "other"),
    "mdx_extra_q":  ("vocals", "drums", "bass", "other"),
}

NOMES_PT = {
    "vocals": "vocal",
    "drums": "bateria",
    "bass": "baixo",
    "guitar": "guitarra",
    "piano": "piano",
    "other": "outros",
    "instrumental": "instrumental",
    "karaoke": "karaoke",
    "rhythm": "ritmica",
    "harmonic": "harmonica",
    "backing": "acompanhamento",
    "vocals_mid": "vocal_centro",
    "vocals_side": "vocal_lateral",
}


@dataclass
class ResultadoSeparacao:
    stems: dict = field(default_factory=dict)   # nome -> ndarray (canais, amostras)
    sample_rate: int = 44100
    modelo: str = ""
    device: str = ""
    residuo_db: float = 0.0
    aviso: list = field(default_factory=list)


# --------------------------------------------------------------------------
# Núcleo
# --------------------------------------------------------------------------

def _separar_via_api(audio: np.ndarray, sr: int, plano) -> tuple[dict, int]:
    """Usa demucs.api.Separator (Demucs >= 4.0.1)."""
    import torch
    from demucs.api import Separator

    kwargs = dict(
        model=plano.modelo,
        device=plano.device,
        shifts=plano.shifts,
        overlap=plano.overlap,
        split=True,
        progress=True,
    )
    if plano.segment:
        kwargs["segment"] = plano.segment
    if plano.device == "cpu":
        kwargs["jobs"] = plano.jobs

    separador = Separator(**kwargs)
    sr_modelo = separador.samplerate

    tensor = torch.from_numpy(np.asarray(audio, dtype=np.float32))
    if tensor.shape[0] == 1:                       # mono → estéreo
        tensor = tensor.repeat(2, 1)
    elif tensor.shape[0] > 2:
        tensor = tensor[:2]

    if sr != sr_modelo:
        import torchaudio
        tensor = torchaudio.functional.resample(tensor, sr, sr_modelo)

    _, saidas = separador.separate_tensor(tensor, sr_modelo)
    stems = {nome: t.detach().cpu().numpy().astype(np.float64)
             for nome, t in saidas.items()}
    return stems, sr_modelo


def _separar_via_cli(caminho_entrada: str, plano, tmpdir: str) -> tuple[dict, int]:
    """Fallback: chama `python -m demucs` e lê os WAVs gerados."""
    import soundfile as sf

    cmd = [
        sys.executable, "-m", "demucs",
        "-n", plano.modelo,
        "-d", plano.device,
        "--shifts", str(plano.shifts),
        "--overlap", str(plano.overlap),
        "-o", tmpdir,
        "--filename", "{stem}.wav",
        caminho_entrada,
    ]
    if plano.segment:
        cmd += ["--segment", str(plano.segment)]
    if plano.device == "cpu" and plano.jobs > 1:
        cmd += ["-j", str(plano.jobs)]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Falha ao executar o Demucs pela linha de comando:\n"
            + (proc.stderr or "")[-1500:]
        )

    stems, sr = {}, 44100
    for raiz, _, arquivos in os.walk(tmpdir):
        for arq in arquivos:
            if arq.endswith(".wav"):
                nome = os.path.splitext(arq)[0]
                dados, sr = sf.read(os.path.join(raiz, arq),
                                    always_2d=True, dtype="float64")
                stems[nome] = dados.T
    if not stems:
        raise RuntimeError("O Demucs não produziu nenhum stem.")
    return stems, sr


def separar(audio: np.ndarray, sr: int, plano,
            caminho_original: Optional[str] = None) -> ResultadoSeparacao:
    """
    Executa a separação com fallback automático:
      API do Demucs → CLI do Demucs → erro instrutivo.
    Em caso de OOM na GPU, reduz o segmento e tenta de novo; persistindo,
    cai para CPU.
    """
    res = ResultadoSeparacao(modelo=plano.modelo, device=plano.device)
    tentativas = 0

    while True:
        try:
            stems, sr_out = _separar_via_api(audio, sr, plano)
            break
        except ImportError as e:
            if caminho_original and shutil.which(sys.executable):
                with tempfile.TemporaryDirectory() as td:
                    stems, sr_out = _separar_via_cli(caminho_original, plano, td)
                break
            raise RuntimeError(
                "Demucs não está instalado. Rode: pip install -U demucs"
            ) from e
        except Exception as e:
            msg = str(e).lower()
            oom = "out of memory" in msg or "cuda error" in msg or "alloc" in msg
            if oom and tentativas < 3:
                tentativas += 1
                atual = plano.segment or 15
                plano.segment = max(4, int(atual * 0.6))
                res.aviso.append(
                    f"Memória insuficiente; segmento reduzido para "
                    f"{plano.segment} s (tentativa {tentativas})."
                )
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                continue
            if oom and plano.device != "cpu":
                res.aviso.append("Sem memória na GPU; alternando para CPU.")
                plano.device = "cpu"
                plano.segment = 10
                tentativas = 0
                continue
            raise

    res.stems = stems
    res.sample_rate = sr_out
    res.device = plano.device

    # Métrica de sanidade: quanto da mixagem não foi explicado pelos stems.
    try:
        soma = sum(stems.values())
        ref = audio
        if ref.shape[0] == 1:
            ref = np.repeat(ref, 2, axis=0)
        n = min(soma.shape[1], ref.shape[1])
        residuo = ref[:, :n] - soma[:, :n]
        pot_ref = float(np.mean(ref[:, :n] ** 2)) + 1e-12
        pot_res = float(np.mean(residuo ** 2)) + 1e-12
        res.residuo_db = round(10 * np.log10(pot_res / pot_ref), 2)
    except Exception:
        pass

    return res


# --------------------------------------------------------------------------
# Stems derivados
# --------------------------------------------------------------------------

def derivar_stems(stems: dict, incluir_midside: bool = True) -> dict:
    """
    Cria as combinações que não vêm do modelo mas são as mais pedidas
    na prática (instrumental, karaokê, base rítmica, etc.).
    """
    extras = {}
    nomes = set(stems)

    def soma(*chaves):
        presentes = [stems[k] for k in chaves if k in stems]
        if not presentes:
            return None
        n = min(a.shape[1] for a in presentes)
        return sum(a[:, :n] for a in presentes)

    if "vocals" in nomes and len(nomes) > 1:
        instrumental = soma(*[k for k in nomes if k != "vocals"])
        if instrumental is not None:
            extras["instrumental"] = instrumental
            extras["karaoke"] = instrumental  # alias explícito para uso musical

    ritmo = soma("drums", "bass")
    if ritmo is not None:
        extras["rhythm"] = ritmo

    harm = soma(*[k for k in ("guitar", "piano", "other") if k in nomes])
    if harm is not None and len(nomes & {"guitar", "piano", "other"}) > 1:
        extras["harmonic"] = harm

    if incluir_midside and "vocals" in nomes and stems["vocals"].shape[0] >= 2:
        v = stems["vocals"]
        mid = (v[0] + v[1]) / 2
        side = (v[0] - v[1]) / 2
        # Mid ≈ voz principal (centralizada); Side ≈ dobras, backing, reverb.
        extras["vocals_mid"] = np.stack([mid, mid])
        extras["vocals_side"] = np.stack([side, -side])

    return extras


def refinar_vocal(vocal: np.ndarray, sr: int, plano) -> Optional[np.ndarray]:
    """
    Segunda passagem opcional com audio-separator (modelos UVR/MDX-Net, livres)
    para remover resíduo instrumental da faixa vocal. Retorna None se a
    biblioteca não estiver instalada.
    """
    try:
        import soundfile as sf
        from audio_separator.separator import Separator as UVR
    except Exception:
        return None

    with tempfile.TemporaryDirectory() as td:
        entrada = os.path.join(td, "vocal.wav")
        sf.write(entrada, vocal.T, sr, subtype="FLOAT")
        try:
            uvr = UVR(output_dir=td, output_format="WAV",
                      use_autocast=(plano.device == "cuda"))
            uvr.load_model(model_filename="UVR-MDX-NET-Voc_FT.onnx")
            arquivos = uvr.separate(entrada)
        except Exception:
            return None

        for arq in arquivos:
            caminho = arq if os.path.isabs(arq) else os.path.join(td, arq)
            if "vocal" in os.path.basename(caminho).lower() and os.path.exists(caminho):
                dados, _ = sf.read(caminho, always_2d=True, dtype="float64")
                return dados.T
    return None
