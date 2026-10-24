"""
export.py — Escrita dos stems em formatos de áudio e vídeo.

Estratégia de qualidade: o master de cada stem é sempre gravado primeiro em
WAV 32-bit float (sem perda e sem truncamento), e todas as demais versões são
derivadas dele por FFmpeg. Isso evita transcodificação em cascata.

Áudio : wav (16/24/32f), flac, aiff, mp3 (LAME V0/320), aac/m4a, opus, ogg
Vídeo : mp4, mkv, webm, mov — com fundo estático, capa do álbum ou
        visualização gerada (waveform, espectro, barras).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


FORMATOS_AUDIO = {
    "wav":  {"ext": "wav",  "args": ["-c:a", "pcm_s24le"]},
    "wav32": {"ext": "wav", "args": ["-c:a", "pcm_f32le"]},
    "wav16": {"ext": "wav", "args": ["-c:a", "pcm_s16le"]},
    "flac": {"ext": "flac", "args": ["-c:a", "flac", "-compression_level", "8"]},
    "aiff": {"ext": "aiff", "args": ["-c:a", "pcm_s24be"]},
    "mp3":  {"ext": "mp3",  "args": ["-c:a", "libmp3lame", "-b:a", "320k"]},
    "mp3v0": {"ext": "mp3", "args": ["-c:a", "libmp3lame", "-q:a", "0"]},
    "m4a":  {"ext": "m4a",  "args": ["-c:a", "aac", "-b:a", "320k"]},
    "aac":  {"ext": "aac",  "args": ["-c:a", "aac", "-b:a", "320k"]},
    "opus": {"ext": "opus", "args": ["-c:a", "libopus", "-b:a", "256k",
                                     "-vbr", "on"]},
    "ogg":  {"ext": "ogg",  "args": ["-c:a", "libvorbis", "-q:a", "9"]},
}

FORMATOS_VIDEO = {
    "mp4":  {"ext": "mp4",  "video": ["-c:v", "libx264", "-preset", "medium",
                                      "-crf", "18", "-pix_fmt", "yuv420p"],
             "audio": ["-c:a", "aac", "-b:a", "320k"],
             "extra": ["-movflags", "+faststart"]},
    "mkv":  {"ext": "mkv",  "video": ["-c:v", "libx264", "-preset", "medium",
                                      "-crf", "16", "-pix_fmt", "yuv420p"],
             # MKV aceita FLAC: vídeo com áudio sem perda.
             "audio": ["-c:a", "flac", "-compression_level", "8"],
             "extra": []},
    "mov":  {"ext": "mov",  "video": ["-c:v", "libx264", "-preset", "medium",
                                      "-crf", "18", "-pix_fmt", "yuv420p"],
             "audio": ["-c:a", "pcm_s24le"], "extra": []},
    "webm": {"ext": "webm", "video": ["-c:v", "libvpx-vp9", "-crf", "30",
                                      "-b:v", "0"],
             "audio": ["-c:a", "libopus", "-b:a", "256k"], "extra": []},
}

VISUALIZACOES = {
    "waveform": (
        "showwaves=s={w}x{h}:mode=cline:colors=0x7BD5F5|0x1CA7EC:"
        "rate=30,format=yuv420p"
    ),
    "espectro": (
        "showspectrum=s={w}x{h}:mode=combined:color=intensity:scale=cbrt:"
        "slide=scroll,format=yuv420p"
    ),
    "barras": (
        "showfreqs=s={w}x{h}:mode=bar:ascale=log:fscale=log:"
        "colors=0x7BD5F5,format=yuv420p"
    ),
    "vetorscope": "avectorscope=s={w}x{h}:zoom=1.5:draw=line,format=yuv420p",
    "preto": None,
}


@dataclass
class ItemExportado:
    stem: str
    formato: str
    caminho: str
    tamanho_bytes: int


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------

def ffmpeg_disponivel() -> bool:
    return shutil.which("ffmpeg") is not None


def _exec(cmd: Sequence[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"FFmpeg falhou:\n{' '.join(cmd[:12])} ...\n"
            + (proc.stderr or "")[-1200:]
        )


def gravar_master(audio: np.ndarray, sr: int, caminho: str,
                  subtipo: str = "FLOAT") -> str:
    """Grava o master WAV do stem (fonte de todas as conversões)."""
    import soundfile as sf
    os.makedirs(os.path.dirname(caminho) or ".", exist_ok=True)
    dados = np.asarray(audio, dtype=np.float32).T
    sf.write(caminho, dados, sr, subtype=subtipo)
    return caminho


def _metadados(titulo: str, extras: dict | None = None) -> list[str]:
    args = ["-metadata", f"title={titulo}",
            "-metadata", "comment=Gerado por StemForge (Demucs, modelos abertos)",
            "-metadata", "encoder=StemForge"]
    for k, v in (extras or {}).items():
        args += ["-metadata", f"{k}={v}"]
    return args


# --------------------------------------------------------------------------
# Exportação de áudio
# --------------------------------------------------------------------------

def exportar_audio(master_wav: str, destino_dir: str, base: str,
                   formatos: Sequence[str],
                   metadados: dict | None = None) -> list[ItemExportado]:
    saidas: list[ItemExportado] = []
    os.makedirs(destino_dir, exist_ok=True)

    for fmt in formatos:
        cfg = FORMATOS_AUDIO.get(fmt)
        if cfg is None:
            continue
        destino = os.path.join(destino_dir, f"{base}.{cfg['ext']}")
        if fmt == "wav32":
            destino = os.path.join(destino_dir, f"{base}.32f.wav")
        elif fmt == "wav16":
            destino = os.path.join(destino_dir, f"{base}.16.wav")

        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-i", master_wav]
        # Dither do FFmpeg apenas quando há redução real de profundidade.
        if fmt == "wav16":
            cmd += ["-af", "aresample=out_sample_fmt=s16:dither_method=shibata"]
        cmd += cfg["args"] + _metadados(base, metadados) + [destino]
        _exec(cmd)
        saidas.append(ItemExportado(base, fmt, destino,
                                    os.path.getsize(destino)))
    return saidas


# --------------------------------------------------------------------------
# Exportação de vídeo
# --------------------------------------------------------------------------

def exportar_video(master_wav: str, destino_dir: str, base: str,
                   formatos: Sequence[str], visual: str = "waveform",
                   resolucao: str = "1920x1080", fps: int = 30,
                   imagem_fundo: Optional[str] = None,
                   titulo: Optional[str] = None) -> list[ItemExportado]:
    """
    Gera vídeos com o áudio do stem. O visual pode ser uma imagem estática
    (capa) ou uma visualização sintetizada pelo próprio FFmpeg.
    """
    saidas: list[ItemExportado] = []
    os.makedirs(destino_dir, exist_ok=True)
    w, h = resolucao.lower().split("x")

    for fmt in formatos:
        cfg = FORMATOS_VIDEO.get(fmt)
        if cfg is None:
            continue
        destino = os.path.join(destino_dir, f"{base}.{cfg['ext']}")

        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]

        if imagem_fundo and os.path.exists(imagem_fundo):
            cmd += ["-loop", "1", "-framerate", str(fps), "-i", imagem_fundo,
                    "-i", master_wav]
            vf = (f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
                  f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black,format=yuv420p")
            cmd += ["-filter:v", vf, "-map", "0:v", "-map", "1:a", "-shortest"]

        elif visual == "preto" or VISUALIZACOES.get(visual) is None:
            cmd += ["-f", "lavfi", "-i",
                    f"color=c=black:s={w}x{h}:r={fps}",
                    "-i", master_wav,
                    "-map", "0:v", "-map", "1:a", "-shortest"]

        else:
            filtro = VISUALIZACOES[visual].format(w=w, h=h)
            texto = (titulo or base).replace(":", "\\:").replace("'", "")
            filtro += (
                f",drawtext=text='{texto}':fontcolor=white@0.85:fontsize={int(h)//24}"
                f":x=(w-text_w)/2:y=h-(text_h*2.5)"
            )
            cmd += ["-i", master_wav,
                    "-filter_complex", f"[0:a]{filtro}[v]",
                    "-map", "[v]", "-map", "0:a"]

        cmd += cfg["video"] + cfg["audio"] + cfg["extra"]
        cmd += ["-r", str(fps)] + _metadados(titulo or base) + [destino]

        try:
            _exec(cmd)
        except RuntimeError:
            # Alguns builds do FFmpeg não trazem a fonte para drawtext.
            cmd_simples = [c for c in cmd]
            for i, c in enumerate(cmd_simples):
                if isinstance(c, str) and ",drawtext=" in c:
                    cmd_simples[i] = c.split(",drawtext=")[0] + "[v]" \
                        if c.endswith("[v]") else c.split(",drawtext=")[0]
            _exec(cmd_simples)

        saidas.append(ItemExportado(base, fmt, destino,
                                    os.path.getsize(destino)))
    return saidas


def exportar_stem(audio: np.ndarray, sr: int, dir_saida: str, nome: str,
                  formatos_audio: Sequence[str],
                  formatos_video: Sequence[str] = (),
                  visual: str = "waveform",
                  resolucao: str = "1920x1080",
                  imagem_fundo: Optional[str] = None,
                  metadados: dict | None = None,
                  manter_master: bool = True) -> list[ItemExportado]:
    """Grava o master e produz todos os formatos pedidos para um stem."""
    os.makedirs(dir_saida, exist_ok=True)
    master = os.path.join(dir_saida, f"{nome}.master.wav")
    gravar_master(audio, sr, master, subtipo="FLOAT")

    itens: list[ItemExportado] = []
    if ffmpeg_disponivel():
        itens += exportar_audio(master, dir_saida, nome, formatos_audio, metadados)
        if formatos_video:
            itens += exportar_video(master, dir_saida, nome, formatos_video,
                                    visual=visual, resolucao=resolucao,
                                    imagem_fundo=imagem_fundo, titulo=nome)
    else:
        # Sem FFmpeg ainda entregamos WAV/FLAC pelo soundfile.
        import soundfile as sf
        for fmt in formatos_audio:
            if fmt.startswith("wav") or fmt == "flac":
                sub = {"wav": "PCM_24", "wav16": "PCM_16",
                       "wav32": "FLOAT", "flac": "PCM_24"}[fmt]
                ext = "flac" if fmt == "flac" else "wav"
                destino = os.path.join(dir_saida, f"{nome}.{ext}")
                sf.write(destino, np.asarray(audio, dtype=np.float32).T, sr,
                         subtype=sub)
                itens.append(ItemExportado(nome, fmt, destino,
                                           os.path.getsize(destino)))

    if manter_master:
        itens.append(ItemExportado(nome, "master_wav32", master,
                                   os.path.getsize(master)))
    else:
        os.remove(master)
    return itens
