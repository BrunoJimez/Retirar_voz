"""
pipeline.py — Orquestra o fluxo completo:

    hardware → análise → pré-processamento → separação →
    stems derivados → pós-processamento por stem → exportação → relatório
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from . import analysis, enhance, export, hardware, separate


@dataclass
class Config:
    entrada: str = ""
    saida: str = "saida"
    formatos_audio: Sequence[str] = ("wav", "flac", "mp3")
    formatos_video: Sequence[str] = ()
    visual: str = "waveform"
    resolucao: str = "1920x1080"
    imagem_fundo: Optional[str] = None

    qualidade: str = "auto"
    modelo: Optional[str] = None
    device: Optional[str] = None
    seis_stems: bool = False

    melhorar_entrada: bool = True
    melhorar_stems: bool = True
    normalizar_stems: bool = False
    alvo_lufs: float = -14.0
    teto_dbtp: float = -1.0
    denoise_forcado: Optional[float] = None

    derivados: bool = True
    refinar_vocal: bool = False
    manter_master: bool = True
    apenas_stems: Optional[Sequence[str]] = None
    somente_analise: bool = False


def _log(msg: str) -> None:
    print(msg, flush=True)


def executar(cfg: Config) -> dict:
    t0 = time.time()
    relatorio: dict = {"entrada": cfg.entrada, "gerado_em": time.strftime("%Y-%m-%d %H:%M:%S")}

    if not os.path.exists(cfg.entrada):
        raise FileNotFoundError(f"Arquivo não encontrado: {cfg.entrada}")

    # ---------------------------------------------------------------- 1. HW
    perfil = hardware.coletar_hardware()
    modelo = cfg.modelo or ("htdemucs_6s" if cfg.seis_stems else None)
    plano = hardware.planejar_execucao(perfil, cfg.qualidade, modelo, cfg.device)
    _log(hardware.relatorio_hardware(perfil, plano))
    relatorio["hardware"] = perfil.to_dict()
    relatorio["plano"] = plano.to_dict()

    # ------------------------------------------------------------ 2. Análise
    _log("\n▶ Carregando e analisando o áudio...")
    audio, sr, subtipo = analysis.carregar_audio(cfg.entrada)
    laudo = analysis.analisar(cfg.entrada, audio, sr)
    _log(analysis.relatorio_laudo(laudo))
    relatorio["analise_entrada"] = laudo.to_dict()

    duracao_min = laudo.duracao_s / 60
    _log(f"\n  Estimativa de separação: "
         f"~{duracao_min * plano.tempo_estimado_por_minuto_s / 60:.1f} min")

    if cfg.somente_analise:
        relatorio["tempo_total_s"] = round(time.time() - t0, 1)
        _salvar_relatorio(cfg, relatorio)
        return relatorio

    # ------------------------------------------- 3. Pré-processamento da fonte
    if cfg.melhorar_entrada:
        _log("\n▶ Condicionando a fonte antes da separação...")
        sug = dict(laudo.sugestoes)
        sug.pop("limitar_true_peak", None)   # limitar antes da separação prejudica
        if cfg.denoise_forcado is not None:
            sug["denoise"] = cfg.denoise_forcado
        # Nunca normalizamos a fonte: alteraria o balanço que o modelo aprendeu.
        audio, rel_pre = enhance.processar(audio, sr, sug, alvo_lufs=None,
                                           teto_dbtp=0.0)
        for e in rel_pre.etapas:
            _log(f"    · {e}")
        relatorio["preprocessamento"] = rel_pre.to_dict()

    # --------------------------------------------------------- 4. Separação
    _log(f"\n▶ Separando com {plano.modelo} em {plano.device}...")
    t_sep = time.time()
    res = separate.separar(audio, sr, plano, caminho_original=cfg.entrada)
    for a in res.aviso:
        _log(f"    ! {a}")
    _log(f"    Concluído em {time.time() - t_sep:.1f} s · "
         f"resíduo da reconstrução: {res.residuo_db} dB")
    relatorio["separacao"] = {
        "modelo": res.modelo, "device": res.device,
        "sample_rate": res.sample_rate,
        "residuo_db": res.residuo_db,
        "stems_modelo": sorted(res.stems),
        "avisos": res.aviso,
        "tempo_s": round(time.time() - t_sep, 1),
    }

    stems = dict(res.stems)
    sr_out = res.sample_rate

    # ------------------------------------------------- 5. Refino do vocal
    if cfg.refinar_vocal and "vocals" in stems:
        _log("\n▶ Refinando a faixa vocal (segunda passagem UVR/MDX-Net)...")
        refinado = separate.refinar_vocal(stems["vocals"], sr_out, plano)
        if refinado is not None:
            n = min(refinado.shape[1], stems["vocals"].shape[1])
            stems["vocals"] = refinado[:, :n]
            _log("    · Vocal refinado.")
        else:
            _log("    · audio-separator indisponível; refino ignorado "
                 "(pip install audio-separator).")

    # ---------------------------------------------------- 6. Stems derivados
    if cfg.derivados:
        extras = separate.derivar_stems(stems)
        stems.update(extras)
        _log(f"\n▶ Stems derivados: {', '.join(sorted(extras))}")

    if cfg.apenas_stems:
        pedidos = set(cfg.apenas_stems)
        stems = {k: v for k, v in stems.items() if k in pedidos}
        if not stems:
            raise ValueError(
                f"Nenhum dos stems pedidos existe. Disponíveis: "
                f"{sorted(set(res.stems) | set(separate.derivar_stems(res.stems)))}"
            )

    # ------------------------------------------- 7. Pós-processamento e export
    _log(f"\n▶ Processando e exportando {len(stems)} stems...")
    os.makedirs(cfg.saida, exist_ok=True)
    base_nome = os.path.splitext(os.path.basename(cfg.entrada))[0]
    dir_projeto = os.path.join(cfg.saida, base_nome)
    os.makedirs(dir_projeto, exist_ok=True)

    relatorio["stems"] = {}
    arquivos_gerados: list[str] = []

    for nome, dados in sorted(stems.items()):
        rotulo = separate.NOMES_PT.get(nome, nome)
        info: dict = {"nome_modelo": nome}

        if cfg.melhorar_stems:
            laudo_stem = analysis.analisar(nome, dados, sr_out, rapido=True)
            sug = dict(laudo_stem.sugestoes)
            # Denoise agressivo em stems apaga cauda de reverb; limitamos a força.
            if "denoise" in sug and not isinstance(sug["denoise"], bool):
                sug["denoise"] = min(float(sug["denoise"]), 0.35)
            sug.pop("origem_lossy", None)
            dados_proc, rel = enhance.processar(
                dados, sr_out, sug,
                alvo_lufs=cfg.alvo_lufs if cfg.normalizar_stems else None,
                teto_dbtp=cfg.teto_dbtp,
            )
            info["analise"] = laudo_stem.to_dict()
            info["processamento"] = rel.to_dict()
        else:
            dados_proc = dados

        itens = export.exportar_stem(
            dados_proc, sr_out, dir_projeto, rotulo,
            formatos_audio=cfg.formatos_audio,
            formatos_video=cfg.formatos_video,
            visual=cfg.visual,
            resolucao=cfg.resolucao,
            imagem_fundo=cfg.imagem_fundo,
            metadados={"album": base_nome, "track": rotulo},
            manter_master=cfg.manter_master,
        )
        info["arquivos"] = [
            {"formato": i.formato, "caminho": i.caminho, "bytes": i.tamanho_bytes}
            for i in itens
        ]
        arquivos_gerados += [i.caminho for i in itens]
        relatorio["stems"][rotulo] = info
        _log(f"    ✓ {rotulo:<16} {len(itens)} arquivo(s)")

    relatorio["tempo_total_s"] = round(time.time() - t0, 1)
    relatorio["diretorio_saida"] = dir_projeto
    relatorio["total_arquivos"] = len(arquivos_gerados)
    _salvar_relatorio(cfg, relatorio, dir_projeto)

    _log(f"\n✔ Finalizado em {relatorio['tempo_total_s']} s · "
         f"{len(arquivos_gerados)} arquivos em {dir_projeto}")
    return relatorio


def _salvar_relatorio(cfg: Config, relatorio: dict,
                      dir_projeto: Optional[str] = None) -> None:
    destino_dir = dir_projeto or cfg.saida
    os.makedirs(destino_dir, exist_ok=True)
    caminho = os.path.join(destino_dir, "relatorio.json")
    with open(caminho, "w", encoding="utf-8") as fh:
        json.dump(relatorio, fh, indent=2, ensure_ascii=False, default=str)
    _log(f"    Relatório técnico: {caminho}")
