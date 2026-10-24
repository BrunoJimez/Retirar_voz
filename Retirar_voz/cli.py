"""
cli.py — Interface de linha de comando do StemForge.

Exemplos:
    stemforge musica.mp3
    stemforge musica.flac --6-stems --qualidade maxima --formatos wav flac mp3
    stemforge musica.wav --video mp4 mkv --visual espectro
    stemforge musica.mp3 --apenas vocals instrumental --formatos mp3
    stemforge --diagnostico
    stemforge musica.mp3 --somente-analise
"""

from __future__ import annotations

import argparse
import json
import sys

from . import export, hardware, separate
from .pipeline import Config, executar


def construir_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="stemforge",
        description="Separa vocal, instrumentos e demais componentes de uma "
                    "música usando modelos de IA abertos e gratuitos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("entrada", nargs="?", help="Arquivo de música de entrada")
    p.add_argument("-o", "--saida", default="saida",
                   help="Diretório de saída (padrão: ./saida)")

    g = p.add_argument_group("modelo e desempenho")
    g.add_argument("--qualidade", choices=["rapido", "equilibrado", "maxima", "auto"],
                   default="auto", help="Preset de qualidade (padrão: auto)")
    g.add_argument("--modelo", choices=sorted(separate.MODELOS),
                   help="Força um modelo específico do Demucs")
    g.add_argument("--6-stems", dest="seis_stems", action="store_true",
                   help="Usa htdemucs_6s (separa também guitarra e piano)")
    g.add_argument("--device", choices=["cuda", "mps", "cpu"],
                   help="Força o dispositivo de execução")
    g.add_argument("--refinar-vocal", action="store_true",
                   help="Segunda passagem no vocal com modelos UVR/MDX-Net")

    g = p.add_argument_group("formatos de saída")
    g.add_argument("--formatos", nargs="+", default=["wav", "flac", "mp3"],
                   choices=sorted(export.FORMATOS_AUDIO),
                   help="Formatos de áudio (padrão: wav flac mp3)")
    g.add_argument("--video", nargs="*", default=[],
                   choices=sorted(export.FORMATOS_VIDEO),
                   help="Formatos de vídeo para cada stem (ex.: mp4 mkv)")
    g.add_argument("--visual", default="waveform",
                   choices=sorted(export.VISUALIZACOES),
                   help="Visualização usada nos vídeos (padrão: waveform)")
    g.add_argument("--resolucao", default="1920x1080",
                   help="Resolução do vídeo (padrão: 1920x1080)")
    g.add_argument("--capa", dest="imagem_fundo",
                   help="Imagem de fundo/capa para os vídeos")
    g.add_argument("--apenas", nargs="+", dest="apenas_stems",
                   help="Exporta somente os stems listados "
                        "(vocals drums bass guitar piano other instrumental "
                        "karaoke rhythm harmonic vocals_mid vocals_side)")
    g.add_argument("--sem-derivados", dest="derivados", action="store_false",
                   help="Não gera instrumental/karaokê/rítmica/etc.")
    g.add_argument("--sem-master", dest="manter_master", action="store_false",
                   help="Não mantém o WAV 32-bit float de cada stem")

    g = p.add_argument_group("processamento de áudio")
    g.add_argument("--sem-melhoria-entrada", dest="melhorar_entrada",
                   action="store_false", help="Não condiciona a fonte")
    g.add_argument("--sem-melhoria-stems", dest="melhorar_stems",
                   action="store_false", help="Não processa os stems")
    g.add_argument("--normalizar", dest="normalizar_stems", action="store_true",
                   help="Normaliza cada stem para o alvo LUFS "
                        "(desligado por padrão: preserva o balanço da mixagem)")
    g.add_argument("--lufs", type=float, default=-14.0,
                   help="Alvo de loudness em LUFS (padrão: -14)")
    g.add_argument("--teto-dbtp", type=float, default=-1.0,
                   help="Teto de true peak em dBTP (padrão: -1.0)")
    g.add_argument("--denoise", type=float, metavar="0.0-0.95",
                   help="Força a redução de ruído na fonte")

    g = p.add_argument_group("modos alternativos")
    g.add_argument("--diagnostico", action="store_true",
                   help="Apenas inspeciona o hardware e sai")
    g.add_argument("--somente-analise", action="store_true",
                   help="Apenas analisa o áudio, sem separar")
    g.add_argument("--json", action="store_true",
                   help="Imprime o relatório final em JSON no stdout")
    return p


def main(argv: list[str] | None = None) -> int:
    args = construir_parser().parse_args(argv)

    if args.diagnostico:
        perfil = hardware.coletar_hardware()
        plano = hardware.planejar_execucao(perfil, args.qualidade,
                                           args.modelo, args.device)
        if args.json:
            print(json.dumps({"hardware": perfil.to_dict(),
                              "plano": plano.to_dict()},
                             indent=2, ensure_ascii=False))
        else:
            print(hardware.relatorio_hardware(perfil, plano))
        return 0

    if not args.entrada:
        construir_parser().print_help()
        return 1

    cfg = Config(
        entrada=args.entrada,
        saida=args.saida,
        formatos_audio=args.formatos,
        formatos_video=args.video,
        visual=args.visual,
        resolucao=args.resolucao,
        imagem_fundo=args.imagem_fundo,
        qualidade=args.qualidade,
        modelo=args.modelo,
        device=args.device,
        seis_stems=args.seis_stems,
        melhorar_entrada=args.melhorar_entrada,
        melhorar_stems=args.melhorar_stems,
        normalizar_stems=args.normalizar_stems,
        alvo_lufs=args.lufs,
        teto_dbtp=args.teto_dbtp,
        denoise_forcado=args.denoise,
        derivados=args.derivados,
        refinar_vocal=args.refinar_vocal,
        manter_master=args.manter_master,
        apenas_stems=args.apenas_stems,
        somente_analise=args.somente_analise,
    )

    try:
        relatorio = executar(cfg)
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"\nErro: {e}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(relatorio, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
