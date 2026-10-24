"""
hardware.py — Inspeção dos recursos da máquina e escolha automática de
configuração de execução (device, modelo, segmentação, paralelismo).

Nenhuma dependência obrigatória além da stdlib. psutil e torch são usados
quando disponíveis, com fallback gracioso.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field, asdict
from typing import Optional


# --------------------------------------------------------------------------
# Estruturas
# --------------------------------------------------------------------------

@dataclass
class GPUInfo:
    present: bool = False
    vendor: str = "none"          # nvidia | amd | apple | intel | none
    name: str = "N/A"
    vram_gb: float = 0.0
    backend: str = "cpu"          # cuda | rocm | mps | cpu
    driver: str = "N/A"
    compute_capability: str = "N/A"


@dataclass
class HardwareProfile:
    so: str = ""
    arquitetura: str = ""
    cpu: str = ""
    nucleos_fisicos: int = 0
    nucleos_logicos: int = 0
    ram_total_gb: float = 0.0
    ram_livre_gb: float = 0.0
    disco_livre_gb: float = 0.0
    gpu: GPUInfo = field(default_factory=GPUInfo)
    ffmpeg: bool = False
    ffmpeg_versao: str = "N/A"
    torch_versao: Optional[str] = None
    avisos: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class RunPlan:
    """Plano de execução derivado do hardware."""
    device: str = "cpu"                 # cuda | mps | cpu
    modelo: str = "htdemucs_ft"
    segment: Optional[int] = None       # segundos por bloco (None = padrão do modelo)
    shifts: int = 1                     # random shift trick (qualidade x tempo)
    overlap: float = 0.25
    jobs: int = 1                       # paralelismo em CPU
    precisao: str = "float32"           # float32 | float16
    lote_memoria_mb: int = 0
    perfil_qualidade: str = "equilibrado"
    justificativa: list = field(default_factory=list)
    tempo_estimado_por_minuto_s: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Coleta
# --------------------------------------------------------------------------

def _run(cmd: list[str], timeout: int = 8) -> str:
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        return (out.stdout or "").strip()
    except Exception:
        return ""


def _cpu_nome() -> str:
    sistema = platform.system()
    if sistema == "Linux":
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as fh:
                for linha in fh:
                    if linha.lower().startswith("model name"):
                        return linha.split(":", 1)[1].strip()
        except Exception:
            pass
    elif sistema == "Darwin":
        nome = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if nome:
            return nome
    elif sistema == "Windows":
        nome = os.environ.get("PROCESSOR_IDENTIFIER", "")
        if nome:
            return nome
    return platform.processor() or platform.machine() or "desconhecido"


def _memoria_gb() -> tuple[float, float]:
    """Retorna (total_gb, disponivel_gb)."""
    try:
        import psutil  # type: ignore
        vm = psutil.virtual_memory()
        return round(vm.total / 2**30, 2), round(vm.available / 2**30, 2)
    except Exception:
        pass

    if platform.system() == "Linux":
        try:
            info = {}
            with open("/proc/meminfo", "r", encoding="utf-8") as fh:
                for linha in fh:
                    chave, _, valor = linha.partition(":")
                    info[chave.strip()] = valor.strip()
            total = int(info["MemTotal"].split()[0]) / 2**20
            disp = int(info.get("MemAvailable", info["MemFree"]).split()[0]) / 2**20
            return round(total, 2), round(disp, 2)
        except Exception:
            pass

    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
        return round(total, 2), round(total * 0.5, 2)
    except Exception:
        return 0.0, 0.0


def _nucleos() -> tuple[int, int]:
    logicos = os.cpu_count() or 1
    fisicos = logicos
    try:
        import psutil  # type: ignore
        fisicos = psutil.cpu_count(logical=False) or logicos
    except Exception:
        if platform.system() == "Linux":
            try:
                ids = set()
                atual = {}
                with open("/proc/cpuinfo", "r", encoding="utf-8") as fh:
                    for linha in fh:
                        if not linha.strip():
                            if "physical id" in atual and "core id" in atual:
                                ids.add((atual["physical id"], atual["core id"]))
                            atual = {}
                            continue
                        k, _, v = linha.partition(":")
                        atual[k.strip()] = v.strip()
                if ids:
                    fisicos = len(ids)
            except Exception:
                pass
    return fisicos, logicos


def _detecta_gpu() -> GPUInfo:
    gpu = GPUInfo()

    # 1) PyTorch é a fonte mais confiável, pois reflete o que será realmente usado.
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gpu.present = True
            gpu.name = props.name
            gpu.vram_gb = round(props.total_memory / 2**30, 2)
            gpu.compute_capability = f"{props.major}.{props.minor}"
            versao = getattr(torch.version, "hip", None)
            if versao:
                gpu.vendor, gpu.backend, gpu.driver = "amd", "rocm", f"ROCm {versao}"
            else:
                gpu.vendor, gpu.backend = "nvidia", "cuda"
                gpu.driver = f"CUDA {torch.version.cuda}"
            return gpu

        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            total, _ = _memoria_gb()
            gpu.present = True
            gpu.vendor, gpu.backend = "apple", "mps"
            gpu.name = "Apple Silicon (GPU integrada)"
            # Memória unificada: a GPU pode endereçar boa parte da RAM.
            gpu.vram_gb = round(total * 0.6, 2)
            return gpu
    except Exception:
        pass

    # 2) nvidia-smi, caso o torch instalado seja a build de CPU.
    if shutil.which("nvidia-smi"):
        saida = _run([
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ])
        if saida:
            primeira = saida.splitlines()[0].split(",")
            if len(primeira) >= 3:
                gpu.present = True
                gpu.vendor = "nvidia"
                gpu.backend = "cpu"  # detectada, mas indisponível para o torch atual
                gpu.name = primeira[0].strip()
                gpu.vram_gb = round(float(primeira[1].strip()) / 1024, 2)
                gpu.driver = primeira[2].strip()
                return gpu

    # 3) rocm-smi
    if shutil.which("rocm-smi"):
        saida = _run(["rocm-smi", "--showproductname"])
        if saida:
            gpu.present = True
            gpu.vendor, gpu.backend = "amd", "cpu"
            gpu.name = saida.splitlines()[-1].strip()[:80]
            return gpu

    # 4) Enumeração genérica (só informativa)
    if platform.system() == "Linux" and shutil.which("lspci"):
        saida = _run(["bash", "-lc", "lspci | grep -Ei 'vga|3d|display'"])
        if saida:
            gpu.present = True
            gpu.name = saida.splitlines()[0].split(":")[-1].strip()[:80]
            baixo = gpu.name.lower()
            gpu.vendor = ("nvidia" if "nvidia" in baixo else
                          "amd" if ("amd" in baixo or "radeon" in baixo) else
                          "intel" if "intel" in baixo else "desconhecido")
    return gpu


def _ffmpeg() -> tuple[bool, str]:
    if not shutil.which("ffmpeg"):
        return False, "N/A"
    saida = _run(["ffmpeg", "-version"])
    primeira = saida.splitlines()[0] if saida else "instalado"
    return True, primeira[:90]


def coletar_hardware() -> HardwareProfile:
    """Levanta o perfil completo da máquina."""
    fisicos, logicos = _nucleos()
    ram_total, ram_livre = _memoria_gb()
    try:
        uso = shutil.disk_usage(os.getcwd())
        disco = round(uso.free / 2**30, 2)
    except Exception:
        disco = 0.0

    ffmpeg_ok, ffmpeg_ver = _ffmpeg()

    try:
        import torch  # type: ignore
        torch_ver = torch.__version__
    except Exception:
        torch_ver = None

    perfil = HardwareProfile(
        so=f"{platform.system()} {platform.release()}",
        arquitetura=platform.machine(),
        cpu=_cpu_nome(),
        nucleos_fisicos=fisicos,
        nucleos_logicos=logicos,
        ram_total_gb=ram_total,
        ram_livre_gb=ram_livre,
        disco_livre_gb=disco,
        gpu=_detecta_gpu(),
        ffmpeg=ffmpeg_ok,
        ffmpeg_versao=ffmpeg_ver,
        torch_versao=torch_ver,
    )

    # Avisos acionáveis
    if not ffmpeg_ok:
        perfil.avisos.append(
            "FFmpeg não encontrado no PATH. É obrigatório para ler formatos "
            "comprimidos e para exportar mp3/mp4/mkv/opus."
        )
    if torch_ver is None:
        perfil.avisos.append(
            "PyTorch não instalado. Rode: pip install -r requirements.txt"
        )
    if perfil.gpu.present and perfil.gpu.backend == "cpu":
        perfil.avisos.append(
            f"GPU {perfil.gpu.name} detectada, mas o PyTorch instalado não a "
            "utiliza (build CPU-only). Reinstale o torch com suporte "
            "CUDA/ROCm para ganho de 10x a 30x na separação."
        )
    if ram_total and ram_total < 8:
        perfil.avisos.append(
            "Menos de 8 GB de RAM: será usado modo de baixa memória "
            "(segmentos curtos), com processamento mais lento."
        )
    if disco and disco < 5:
        perfil.avisos.append(
            "Menos de 5 GB livres em disco. Stems em WAV 32-bit ocupam bastante "
            "espaço; considere exportar em FLAC."
        )
    return perfil


# --------------------------------------------------------------------------
# Planejamento
# --------------------------------------------------------------------------

# Custo aproximado em segundos de processamento por minuto de áudio,
# medido com htdemucs, shifts=1. Usado apenas para estimativa exibida ao usuário.
_CUSTO_BASE = {
    "cuda_alta": 2.5,
    "cuda_media": 5.0,
    "cuda_baixa": 12.0,
    "mps": 25.0,
    "cpu": 90.0,
}


def planejar_execucao(
    perfil: HardwareProfile,
    qualidade: str = "auto",
    modelo: Optional[str] = None,
    forcar_device: Optional[str] = None,
) -> RunPlan:
    """
    Traduz o perfil de hardware em parâmetros concretos de execução.

    qualidade: 'rapido' | 'equilibrado' | 'maxima' | 'auto'
    """
    plano = RunPlan()
    j = plano.justificativa

    gpu = perfil.gpu
    if forcar_device:
        plano.device = forcar_device
        j.append(f"Device forçado pelo usuário: {forcar_device}.")
    elif gpu.backend in ("cuda", "rocm"):
        plano.device = "cuda"  # torch usa 'cuda' também para ROCm
        j.append(f"GPU {gpu.name} ({gpu.vram_gb} GB) disponível via {gpu.backend}.")
    elif gpu.backend == "mps":
        plano.device = "mps"
        j.append("Apple Silicon detectado; usando backend MPS (Metal).")
    else:
        plano.device = "cpu"
        j.append("Sem aceleração por GPU disponível; execução em CPU.")

    # Qualidade automática: hardware forte pede o preset melhor.
    if qualidade == "auto":
        if plano.device == "cuda" and gpu.vram_gb >= 8:
            qualidade = "maxima"
        elif plano.device in ("cuda", "mps"):
            qualidade = "equilibrado"
        elif perfil.nucleos_fisicos >= 8 and perfil.ram_total_gb >= 16:
            qualidade = "equilibrado"
        else:
            qualidade = "rapido"
        j.append(f"Perfil de qualidade escolhido automaticamente: {qualidade}.")
    plano.perfil_qualidade = qualidade

    # Modelo: htdemucs_ft é o mais preciso (4 passagens); _6s separa guitarra e piano.
    if modelo:
        plano.modelo = modelo
    elif qualidade == "maxima":
        plano.modelo = "htdemucs_ft"
    elif qualidade == "equilibrado":
        plano.modelo = "htdemucs"
    else:
        plano.modelo = "htdemucs"
    j.append(f"Modelo base: {plano.modelo}.")

    # shifts multiplica o tempo de inferência, mas melhora ~0.2-0.5 dB de SDR.
    plano.shifts = {"rapido": 0, "equilibrado": 1, "maxima": 4}[qualidade]
    plano.overlap = {"rapido": 0.10, "equilibrado": 0.25, "maxima": 0.50}[qualidade]

    # Segmentação em função da memória disponível para o modelo.
    if plano.device == "cuda":
        vram = gpu.vram_gb
        if vram >= 16:
            plano.segment = None            # peça inteira, sem partição forçada
            plano.lote_memoria_mb = 16000
            base = "cuda_alta"
        elif vram >= 8:
            plano.segment = 15
            plano.lote_memoria_mb = 8000
            base = "cuda_alta"
        elif vram >= 6:
            plano.segment = 10
            plano.lote_memoria_mb = 6000
            base = "cuda_media"
        elif vram >= 4:
            plano.segment = 7
            plano.lote_memoria_mb = 4000
            base = "cuda_baixa"
        else:
            plano.segment = 5
            plano.lote_memoria_mb = 2000
            base = "cuda_baixa"
            j.append("VRAM baixa (<4 GB): segmentos curtos para evitar OOM.")
        # float16 acelera bastante em placas com tensor cores (CC >= 7.0).
        cc = gpu.compute_capability
        try:
            if cc != "N/A" and float(cc) >= 7.0 and qualidade != "maxima":
                plano.precisao = "float16"
                j.append(f"Compute capability {cc}: meia precisão habilitada.")
        except ValueError:
            pass
        plano.jobs = 1
    elif plano.device == "mps":
        plano.segment = 10
        plano.jobs = 1
        base = "mps"
        j.append("MPS: segmento de 10 s (limite prático do backend Metal).")
    else:
        ram = perfil.ram_total_gb or 8
        plano.segment = 20 if ram >= 16 else (10 if ram >= 8 else 6)
        # Cada job carrega uma cópia do modelo (~1.5 GB de pico com htdemucs).
        teto_por_ram = max(1, int((ram - 2) // 2))
        plano.jobs = max(1, min(perfil.nucleos_fisicos or 1, teto_por_ram, 4))
        base = "cpu"
        j.append(
            f"CPU: {plano.jobs} job(s) paralelo(s), segmento de {plano.segment} s "
            f"(limitado por {ram} GB de RAM)."
        )

    fator_shift = max(1, plano.shifts)
    fator_modelo = 4.0 if plano.modelo == "htdemucs_ft" else 1.0
    fator_overlap = 1.0 + plano.overlap
    plano.tempo_estimado_por_minuto_s = round(
        _CUSTO_BASE[base] * fator_shift * fator_modelo * fator_overlap
        / max(1, plano.jobs if plano.device == "cpu" else 1),
        1,
    )
    return plano


def relatorio_hardware(perfil: HardwareProfile, plano: RunPlan) -> str:
    """Texto legível com o diagnóstico da máquina e o plano escolhido."""
    g = perfil.gpu
    linhas = [
        "═" * 62,
        " DIAGNÓSTICO DE HARDWARE",
        "═" * 62,
        f"  Sistema........: {perfil.so} ({perfil.arquitetura})",
        f"  CPU............: {perfil.cpu}",
        f"  Núcleos........: {perfil.nucleos_fisicos} físicos / "
        f"{perfil.nucleos_logicos} lógicos",
        f"  RAM............: {perfil.ram_total_gb} GB "
        f"({perfil.ram_livre_gb} GB livres)",
        f"  Disco livre....: {perfil.disco_livre_gb} GB",
        f"  GPU............: {g.name}"
        + (f" — {g.vram_gb} GB VRAM" if g.vram_gb else ""),
        f"  Backend GPU....: {g.backend} ({g.driver})",
        f"  PyTorch........: {perfil.torch_versao or 'não instalado'}",
        f"  FFmpeg.........: {'ok' if perfil.ffmpeg else 'AUSENTE'}",
        "",
        "─" * 62,
        " PLANO DE EXECUÇÃO",
        "─" * 62,
        f"  Device.........: {plano.device}",
        f"  Modelo.........: {plano.modelo}",
        f"  Qualidade......: {plano.perfil_qualidade}",
        f"  Shifts.........: {plano.shifts}   Overlap: {plano.overlap}",
        f"  Segmento.......: {plano.segment or 'automático'} s   "
        f"Jobs: {plano.jobs}   Precisão: {plano.precisao}",
        f"  Estimativa.....: ~{plano.tempo_estimado_por_minuto_s} s de "
        "processamento por minuto de música",
    ]
    if plano.justificativa:
        linhas.append("")
        linhas.append("  Decisões:")
        linhas += [f"    · {m}" for m in plano.justificativa]
    if perfil.avisos:
        linhas.append("")
        linhas.append("  Avisos:")
        linhas += [f"    ! {a}" for a in perfil.avisos]
    linhas.append("═" * 62)
    return "\n".join(linhas)


if __name__ == "__main__":
    p = coletar_hardware()
    pl = planejar_execucao(p)
    print(relatorio_hardware(p, pl))
    print()
    print(json.dumps({"hardware": p.to_dict(), "plano": pl.to_dict()},
                     indent=2, ensure_ascii=False))
