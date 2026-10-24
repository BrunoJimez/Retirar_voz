"""
StemForge — separação profissional de faixas musicais com modelos de IA
abertos, livres e gratuitos (Demucs v4 / UVR-MDX-Net).

Sem APIs pagas, sem chaves, sem serviços em nuvem: tudo roda localmente.
"""

__version__ = "1.0.0"
__all__ = ["hardware", "analysis", "enhance", "separate", "export", "pipeline"]

from . import analysis, enhance, export, hardware, pipeline, separate  # noqa: F401
