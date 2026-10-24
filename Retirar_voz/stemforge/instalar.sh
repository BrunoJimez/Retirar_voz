#!/usr/bin/env bash
# instalar.sh — instalação assistida do StemForge com detecção de GPU.
set -e

echo "── StemForge · instalação ──"

command -v ffmpeg >/dev/null 2>&1 || {
  echo "AVISO: FFmpeg não encontrado. Instale antes de continuar:"
  echo "  Debian/Ubuntu : sudo apt install ffmpeg"
  echo "  macOS         : brew install ffmpeg"
  echo "  Windows       : winget install Gyan.FFmpeg"
}

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -U pip wheel

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU NVIDIA detectada — instalando PyTorch com CUDA 12.1."
  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
elif command -v rocm-smi >/dev/null 2>&1; then
  echo "GPU AMD detectada — instalando PyTorch com ROCm 6.0."
  pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
else
  echo "Sem GPU dedicada detectada — instalando PyTorch padrão (CPU/MPS)."
  pip install torch torchaudio
fi

pip install -r requirements.txt
pip install -e .

echo
echo "Pronto. Ative o ambiente com:  source .venv/bin/activate"
echo "E rode o diagnóstico com:      stemforge --diagnostico"
