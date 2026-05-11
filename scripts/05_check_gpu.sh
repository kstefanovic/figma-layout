#!/usr/bin/env bash
set -euo pipefail

cd /root/figma-layout

echo "Checking NVIDIA GPU..."
nvidia-smi || true

echo ""
echo "Checking PyTorch CUDA..."
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"gpu {i}:", torch.cuda.get_device_name(i))
PY