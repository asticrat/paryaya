#!/usr/bin/env bash
# RunPod A100 40GB setup for Whisper fine-tuning.
#
# Run this ONCE after launching a RunPod pod:
#   bash scripts/setup_runpod_whisper.sh
#
# Then fine-tune:
#   python scripts/finetune_whisper.py --config configs/finetune_whisper.yaml
#
# Recommended RunPod template:
#   Image: pytorch/pytorch:2.8.0-cuda12.4-cudnn9-runtime
#   GPU:   A100 40GB  (~$2.50/hr)
#   Disk:  50 GB
set -euo pipefail

echo "=================================================="
echo "  Paryaya — Whisper fine-tune RunPod setup"
echo "=================================================="

# ── System deps ───────────────────────────────────────
apt-get update -qq && apt-get install -y -qq \
    git ffmpeg libsndfile1 tmux htop nvtop 2>/dev/null
echo "✅ System deps installed"

# ── Clone / update repo ───────────────────────────────
REPO_DIR="/workspace/paryaya"
if [ -d "$REPO_DIR/.git" ]; then
    echo "Repo exists — pulling latest..."
    git -C "$REPO_DIR" pull
else
    git clone https://github.com/asticrat/paryaya.git "$REPO_DIR"
fi
cd "$REPO_DIR"
echo "✅ Repo ready at $REPO_DIR"

# ── Python deps ───────────────────────────────────────
pip install -q --upgrade pip
pip install -q \
    "transformers>=4.43" \
    "datasets>=2.21" \
    "accelerate>=0.33" \
    "evaluate>=0.4" \
    "jiwer>=3.0" \
    "soundfile>=0.12" \
    librosa \
    wandb \
    pyyaml \
    "huggingface-hub>=0.24"

# Install the paryaya package itself
pip install -q -e .
echo "✅ Python deps installed"

# ── Verify GPU ────────────────────────────────────────
echo ""
python3 -c "
import torch
print(f'CUDA available : {torch.cuda.is_available()}')
print(f'GPU            : {torch.cuda.get_device_name(0)}')
print(f'VRAM           : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# ── Prompt for secrets ────────────────────────────────
echo ""
echo "=================================================="
echo "  Set these before training:"
echo ""
echo "  export HF_TOKEN=hf_xxxxxxxxxxxx       # huggingface.co/settings/tokens"
echo "  export WANDB_API_KEY=xxxxxxxxxxxx     # wandb.ai/settings (optional)"
echo "  export WANDB_PROJECT=paryaya-whisper  # optional"
echo ""
echo "  Then:"
echo "  cd /workspace/paryaya"
echo "  tmux new -s train"
echo "  python scripts/finetune_whisper.py --config configs/finetune_whisper.yaml"
echo "=================================================="
