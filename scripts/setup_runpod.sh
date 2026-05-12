#!/usr/bin/env bash
# scripts/setup_runpod.sh — One-command RunPod A100 training environment setup.
#
# Run on a fresh RunPod PyTorch 2.2 instance:
#   bash scripts/setup_runpod.sh
set -euo pipefail

echo "🚀 Setting up Paryaya ASR training environment on RunPod A100..."

# System packages
apt-get update -q
apt-get install -y -q libsndfile1 ffmpeg sox wget unzip git

# Python packages
pip install --upgrade pip -q
pip install -r requirements.txt -q
pip install -e . -q

# Verify CUDA GPU is available
python -c "
import torch
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}  ({torch.cuda.get_device_properties(0).total_memory // 1024**3} GB)')
else:
    print('⚠ No CUDA GPU detected — training will be slow on CPU')
"

# Create data directory structure
mkdir -p data/{raw,processed,augmented,synthetic,text_corpus,noise_samples,manifests,vocab}
mkdir -p checkpoints logs

echo ""
echo "📥 Downloading free datasets (this may take 30–60 minutes)..."
python -m paryaya.data.download --output data/raw/

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Preprocess audio:"
echo "     python -m paryaya.data.preprocess --input data/raw/ --output data/processed/"
echo ""
echo "  2. (Optional) Generate synthetic TTS data:"
echo "     python -m paryaya.data.text_corpus_builder --output data/text_corpus/nepali_sentences.txt"
echo "     python -m paryaya.data.synthetic_tts --text_file data/text_corpus/nepali_sentences.txt \\"
echo "            --output data/synthetic/ --backend gtts"
echo ""
echo "  3. Augment and build manifests:"
echo "     python -m paryaya.data.augment --input data/processed/ --output data/augmented/"
echo "     python -m paryaya.data.manifest"
echo ""
echo "  4. Train:"
echo "     WANDB_API_KEY=\$WANDB_API_KEY python -m paryaya.training.train \\"
echo "            --config configs/model_medium.yaml \\"
echo "            --train  data/manifests/train.json \\"
echo "            --valid  data/manifests/valid.json \\"
echo "            --out_dir checkpoints/ --epochs 100"
echo ""
echo "  5. After training, download your model:"
echo "     scp -P <port> root@<pod-ip>:paryaya/checkpoints/best_model.pt ./"
echo "     scp -P <port> root@<pod-ip>:paryaya/data/vocab/nepali_vocab.json data/vocab/"
