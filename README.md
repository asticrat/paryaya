# Paryaya · पर्याय

**Nepali Automatic Speech Recognition — built from scratch, sold as a REST API.**

Converts spoken Nepali audio to written Devanagari text. No Whisper, no fine-tuning — a Conformer trained entirely on Nepali speech data. First dedicated Nepali ASR API.

```
 16 kHz audio → 80-dim Log-Mel → SpecAugment
       │
       ▼
 Conv Subsampler (4×) — reduces sequence length 4×
       │
       ▼
 18 × Conformer Block  [d=512, 8 heads, conv kernel=31]
  ┌───┴───────────────────────┐
  │  FF(½) → MHA → Conv → FF(½) → LayerNorm
  └───────────────────────────┘
       │
  ┌────┴──────────┐
  │               │
CTC Head     6 × Transformer Decoder
(training)        │
                  ▼
           Devanagari text
           Loss = 0.3·CTC + 0.7·CE
```

---

## Quick Start

```bash
git clone https://github.com/yourname/paryaya.git && cd paryaya
pip install -e .
cp .env.example .env        # fill in WANDB_API_KEY, etc.
python -c "import paryaya; print(paryaya.__version__)"   # 1.0.0
```

---

## Data Pipeline

```bash
# 1. Download free datasets (~177 hours total)
python -m paryaya.data.download --output data/raw/

# 2. Standardise to 16 kHz mono WAV, strip silence, validate Devanagari
python -m paryaya.data.preprocess --input data/raw/ --output data/processed/

# 3. Generate synthetic Nepali speech from text (zero cost with gTTS)
python -m paryaya.data.synthetic_tts \
    --text_file data/text_corpus/nepali_sentences.txt \
    --output    data/synthetic/ --backend gtts

# 4. Speed-perturb (0.9×, 1.1×) + noise injection (SNR 10/20 dB)
python -m paryaya.data.augment \
    --input data/processed/ --output data/augmented/

# 5. Build 90/5/5 train/valid/test manifests
python -m paryaya.data.manifest

# 6. Build vocabulary from manifests
python -m paryaya.data.tokenizer --build data/manifests/train.json \
    --out data/vocab/nepali_vocab.json
```

---

## Training

### Local smoke test (CPU / small GPU)
```bash
python -m paryaya.training.train \
    --config  configs/model_small.yaml \
    --train   data/manifests/train.json \
    --valid   data/manifests/valid.json \
    --out_dir checkpoints/ --epochs 2
```

### Full run on RunPod A100
```bash
# On RunPod instance:
bash scripts/setup_runpod.sh
python -m paryaya.training.train \
    --config  configs/model_medium.yaml \
    --train   data/manifests/train.json \
    --valid   data/manifests/valid.json \
    --out_dir checkpoints/ --epochs 100

# Download trained weights to your server:
scp -P <port> root@<pod-ip>:paryaya/checkpoints/best_model.pt ./checkpoints/
scp -P <port> root@<pod-ip>:paryaya/data/vocab/nepali_vocab.json data/vocab/
```

Monitor training live: https://wandb.ai — project `paryaya-asr`

---

## Evaluate

```bash
python -m paryaya.training.evaluate \
    --checkpoint checkpoints/best_model.pt \
    --test       data/manifests/test.json
```

Target: **WER < 15%**, **CER < 8%**, **RTF < 0.3**

---

## API Deployment

```bash
# Start full stack (API + Redis + Celery worker + Nginx)
docker compose -f docker/docker-compose.yml up -d

# Health check
curl http://localhost:8000/health

# Transcribe
curl -X POST http://localhost:8000/v1/transcribe \
     -H "Authorization: Bearer sk-paryaya-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" \
     -F "file=@audio.wav"

# Create API key (admin)
curl -X POST http://localhost:8000/auth/keys \
     -H "X-Admin-Key: $ADMIN_SECRET_KEY" \
     -H "Content-Type: application/json" \
     -d '{"company": "NepTelecom", "plan": "business"}'
```

---

## API Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/v1/transcribe` | Bearer token | Upload audio → Devanagari transcript |
| `WS`   | `/v1/stream`     | Bearer token | Real-time streaming transcription |
| `GET`  | `/health`        | None | Model status + version |
| `POST` | `/auth/keys`     | Admin key | Create API key |

### Response example

```json
{
  "transcript":    "नमस्ते, मेरो नाम राम हो।",
  "confidence":    0.94,
  "durationSec":   5.3,
  "processingMs":  210,
  "model":         "paryaya-v1.0"
}
```

---

## Pricing

| Plan | Price | Limit |
|------|-------|-------|
| Starter | $29/month | 1,000 min/month |
| Business | $149/month | 10,000 min/month |
| Enterprise | Custom | Unlimited + SLA |
| Pay-as-you-go | $0.015/min | No monthly commit |

Rate limits: Starter 60 req/min · Business 200 req/min · Enterprise unlimited

---

## Target Metrics

| Metric | Target |
|--------|--------|
| Word Error Rate (WER) | < 15% |
| Character Error Rate (CER) | < 8% |
| Real-Time Factor (RTF) | < 0.3 |
| API Latency | < 500 ms |
| Uptime | 99.5% |

---

## Implementation Phases

| Phase | Scope | Status |
|-------|-------|--------|
| 1 — Scaffold | Directories, configs, requirements | ✅ |
| 2 — Data Pipeline | download, preprocess, augment, dataset | ⬜ |
| 3 — Model | ParyayaASR, Conformer, decoder, tokenizer | ⬜ |
| 4 — Training | train loop, loss, optimizer, evaluate | ⬜ |
| 5 — Inference | beam search, VAD, transcribe, postprocess | ⬜ |
| 6 — API | FastAPI, WebSocket, auth, rate limit, Docker | ⬜ |
| 7 — Launch | VPS, HTTPS, monitoring, beta customers | ⬜ |
