# PARYAYA — RunPod Whisper Fine-Tuning Guide
### Complete step-by-step: accounts → pod → train → download → deploy
**Cost: ~$10–20 | Time: ~3–6 hours training (mostly unattended)**

---

## WHAT THIS GUIDE DOES

Fine-tunes OpenAI Whisper medium (244M params) on the Mozilla Common Voice 17 Nepali dataset
using a RunPod A100 GPU. After training, you download the checkpoint and point the Paryaya API
at it. No custom model needed.

```
Before:  ASR_BACKEND=paryaya  →  custom conformer (untrained, random weights)
After:   ASR_BACKEND=whisper  →  Whisper medium fine-tuned on real Nepali speech
```

---

## PART 0 — DO THESE FIRST (before touching RunPod)
*~30 minutes. Do all three before launching a pod — you will need these credentials.*

---

### 0.1 — Create a HuggingFace Account and Accept Dataset Terms

The Common Voice dataset requires a free HuggingFace account and explicit terms acceptance.
You cannot download the dataset without this. Do it now so it is ready when training starts.

**Step 1 — Create account**

1. Go to **huggingface.co**
2. Click **Sign Up** in the top right
3. Enter your email and a password
4. Check your email and click the verification link

**Step 2 — Accept Common Voice 17 terms**

1. While logged in, go to:
   **huggingface.co/datasets/mozilla-foundation/common_voice_17_0**
2. Scroll down until you see a blue banner that says **"You need to agree to share your contact
   information to access this dataset"**
3. Click **Agree and access dataset**
4. Fill in your name and email if asked, click **Submit**
5. The page should now show the dataset files — if you see the files, terms are accepted

> If you skip this step, training will fail with:
> `datasets.exceptions.DatasetNotFoundError` or `401 Unauthorized`

**Step 3 — Create an access token**

1. Go to **huggingface.co/settings/tokens**
2. Click **New token**
3. Name it: `paryaya-training`
4. Role: **Read**
5. Click **Generate a token**
6. Copy the token — it looks like: `hf_ABcDefGhIjKlMnOpQrStUvWxYz123456`
7. **Save it somewhere safe** — you will need it in Part 2

---

### 0.2 — Create a Weights & Biases Account (optional but strongly recommended)

W&B gives you real-time charts of your training loss and WER so you can watch from your phone
without keeping SSH open. It is free.

1. Go to **wandb.ai** and click **Sign up**
2. Sign up with Google or email
3. Once logged in, go to **wandb.ai/settings**
4. Scroll to **API keys**, click **New key**
5. Copy the key — it looks like: `abcdef1234567890abcdef1234567890abcdef12`
6. Save it alongside your HuggingFace token

> If you skip W&B, training still works. Just set `report_to: "none"` in the config.
> You will only see progress in the terminal.

---

### 0.3 — Make Sure Your SSH Key Exists

You need an SSH key to connect to RunPod. Check if you already have one:

```bash
# Run on your Mac terminal:
ls ~/.ssh/id_ed25519.pub
```

**If the file exists** — skip to 0.3b. You already have a key.

**If you get "No such file or directory"** — create one:

```bash
ssh-keygen -t ed25519 -C "paryaya-runpod"
# Press Enter three times (accept defaults, no passphrase)
```

**0.3b — Copy your public key to the clipboard:**

```bash
cat ~/.ssh/id_ed25519.pub
# Select all the output and copy it — you will paste it into RunPod in Part 1
```

The output looks like:
```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI... paryaya-runpod
```

---

## PART 1 — CREATE RUNPOD ACCOUNT AND LAUNCH GPU POD
*~20 minutes*

---

### 1.1 — Create RunPod Account and Add Credits

1. Go to **runpod.io** and click **Sign Up**
2. Enter your email and password, verify your email
3. Click **Billing** in the left sidebar
4. Click **Add Payment Method** and enter your credit card
5. Click **Add Credits** and add **$30** to start
   - Training costs ~$10–20 total
   - $30 gives you buffer for setup time and mistakes
   - You can add more later; unused credits do not expire

---

### 1.2 — Add Your SSH Key to RunPod

Before launching a pod, add your SSH key so you can connect without a password.

1. Click **Settings** in the left sidebar (gear icon)
2. Click **SSH Public Keys**
3. Click **Add SSH Key**
4. Paste the entire contents of your public key (from step 0.3b)
5. Give it a name: `mac-paryaya`
6. Click **Save**

---

### 1.3 — Launch the A100 Pod

1. Click **Secure Cloud** in the left sidebar
   > Use **Secure Cloud**, not Community Cloud. Community Cloud pods can be interrupted
   > mid-training. Secure Cloud pods are stable and guaranteed.

2. In the GPU list, find **A100 PCIe 40GB**
   - It shows at **~$1.89/hr** (price may vary slightly)
   - If unavailable, use **A100 SXM 80GB** (~$2.79/hr, faster but more expensive)

3. Click **Deploy** next to the A100 PCIe 40GB

4. You will see the **Pod Configuration** screen. Fill it in **exactly** as follows:

   | Setting | Value | Why |
   |---|---|---|
   | Template | RunPod PyTorch 2.2 | Has CUDA 12.1 + Python pre-installed |
   | Container Disk | 30 GB | For OS, pip packages (~15 GB needed) |
   | Volume Disk | 80 GB | For dataset (~15 GB) + checkpoints (~5 GB) + cache |
   | Volume Mount Path | /workspace | Where all your data will live |
   | Expose TCP Ports | 22 | Required for SSH access |

   > **Critical:** Set Volume Disk to 80 GB. The Common Voice dataset is ~15 GB and
   > HuggingFace caches model weights during training. Running out of disk mid-training
   > corrupts your checkpoint and wastes GPU time.

5. Click **Deploy** and wait **2–5 minutes** for the pod to reach **Running** status

---

### 1.4 — Connect to Your Pod via SSH

1. Once the pod shows **Running**, click **Connect**
2. Click **SSH over exposed TCP** — you will see a command like:
   ```
   ssh root@XX.XX.XX.XX -p XXXXX -i ~/.ssh/id_ed25519
   ```
3. Copy that exact command and run it in your Mac terminal

**If the connection works**, you will see a prompt like:
```
root@A5F8D3B2:~#
```
Skip to step 1.5.

**If you get "Connection timed out" or "Permission denied":**

```bash
# Try adding -o StrictHostKeyChecking=no to bypass host key prompt:
ssh root@XX.XX.XX.XX -p XXXXX -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no
```

If still failing, add your key to the SSH agent first:
```bash
# On your Mac:
ssh-add ~/.ssh/id_ed25519
# Then retry the ssh command
```

---

### 1.5 — Verify the GPU is Working

Run these commands on the **RunPod terminal** (after SSH-ing in):

```bash
# Verify you have a GPU
nvidia-smi
```

Expected output (approximately):
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.xx    Driver Version: 535.xx    CUDA Version: 12.2          |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|   0  A100-PCIE-40GB      Off  | 00000000:00:00.0 Off |                    0 |
|  N/A   30C    P0    34W / 250W|      0MiB / 40960MiB |      0%      Default |
+-----------------------------------------------------------------------------+
```

```bash
# Verify PyTorch + CUDA
python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

Expected output:
```
CUDA: True
GPU: NVIDIA A100-PCIE-40GB
```

**If CUDA shows False:**
```bash
pip uninstall torch torchaudio -y
pip install torch==2.2.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
python3 -c "import torch; print(torch.cuda.is_available())"
```

---

## PART 2 — SET UP THE POD
*~15 minutes (runs mostly automatically)*

---

### 2.1 — Set the HuggingFace Cache to the Persistent Volume

By default, HuggingFace downloads go to `~/.cache/` which is on the **container disk** (wiped
on pod restart). Move the cache to `/workspace` so downloads survive reconnection:

```bash
# Run on the RunPod terminal:
echo 'export HF_HOME=/workspace/.cache/huggingface' >> ~/.bashrc
echo 'export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets' >> ~/.bashrc
source ~/.bashrc

# Create the cache directory
mkdir -p /workspace/.cache/huggingface/datasets
```

---

### 2.2 — Clone the Paryaya Repo

```bash
cd /workspace
git clone https://github.com/asticrat/paryaya.git
cd paryaya
```

Expected output:
```
Cloning into 'paryaya'...
remote: Enumerating objects: 120, done.
...
Resolving deltas: done.
```

Verify the key files are present:
```bash
ls scripts/
# Should include: finetune_whisper.py  setup_runpod_whisper.sh  test_whisper.py

ls configs/
# Should include: finetune_whisper.yaml
```

If you get `Repository not found`, the repo may be private. In that case, use HTTPS with a token:
```bash
git clone https://YOUR_HF_TOKEN@github.com/asticrat/paryaya.git
```

---

### 2.3 — Run the Setup Script

This installs all Python dependencies. It takes **5–10 minutes**.

```bash
cd /workspace/paryaya
bash scripts/setup_runpod_whisper.sh
```

You will see output like:
```
==================================================
  Paryaya — Whisper fine-tune RunPod setup
==================================================
✅ System deps installed
Repo exists — pulling latest...
✅ Repo ready at /workspace/paryaya
...installing packages...
✅ Python deps installed
CUDA available : True
GPU            : NVIDIA A100-PCIE-40GB
VRAM           : 40.1 GB
```

**If a package fails to install**, install it manually:
```bash
pip install "transformers>=4.43" "datasets>=2.21" "accelerate>=0.33" "evaluate>=0.4" \
    "jiwer>=3.0" "soundfile>=0.12" librosa wandb pyyaml "huggingface-hub>=0.24"
pip install -e .
```

---

### 2.4 — Set Your Secret Keys

```bash
# Set HuggingFace token (from step 0.1 — required)
export HF_TOKEN=hf_ABcDefGhIjKlMnOpQrStUvWxYz123456

# Set WandB key (from step 0.2 — optional)
export WANDB_API_KEY=abcdef1234567890abcdef1234567890abcdef12
export WANDB_PROJECT=paryaya-whisper

# Make them persist if pod restarts (saves to .bashrc)
echo "export HF_TOKEN=hf_ABcDefGhIjKlMnOpQrStUvWxYz123456" >> ~/.bashrc
echo "export WANDB_API_KEY=abcdef1234567890abcdef1234567890abcdef12" >> ~/.bashrc
echo "export WANDB_PROJECT=paryaya-whisper" >> ~/.bashrc
```

> Replace the token values above with your actual keys from steps 0.1 and 0.2.

Verify the keys are set:
```bash
echo "HF_TOKEN starts with: ${HF_TOKEN:0:5}"
# Should print: HF_TOKEN starts with: hf_AB

echo "WANDB key set: $([ -n "$WANDB_API_KEY" ] && echo YES || echo NO)"
# Should print: WANDB key set: YES
```

---

### 2.5 — Log In to W&B (skip if not using W&B)

```bash
wandb login
# Paste your API key when prompted
# Press Enter

wandb status
# Should show: Currently logged in as: YOUR_USERNAME
```

If you see `wandb: ERROR`, your key is wrong. Get a new one at wandb.ai/settings → API Keys.

---

### 2.6 — Final Readiness Check

Run this block as-is — it checks everything is ready before you spend money training:

```bash
python3 - <<'EOF'
import sys, os

errors = []

# Check GPU
try:
    import torch
    if not torch.cuda.is_available():
        errors.append("CUDA not available — GPU not detected")
    else:
        gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅  GPU: {torch.cuda.get_device_name(0)} ({gb:.1f} GB)")
except Exception as e:
    errors.append(f"torch import failed: {e}")

# Check transformers
try:
    import transformers
    print(f"✅  transformers: {transformers.__version__}")
except ImportError:
    errors.append("transformers not installed")

# Check datasets
try:
    import datasets
    print(f"✅  datasets: {datasets.__version__}")
except ImportError:
    errors.append("datasets not installed")

# Check HF token
token = os.getenv("HF_TOKEN", "")
if not token.startswith("hf_"):
    errors.append("HF_TOKEN not set or invalid (must start with hf_)")
else:
    print(f"✅  HF_TOKEN: set ({token[:8]}...)")

# Check disk space
import shutil
total, used, free = shutil.disk_usage("/workspace")
free_gb = free / 1e9
if free_gb < 30:
    errors.append(f"Low disk space on /workspace: {free_gb:.1f} GB free (need 30+ GB)")
else:
    print(f"✅  Disk space: {free_gb:.1f} GB free on /workspace")

if errors:
    print("\n❌  PROBLEMS FOUND:")
    for e in errors:
        print(f"   - {e}")
    sys.exit(1)
else:
    print("\n🟢  All checks passed — ready to train!")
EOF
```

All lines should show ✅. Fix any ❌ before proceeding.

---

## PART 3 — SMOKE TEST (run this before the full training)
*~5 minutes — verifies the entire pipeline on 2 steps*

This runs the training script with only 8 samples and 2 optimizer steps. It downloads a tiny
slice of Common Voice, runs preprocessing, does 2 training steps, and confirms everything works.
**Do not skip this.** Finding a bug here costs 5 minutes; finding it after 3 hours of training
costs money and time.

```bash
cd /workspace/paryaya

python3 scripts/finetune_whisper.py \
    --config configs/finetune_whisper.yaml \
    --smoke_test
```

**What you will see (normal output):**

```
Loading processor from openai/whisper-medium ...
Downloading model.safetensors: 100%|████████| 1.52G/1.52G ...

Loading mozilla-foundation/common_voice_17_0 (ne) ...
Downloading data: 100%|████████| ...

  ⚡ Smoke test mode — 2 steps only
Pre-processing audio + tokenising transcripts ...
  train=8  eval=4

Loading openai/whisper-medium weights ...
🚀 Training on cuda | steps=2 | fp16=True
...
{'loss': 8.2341, 'grad_norm': ..., 'learning_rate': ..., 'epoch': ...}
{'eval_loss': ..., 'eval_wer': ..., ...}

✅ Training complete.
   Best model saved → checkpoints/whisper-medium-nepali/best
```

**The smoke test passes if:**
- You see `🚀 Training on cuda` (not cpu)
- You see `✅ Training complete.`
- No Python traceback appears

**Common errors and fixes:**

**Error: `DatasetNotFoundError` or `401 Unauthorized`**
```
# Your HF token is wrong or you didn't accept dataset terms
# 1. Re-check that you accepted terms at:
#    huggingface.co/datasets/mozilla-foundation/common_voice_17_0
# 2. Verify your token:
python3 -c "from huggingface_hub import whoami; print(whoami())"
# Should print your username. If it errors, your HF_TOKEN is invalid.
# Get a new token at huggingface.co/settings/tokens
```

**Error: `CUDA out of memory`**
```bash
# Reduce batch size for smoke test only:
# Edit configs/finetune_whisper.yaml:
sed -i 's/per_device_train_batch_size: 16/per_device_train_batch_size: 8/' configs/finetune_whisper.yaml
sed -i 's/per_device_eval_batch_size: 8/per_device_eval_batch_size: 4/' configs/finetune_whisper.yaml
# Then rerun the smoke test
```

**Error: `No space left on device`**
```bash
df -h /workspace
# If /workspace is full, you need more volume disk.
# Stop: the pod needs to be recreated with more volume space.
# Clean up first and try:
rm -rf /workspace/.cache/huggingface/hub  # removes downloaded model weights from cache
# Then rerun
```

**Error: `ModuleNotFoundError: No module named 'evaluate'`**
```bash
pip install evaluate>=0.4 jiwer
python3 scripts/finetune_whisper.py --config configs/finetune_whisper.yaml --smoke_test
```

---

## PART 4 — FULL TRAINING
*~3–6 hours unattended — start before bed*

---

### 4.1 — Start a tmux Session

tmux keeps training running even if your SSH connection drops. This is critical — without it,
closing your terminal kills training.

```bash
# Create a new tmux session named "train"
tmux new -s train
```

You are now inside tmux. The green bar at the bottom confirms you are in the session.

---

### 4.2 — Launch Training

```bash
cd /workspace/paryaya

python3 scripts/finetune_whisper.py \
    --config configs/finetune_whisper.yaml
```

**What you will see:**

```
Loading processor from openai/whisper-medium ...
Loading mozilla-foundation/common_voice_17_0 (ne) ...
Pre-processing audio + tokenising transcripts ...
  train=8,943  eval=2,843

Loading openai/whisper-medium weights ...

🚀 Training on cuda | steps=4000 | fp16=True
   Checkpoint dir: checkpoints/whisper-medium-nepali

{'loss': 8.4123, 'grad_norm': 2.341, 'learning_rate': 2e-08, 'epoch': 0.09}
{'loss': 7.8234, 'grad_norm': 2.198, 'learning_rate': 4e-07, 'epoch': 0.18}
...
```

**Normal behaviour — what to expect at each stage:**

| Steps | Loss | WER | What is happening |
|---|---|---|---|
| 1–100 | 7–9 | 80–95% | Learning to output Devanagari at all — looks terrible, completely normal |
| 100–500 | 5–7 | 50–80% | Learning Nepali phoneme patterns |
| 500–1500 | 3–5 | 30–50% | Starting to produce real words |
| 1500–3000 | 1.5–3 | 15–30% | Refining word boundaries and common vocabulary |
| 3000–4000 | 0.8–2 | 8–20% | Final polish — target is WER < 20% |

> **Do not panic at early WER values of 80–90%.** This is normal for the first few hundred steps.
> The model already understands Nepali from pre-training — it just needs fine-tuning alignment.

---

### 4.3 — Detach from tmux (leave training running)

**Press `Ctrl+B`, then `D`** — this detaches without stopping training.

You will see:
```
[detached (from session train)]
```

You can now safely close your terminal or turn off your laptop. Training continues on RunPod.

**To reattach and check progress:**
```bash
# SSH back into the pod, then:
tmux attach -t train
```

---

### 4.4 — Monitor Training Progress

**Option A — W&B Dashboard (recommended)**

Go to **wandb.ai/YOUR_USERNAME/paryaya-whisper** in your browser.

You will see live charts for:
- `train/loss` — should decrease steadily
- `eval/wer` — key metric, want this below 0.20 (20%) by end of training
- `eval/loss` — should decrease with small bumps

**Option B — From the terminal**

Open a second SSH connection to the pod and run:

```bash
# Check GPU is being used (should be 85-99%)
watch -n 5 nvidia-smi

# Check checkpoints are being saved
ls -lht /workspace/paryaya/checkpoints/whisper-medium-nepali/
```

You should see checkpoint folders like `checkpoint-500`, `checkpoint-1000`, etc. and a
`trainer_state.json` that updates every logging step.

---

### 4.5 — If Your SSH Connection Drops Mid-Training

Training is still running on the pod (tmux keeps it alive). Just reconnect:

```bash
# From your Mac terminal:
ssh root@XX.XX.XX.XX -p XXXXX -i ~/.ssh/id_ed25519

# Reattach to the training session:
tmux attach -t train
```

If you see `can't find session train` (the pod restarted):

```bash
cd /workspace/paryaya
source ~/.bashrc  # reloads your HF_TOKEN and WANDB keys

# Resume from last checkpoint:
python3 scripts/finetune_whisper.py \
    --config configs/finetune_whisper.yaml
```

> The script automatically resumes from the latest checkpoint via
> `Seq2SeqTrainer`'s `resume_from_checkpoint` behaviour. Check the
> `checkpoints/whisper-medium-nepali/` folder — whichever checkpoint exists will be loaded.

Actually to be explicit about resume, run:
```bash
# Find the latest checkpoint:
ls checkpoints/whisper-medium-nepali/

# Start a new tmux session and resume:
tmux new -s train
cd /workspace/paryaya
python3 - <<'EOF'
from transformers import Seq2SeqTrainer
# The trainer auto-detects checkpoints in output_dir and resumes
# Just rerun the same command — it picks up from the latest checkpoint
EOF

python3 scripts/finetune_whisper.py --config configs/finetune_whisper.yaml
```

---

### 4.6 — When Training Finishes

You will see:
```
✅ Training complete.
   Best model saved → checkpoints/whisper-medium-nepali/best

   Deploy to Paryaya API:
   export ASR_BACKEND=whisper
   export WHISPER_MODEL_PATH=checkpoints/whisper-medium-nepali/best
   uvicorn paryaya.api.main:app --host 0.0.0.0 --port 8000
```

Check the final WER:
```bash
cat /workspace/paryaya/checkpoints/whisper-medium-nepali/trainer_state.json | \
    python3 -c "import sys, json; s = json.load(sys.stdin); \
    best = min(s['log_history'], key=lambda x: x.get('eval_wer', 999)); \
    print(f'Best WER: {best.get(\"eval_wer\", \"N/A\"):.1%} at step {best.get(\"step\", \"?\")}')
"
```

**Target:** WER below 30% means the fine-tuning worked. Below 20% is excellent.
If WER is above 40%: contact — something went wrong with the dataset or training.

---

### 4.7 — Early Stopping Behaviour

The config includes `early_stopping_patience: 5`. This means if WER does not improve for
5 consecutive evaluations (5 × 500 = 2500 steps), training stops automatically.

If training stops before step 4000, that is fine — early stopping found the best point.
The best checkpoint is still saved.

---

## PART 5 — DOWNLOAD THE MODEL AND TERMINATE THE POD
*~15 minutes — do not skip any step here*

> **CRITICAL:** Once you terminate the pod, the volume data is deleted.
> Download everything before clicking Terminate.

---

### 5.1 — Check the Model Was Saved

```bash
# On the RunPod terminal:
ls -lh /workspace/paryaya/checkpoints/whisper-medium-nepali/best/
```

Expected output (the exact files may vary but you need all of these):
```
config.json
generation_config.json
model.safetensors        (or pytorch_model.bin — ~1.5 GB)
preprocessor_config.json
special_tokens_map.json
tokenizer.json
tokenizer_config.json
vocab.json
added_tokens.json
normalizer.json
merges.txt
```

If the `best/` folder is empty or missing, the model checkpoint was not saved. Check:
```bash
ls /workspace/paryaya/checkpoints/whisper-medium-nepali/
# Look for checkpoint-XXXX folders. Use the one with the highest number:
ls /workspace/paryaya/checkpoints/whisper-medium-nepali/checkpoint-3500/
```

If there are checkpoint folders but no `best/`, manually save the best one:
```bash
python3 - <<'EOF'
import json
from pathlib import Path
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Find best checkpoint from trainer state
state_file = Path("checkpoints/whisper-medium-nepali/trainer_state.json")
state = json.loads(state_file.read_text())
best_ckpt = state.get("best_model_checkpoint", "checkpoints/whisper-medium-nepali/checkpoint-500")
print(f"Best checkpoint: {best_ckpt}")

# Save it in HF format
model = WhisperForConditionalGeneration.from_pretrained(best_ckpt)
proc  = WhisperProcessor.from_pretrained(best_ckpt)

best_dir = Path("checkpoints/whisper-medium-nepali/best")
best_dir.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(best_dir))
proc.save_pretrained(str(best_dir))
print(f"Saved to {best_dir}")
EOF
```

---

### 5.2 — Download the Model to Your Mac

Open a **new terminal on your Mac** (do not close the RunPod SSH session yet).

```bash
# On your Mac — create a folder for the model:
mkdir -p ~/paryaya_model/whisper-medium-nepali
cd ~/paryaya_model

# Download the entire best/ folder (all HuggingFace model files)
# Replace XX.XX.XX.XX and XXXXX with your pod IP and port
scp -P XXXXX -r \
    root@XX.XX.XX.XX:/workspace/paryaya/checkpoints/whisper-medium-nepali/best/ \
    ~/paryaya_model/whisper-medium-nepali/

# This takes 3-8 minutes (about 1.5 GB)
```

**Verify the download:**
```bash
ls -lh ~/paryaya_model/whisper-medium-nepali/
# Should show model.safetensors at ~1.5 GB and 8-10 other config files

# Check total size
du -sh ~/paryaya_model/
# Should show ~1.5-1.6 GB
```

**Verify the model loads correctly:**
```bash
cd ~/paryaya_model
python3 - <<'EOF'
from transformers import WhisperForConditionalGeneration, WhisperProcessor
model = WhisperForConditionalGeneration.from_pretrained("whisper-medium-nepali")
proc  = WhisperProcessor.from_pretrained("whisper-medium-nepali")
print(f"✅ Model loaded: {model.num_parameters():,} parameters")
print(f"✅ Language: {model.generation_config.language}")
print(f"✅ Task: {model.generation_config.task}")
EOF
```

Expected output:
```
✅ Model loaded: 307,198,976 parameters
✅ Language: nepali
✅ Task: transcribe
```

---

### 5.3 — Copy the Model to Your Paryaya Project

```bash
# Copy to the paryaya project checkpoints folder
cp -r ~/paryaya_model/whisper-medium-nepali/ \
    /Users/yaxzyra/Documents/asti-lab/paryaya/checkpoints/whisper-medium-nepali/

echo "Done — model is at:"
ls /Users/yaxzyra/Documents/asti-lab/paryaya/checkpoints/whisper-medium-nepali/
```

---

### 5.4 — Terminate the RunPod Pod

1. Go to **runpod.io** → **My Pods**
2. Find your pod (it shows as "Running")
3. Click the **three dots (...)** menu on the right side of the pod
4. Click **Terminate Pod**
5. Confirm by clicking **Terminate** in the dialog
6. The pod disappears from the list and billing stops immediately

> **Do NOT just stop the pod** — a stopped pod still charges for volume storage.
> Terminate it completely once you have downloaded all files.

**After terminating, check your spend:**
Go to **runpod.io → Billing → Usage** to confirm the charges. Should be under $20.

---

## PART 6 — DEPLOY THE FINE-TUNED MODEL VIA PARYAYA API
*~10 minutes*

---

### 6.1 — Run Locally First (Quick Test)

Test the fine-tuned model locally before deploying to a server.

```bash
cd /Users/yaxzyra/Documents/asti-lab/paryaya

# Activate your virtual environment
source .venv/bin/activate

# Set env vars to use the fine-tuned Whisper
export ASR_BACKEND=whisper
export WHISPER_MODEL_PATH=/Users/yaxzyra/Documents/asti-lab/paryaya/checkpoints/whisper-medium-nepali

# Start the API
uvicorn paryaya.api.main:app --host 0.0.0.0 --port 8000
```

In the startup logs you should see:
```
INFO  Starting Paryaya API | backend=whisper device=mps
INFO  Whisper backend loaded: .../checkpoints/whisper-medium-nepali
INFO  Application startup complete.
```

Leave it running and open a new terminal to test it:

```bash
# Test health endpoint
curl http://localhost:8000/health
# Expected: {"status":"ok","model_loaded":true,"backend":"whisper","device":"mps","version":"1.0.0"}

# Test with a real Nepali audio file
curl -X POST http://localhost:8000/v1/transcribe \
    -H "Authorization: Bearer sk-paryaya-testkey123" \
    -F "file=@/path/to/nepali_audio.wav"
```

---

### 6.2 — Update the Docker Compose Config for Production

When you deploy to your server, set these environment variables:

```bash
# In your .env file (or docker-compose.yml environment section):
ASR_BACKEND=whisper
WHISPER_MODEL_PATH=/app/checkpoints/whisper-medium-nepali
```

**In docker/docker-compose.yml**, add the model mount:

```yaml
services:
  api:
    environment:
      - ASR_BACKEND=whisper
      - WHISPER_MODEL_PATH=/app/checkpoints/whisper-medium-nepali
    volumes:
      - ./checkpoints:/app/checkpoints  # add this line if not already present
```

Then deploy:
```bash
# On your production server:
docker compose -f docker/docker-compose.yml up -d
docker compose -f docker/docker-compose.yml logs -f api
```

---

### 6.3 — Push the Fine-Tuned Model to HuggingFace Hub (Optional)

Storing the model on HuggingFace Hub lets you pull it from any server without SCP.

```bash
cd /Users/yaxzyra/Documents/asti-lab/paryaya

python3 - <<'EOF'
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from huggingface_hub import HfApi

model = WhisperForConditionalGeneration.from_pretrained("checkpoints/whisper-medium-nepali")
proc  = WhisperProcessor.from_pretrained("checkpoints/whisper-medium-nepali")

# Push to your HF account (creates asticrat/whisper-medium-nepali)
model.push_to_hub("asticrat/whisper-medium-nepali", private=True)
proc.push_to_hub("asticrat/whisper-medium-nepali", private=True)
print("Pushed to HuggingFace Hub!")
EOF
```

After pushing, you can use the HF repo name instead of a local path:
```bash
export WHISPER_MODEL_PATH=asticrat/whisper-medium-nepali
```

---

## TROUBLESHOOTING

---

### Training Problems

**Problem: Loss is 8+ and not decreasing after 200 steps**

Normal for the first 100 steps. If still stuck at step 200+:
```bash
# Check that fp16 is working (not falling back to fp32)
grep "fp16" configs/finetune_whisper.yaml
# Should show: fp16: true

# Check GPU memory usage during training
# In a second SSH connection:
watch -n 2 nvidia-smi
# GPU memory should be 25-38 GB used. If under 10 GB, fp16 is not being used.
```

**Problem: `CUDA out of memory`**

```bash
# Reduce batch size:
nano configs/finetune_whisper.yaml
# Change: per_device_train_batch_size: 16  →  8
# Change: per_device_eval_batch_size: 8    →  4
# Change: gradient_accumulation_steps: 2   →  4  (keeps effective batch = 32)
# Save and rerun
```

**Problem: Training step speed is very slow (>5 seconds per step)**

```bash
# Check if dataloader workers are the issue:
# In configs/finetune_whisper.yaml, the script hardcodes 4 workers for cuda.
# This should be fine on A100. If slow, it may be the audio preprocessing map().
# The first run includes offline preprocessing which is slow; subsequent steps are fast.
```

**Problem: W&B not logging / "wandb: ERROR"**

```bash
# Re-login:
wandb login --relogin
# Paste your API key

# Or disable W&B and train without it:
export WANDB_MODE=disabled
python3 scripts/finetune_whisper.py --config configs/finetune_whisper.yaml
```

**Problem: `evaluate.load("wer")` fails**

```bash
pip install evaluate>=0.4 jiwer>=3.0
python3 -c "import evaluate; m = evaluate.load('wer'); print('WER metric OK')"
```

---

### Connection Problems

**Problem: SSH disconnects every few minutes**

```bash
# Add keepalive to your SSH command:
ssh root@XX.XX.XX.XX -p XXXXX -i ~/.ssh/id_ed25519 \
    -o ServerAliveInterval=60 \
    -o ServerAliveCountMax=10
```

**Problem: `ssh: connect to host github.com port 22: Operation timed out`**

```bash
# This is a network restriction. Add to ~/.ssh/config on your Mac:
nano ~/.ssh/config
# Add:
# Host github.com
#   Hostname ssh.github.com
#   Port 443
#   AddKeysToAgent yes
#   IdentityFile ~/.ssh/id_ed25519
```

**Problem: Pod IP/port changed after reconnect**

1. Go to runpod.io → My Pods
2. Click **Connect** on your pod
3. Copy the new SSH command (IP and port may change after restart)

---

### Dataset Problems

**Problem: `EmptyDatasetError`**

Your HF token is missing or the dataset terms were not accepted.

```bash
# 1. Check token:
python3 -c "
import os
from huggingface_hub import whoami
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN', '')
try:
    info = whoami()
    print(f'Logged in as: {info[\"name\"]}')
except Exception as e:
    print(f'Token invalid: {e}')
"

# 2. Go to huggingface.co/datasets/mozilla-foundation/common_voice_17_0
#    and verify you see dataset files (not an access wall)

# 3. Re-export token and retry
export HF_TOKEN=hf_your_token_here
python3 scripts/finetune_whisper.py --config configs/finetune_whisper.yaml --smoke_test
```

**Problem: Dataset download is very slow**

RunPod usually has fast internet (~500 Mbps). If it is slow:
```bash
# Check network speed
curl -o /dev/null -s https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0/resolve/main/README.md \
    -w "Speed: %{speed_download} bytes/sec\n"
```

Common Voice Nepali is ~8-15 GB and should download in 5-20 minutes on RunPod.

---

## QUICK REFERENCE

### Commands You Will Use Most

```bash
# Reconnect to pod
ssh root@XX.XX.XX.XX -p XXXXX -i ~/.ssh/id_ed25519

# Reattach to training
tmux attach -t train

# Check GPU usage
nvidia-smi

# Check training is alive
ls -lt /workspace/paryaya/checkpoints/whisper-medium-nepali/

# Check disk space
df -h /workspace

# Run smoke test
python3 /workspace/paryaya/scripts/finetune_whisper.py \
    --config /workspace/paryaya/configs/finetune_whisper.yaml \
    --smoke_test

# Download model (run on Mac)
scp -P XXXXX -r root@XX.XX.XX.XX:/workspace/paryaya/checkpoints/whisper-medium-nepali/best/ \
    ~/paryaya_model/whisper-medium-nepali/
```

### Cost Summary

| Action | Duration | Cost |
|---|---|---|
| Setup + smoke test | ~30 min | ~$1.00 |
| Full training | ~3–6 hours | ~$6–12 |
| Download + terminate | ~20 min | ~$0.60 |
| **Total** | **~4–7 hours** | **~$8–14** |

At $1.89/hr on A100 PCIe 40GB, a 6-hour run costs $11.34. Well under $20.

### Environment Variables for Deployment

```bash
# Required
ASR_BACKEND=whisper
WHISPER_MODEL_PATH=/path/to/checkpoints/whisper-medium-nepali

# Optional (existing API vars work unchanged)
MAX_AUDIO_MB=50
BEAM_WIDTH=5  # lower = faster inference, slightly lower accuracy
```

---

*Paryaya — पर्याय*
