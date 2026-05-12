#!/usr/bin/env bash
# Download trained model artifacts from a RunPod instance.
#
# Usage:
#   bash scripts/export_model.sh <runpod_ip> <ssh_port>
#
# Example:
#   bash scripts/export_model.sh 192.168.1.100 22
#
# Requires: ssh, scp, and an SSH key configured for the pod.
set -euo pipefail

RUNPOD_IP="${1:?Usage: $0 <runpod_ip> <ssh_port>}"
SSH_PORT="${2:-22}"
REMOTE_USER="${REMOTE_USER:-root}"
REMOTE_DIR="${REMOTE_DIR:-/app}"

SSH_OPTS="-p ${SSH_PORT} -o StrictHostKeyChecking=no -o ConnectTimeout=10"

echo "📡  Connecting to ${REMOTE_USER}@${RUNPOD_IP}:${SSH_PORT} ..."

# Verify connection
ssh ${SSH_OPTS} "${REMOTE_USER}@${RUNPOD_IP}" "echo '✅ SSH OK'"

FILES=(
  "checkpoints/best_model.pt"
  "data/vocab/nepali_vocab.json"
  "configs/model_medium.yaml"
)

mkdir -p checkpoints data/vocab configs

echo ""
echo "⬇️   Downloading model artifacts ..."
echo ""

for REMOTE_FILE in "${FILES[@]}"; do
  LOCAL_FILE="${REMOTE_FILE}"
  echo -n "  ${REMOTE_FILE} ... "
  scp ${SSH_OPTS} \
    "${REMOTE_USER}@${RUNPOD_IP}:${REMOTE_DIR}/${REMOTE_FILE}" \
    "${LOCAL_FILE}"
  SIZE=$(du -sh "${LOCAL_FILE}" | cut -f1)
  echo "${SIZE}"
done

echo ""
echo "✅  Download complete."
echo ""
echo "File sizes:"
du -sh checkpoints/best_model.pt data/vocab/nepali_vocab.json configs/model_medium.yaml
echo ""
echo "⚠️   You may now terminate the RunPod instance to avoid further charges."
echo "    RunPod dashboard: https://www.runpod.io/console/pods"
