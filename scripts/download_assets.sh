#!/usr/bin/env bash
# Download only missing assets. Hugging Face resumes partial snapshots safely.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${HF_ENDPOINT:=https://hf-mirror.com}"
: "${HF_HUB_DISABLE_XET:=1}"
export HF_ENDPOINT HF_HUB_DISABLE_XET

command -v hf >/dev/null || {
  echo "Missing 'hf' CLI. Install dependencies first: pip install -r requirements.txt" >&2
  exit 1
}

download_if_missing() {
  local label="$1"
  local required_file="$2"
  shift 2
  if [[ -s "${required_file}" ]]; then
    echo "[skip] ${label}: ${required_file} already exists"
  else
    echo "[download/resume] ${label}"
    "$@"
  fi
}

mkdir -p "${PROJECT_ROOT}/models" "${PROJECT_ROOT}/data"

download_if_missing \
  "Qwen2.5-1.5B-Instruct" \
  "${PROJECT_ROOT}/models/Qwen2.5-1.5B-Instruct/model.safetensors" \
  hf download Qwen/Qwen2.5-1.5B-Instruct \
    --local-dir "${PROJECT_ROOT}/models/Qwen2.5-1.5B-Instruct"

download_if_missing \
  "UltraFeedback Binarized" \
  "${PROJECT_ROOT}/data/ultrafeedback_binarized/data/train-00000-of-00001.parquet" \
  hf download trl-lib/ultrafeedback_binarized --repo-type dataset \
    --local-dir "${PROJECT_ROOT}/data/ultrafeedback_binarized"

download_if_missing \
  "IFEval" \
  "${PROJECT_ROOT}/data/ifeval/ifeval_input_data.jsonl" \
  hf download google/IFEval --repo-type dataset \
    --local-dir "${PROJECT_ROOT}/data/ifeval"

download_if_missing \
  "GSM8K" \
  "${PROJECT_ROOT}/data/gsm8k/main/test-00000-of-00001.parquet" \
  hf download openai/gsm8k --repo-type dataset \
    --local-dir "${PROJECT_ROOT}/data/gsm8k"
