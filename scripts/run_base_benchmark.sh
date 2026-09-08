#!/usr/bin/env bash
# Run the complete immutable pre-DPO benchmark after models/ and data/ are ready.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVALUATOR_ROOT="${PROJECT_ROOT}/third_party/google-research"

if [[ ! -f "${EVALUATOR_ROOT}/instruction_following_eval/evaluation_lib.py" ]]; then
  mkdir -p "${PROJECT_ROOT}/third_party"
  git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/google-research/google-research.git "${EVALUATOR_ROOT}"
  git -C "${EVALUATOR_ROOT}" sparse-checkout set instruction_following_eval
fi

if python -c "import nltk; nltk.data.find('tokenizers/punkt_tab')"; then
  echo "[skip] NLTK punkt_tab already exists"
else
  python -m nltk.downloader punkt_tab
fi
python "${PROJECT_ROOT}/scripts/benchmark_base_model.py"
python "${PROJECT_ROOT}/scripts/finalize_base_metrics.py"
