#!/bin/bash
set -euo pipefail

TAG="$1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER_REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTER_REPO="$(cd "${INNER_REPO}/.." && pwd)"

if [ "${TAG}" = "JacobiForcing_Coder_7B_v1" ]; then
  MODEL_DIR="${OUTER_REPO}/models/JacobiForcing_Coder_7B_v1"
  CKPT_DIR=""
  IS_OFFICIAL_MODEL=1
else
  MODEL_DIR="${OUTER_REPO}/outputs/${TAG}"
  IS_OFFICIAL_MODEL=0

  # Infer checkpoint step from tag like *_s12000_* / *_s15000_* / *_s20000_*.
  STEP_FROM_TAG=""
  if [[ "${TAG}" =~ _s([0-9]+) ]]; then
    STEP_FROM_TAG="${BASH_REMATCH[1]}"
  fi

  if [ -n "${STEP_FROM_TAG}" ]; then
    CKPT_DIR="${MODEL_DIR}/checkpoint-${STEP_FROM_TAG}"
  else
    CKPT_DIR=""
  fi
fi

echo "======================================================================"
echo "[PREP CONFIG]"
echo "TAG=${TAG}"
echo "MODEL_DIR=${MODEL_DIR}"
echo "CKPT_DIR=${CKPT_DIR}"
echo "IS_OFFICIAL_MODEL=${IS_OFFICIAL_MODEL}"
echo "======================================================================"

if [ ! -d "${MODEL_DIR}" ]; then
  echo "[ERROR] MODEL_DIR not found: ${MODEL_DIR}"
  exit 1
fi

if [ "${IS_OFFICIAL_MODEL}" -eq 1 ]; then
  echo "[OFFICIAL MODEL] Skip preparation."
  exit 0
fi

HAS_HF_WEIGHTS=0

if ls "${MODEL_DIR}"/pytorch_model-*.bin >/dev/null 2>&1 && [ -f "${MODEL_DIR}/pytorch_model.bin.index.json" ]; then
  HAS_HF_WEIGHTS=1
fi

if [ -f "${MODEL_DIR}/model.safetensors" ]; then
  HAS_HF_WEIGHTS=1
fi

if ls "${MODEL_DIR}"/model-*.safetensors >/dev/null 2>&1; then
  HAS_HF_WEIGHTS=1
fi

if [ "${HAS_HF_WEIGHTS}" -eq 0 ]; then
  echo "======================================================================"
  echo "[CONVERT] root HF shards not found. Trying ZeRO -> fp32 conversion."
  echo "======================================================================"

  if [ -z "${CKPT_DIR}" ] || [ ! -d "${CKPT_DIR}" ]; then
    echo "[ERROR] checkpoint dir not found: ${CKPT_DIR}"
    exit 1
  fi

  if [ ! -f "${CKPT_DIR}/zero_to_fp32.py" ]; then
    echo "[ERROR] zero_to_fp32.py not found in ${CKPT_DIR}"
    exit 1
  fi

  # If there is a tiny placeholder file, move it away.
  if [ -f "${MODEL_DIR}/pytorch_model.bin" ]; then
    size_bytes=$(stat -c%s "${MODEL_DIR}/pytorch_model.bin")
    if [ "${size_bytes}" -lt 1000000000 ]; then
      echo "[BACKUP] tiny pytorch_model.bin -> pytorch_model.bin.placeholder"
      mv "${MODEL_DIR}/pytorch_model.bin" "${MODEL_DIR}/pytorch_model.bin.placeholder" || true
    fi
  fi

  cd "${CKPT_DIR}"

  # IMPORTANT:
  # Save sharded HF model files directly into MODEL_DIR, not MODEL_DIR/pytorch_model.bin directory.
  python zero_to_fp32.py . "${MODEL_DIR}"

  cd - >/dev/null
else
  echo "[PREP] HF weights already exist. Skip ZeRO conversion."
fi

# Prefix fix if needed.
FIX_PREFIX_SCRIPT="${OUTER_REPO}/fix_zero_fp32_prefix.py"

if [ -f "${MODEL_DIR}/pytorch_model-00001-of-00007.bin" ]; then
  NEED_FIX=$(python - <<PY
import torch
p="${MODEL_DIR}/pytorch_model-00001-of-00007.bin"
sd=torch.load(p, map_location="cpu")
print(int(any(k.startswith("module.") or k.startswith("model.module.") for k in sd.keys())))
PY
)
  if [ "${NEED_FIX}" = "1" ]; then
    echo "======================================================================"
    echo "[PREFIX FIX] Removing module/model.module prefix"
    echo "======================================================================"
    python "${FIX_PREFIX_SCRIPT}" "${MODEL_DIR}"
  else
    echo "[PREFIX FIX] Not needed."
  fi
else
  echo "[PREFIX FIX] No pytorch_model-00001-of-00007.bin found; skip."
fi

echo "======================================================================"
echo "[PREP DONE]"
ls -lh "${MODEL_DIR}" | grep -E "config|pytorch_model|model.*safetensors|tokenizer" || true
echo "======================================================================"