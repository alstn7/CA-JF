#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# 0916 Rebuttal control training queue
#
# Order:
#   T1 Code matched JF, seed42
#   T2 Math matched JF, seed42
#   T5 Code alpha=0 + hybrid CTM, seed42
#   T6 Code alpha=0 + residual CTM, seed42
#   T7 Code alpha=0 + anchor CTM, seed42
#   T8 Code random-anchor Full CA-JF, seed42
#   T3 Code Full CA-JF, seed43
#   T4 Code Full CA-JF, seed44
#
# Training uses all 4 GPUs sequentially.
#
# Important:
#   - S1 is immediately converted from ZeRO-3 -> HF shards and validated,
#     because S2 must load the prepared S1.
#   - S2 is NOT prepared until all training jobs are finished.
#   - Root tiny pytorch_model.bin is NEVER treated as a completed model.
#   - Existing incomplete output dirs cause a hard stop rather than being
#     silently overwritten/resumed.
###############################################################################

###############################################################################
# Paths
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER_REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTER_REPO="$(cd "${INNER_REPO}/.." && pwd)"

AUTO_ANCHOR_VOCAB="${AUTO_ANCHOR_VOCAB:-/home/irteam/data-vol1/minsu/CA-JacobiForcing/analysis/automatic_anchor_global_median_token_ids.json}"

cd "${INNER_REPO}"

MP_JACOBI_ENV="/home/irteam/data-vol1/conda/envs/mp_jacobi"
export PATH="${MP_JACOBI_ENV}/bin:${PATH}"
hash -r


TRAIN_PY="${INNER_REPO}/train/soft_flexattn_train_cllm_multiblock_auto.py"
TRAINER_PY="${INNER_REPO}/train/soft_flexattn_cllm_trainer_multiblock_auto.py"
PREP_SCRIPT="${INNER_REPO}/scripts/train/prepare_jacobi_model_for_eval.sh"

CODE_DS="${INNER_REPO}/scripts/train/ds_config_cpu_offloading.json"
MATH_DS="${INNER_REPO}/scripts/train/ds_config_cpu_offloading_epjf_gas1.json"

CODE_MODEL="${OUTER_REPO}/models/Qwen2.5-Coder-7B-Instruct"
MATH_MODEL="${OUTER_REPO}/models/Qwen2.5-Math-7B-Instruct"

CODE_S1_DATA="${OUTER_REPO}/OpenCodeInstruct_training_data_n16w16/merged_data_v2_8_30_opencodeinstruct_progressive_noise_cyclic_all.jsonl"
CODE_S2_DATA="${OUTER_REPO}/OpenCodeInstruct_training_data_n32w16/merged-traj-data-oct-16-n32w16.jsonl"

MATH_S1_DATA="${OUTER_REPO}/OpenThoughts_Math_training_data_n16w16/data"
MATH_S2_DATA="${OUTER_REPO}/OpenThoughts_Math_training_data_n64w32/data"

###############################################################################
# Runtime
###############################################################################

RUN_ID="${RUN_ID:-20260916_rebuttal_v1}"

STEPS="${STEPS:-10000}"
SAVE_STEPS="${SAVE_STEPS:-2000}"

NPROC="${NPROC:-4}"
PER_DEVICE_BS="${PER_DEVICE_BS:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
GLOBAL_BATCH=$((NPROC * PER_DEVICE_BS * GRAD_ACCUM))

LR="${LR:-1e-6}"

# Respect caller-provided CUDA_VISIBLE_DEVICES.
# Default remains 1,2,3,4 only when nothing is specified.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4}"

export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256}"

# Code main did not use the Math dense/eager overrides.
unset JF_LOAD_ATTN_IMPL || true
unset JF_ATTN_BACKEND || true

PREPARE_FINALS_AFTER_TRAIN="${PREPARE_FINALS_AFTER_TRAIN:-1}"

LOG_ROOT="${OUTER_REPO}/outputs/logs_0916_rebuttal_controls_${RUN_ID}"
MANIFEST="${LOG_ROOT}/manifest.tsv"

mkdir -p "${LOG_ROOT}"

###############################################################################
# Logging / checks
###############################################################################

log () {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

die () {
    echo "[ERROR] $*" >&2
    exit 1
}

check_file () {
    [[ -f "$1" ]] || die "Missing file: $1"
}

check_dir () {
    [[ -d "$1" ]] || die "Missing directory: $1"
}

###############################################################################
# Strict raw ZeRO checkpoint completion test
###############################################################################

raw_training_done () {
    local OUT_DIR="$1"
    local CKPT="${OUT_DIR}/checkpoint-${STEPS}"

    [[ -d "${CKPT}" ]] || return 1
    [[ -f "${CKPT}/zero_to_fp32.py" ]] || return 1
    [[ -f "${CKPT}/trainer_state.json" ]] || return 1
    [[ -f "${CKPT}/latest" ]] || return 1
    [[ -d "${CKPT}/global_step${STEPS}" ]] || return 1

    [[ "$(tr -d '[:space:]' < "${CKPT}/latest")" == "global_step${STEPS}" ]] || return 1

    python - "${CKPT}/trainer_state.json" "${STEPS}" <<'PY'
import json
import sys

path = sys.argv[1]
expected = int(sys.argv[2])

try:
    obj = json.load(open(path))
    actual = int(obj.get("global_step", -1))
except Exception:
    raise SystemExit(1)

raise SystemExit(0 if actual == expected else 1)
PY
}

assert_raw_training_done () {
    local OUT_DIR="$1"

    if ! raw_training_done "${OUT_DIR}"; then
        echo "======================================================================"
        echo "[RAW CHECK FAILED]"
        echo "OUT_DIR=${OUT_DIR}"
        echo "Expected checkpoint-${STEPS} with:"
        echo "  zero_to_fp32.py"
        echo "  trainer_state.json global_step=${STEPS}"
        echo "  latest=global_step${STEPS}"
        echo "  global_step${STEPS}/"
        echo "======================================================================"
        find "${OUT_DIR}" -maxdepth 2 -type f 2>/dev/null | sort | tail -80 || true
        exit 1
    fi
}

###############################################################################
# HF shard validation
###############################################################################

hf_files_complete () {
    local MODEL_DIR="$1"

    python - "${MODEL_DIR}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
idx = root / "pytorch_model.bin.index.json"

if not idx.is_file():
    raise SystemExit(1)

try:
    obj = json.loads(idx.read_text())
    weight_map = obj["weight_map"]
except Exception:
    raise SystemExit(1)

if not weight_map:
    raise SystemExit(1)

shards = sorted(set(weight_map.values()))

if not shards:
    raise SystemExit(1)

for shard in shards:
    p = root / shard
    if not p.is_file():
        raise SystemExit(1)
    if p.stat().st_size <= 0:
        raise SystemExit(1)

raise SystemExit(0)
PY
}

hf_prefix_clean () {
    local MODEL_DIR="$1"

    python - "${MODEL_DIR}" <<'PY'
import json
import sys
from pathlib import Path

idx = Path(sys.argv[1]) / "pytorch_model.bin.index.json"

if not idx.is_file():
    raise SystemExit(1)

try:
    obj = json.loads(idx.read_text())
    keys = obj["weight_map"].keys()
except Exception:
    raise SystemExit(1)

bad = any(
    k.startswith("module.") or k.startswith("model.module.")
    for k in keys
)

raise SystemExit(1 if bad else 0)
PY
}

hf_ready () {
    local MODEL_DIR="$1"
    hf_files_complete "${MODEL_DIR}" && hf_prefix_clean "${MODEL_DIR}"
}

has_partial_hf_artifacts () {
    local MODEL_DIR="$1"

    [[ -f "${MODEL_DIR}/pytorch_model.bin.index.json" ]] && return 0

    compgen -G "${MODEL_DIR}/pytorch_model-*.bin" >/dev/null 2>&1 && return 0

    return 1
}

validate_hf_model () {
    local MODEL_DIR="$1"

    python - "${MODEL_DIR}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
idx_path = root / "pytorch_model.bin.index.json"

if not idx_path.is_file():
    raise SystemExit(f"[FAIL] missing index: {idx_path}")

obj = json.loads(idx_path.read_text())
weight_map = obj["weight_map"]

keys = list(weight_map.keys())
shards = sorted(set(weight_map.values()))

bad = [
    k for k in keys
    if k.startswith("module.") or k.startswith("model.module.")
]

missing = [
    s for s in shards
    if not (root / s).is_file()
]

empty = [
    s for s in shards
    if (root / s).is_file() and (root / s).stat().st_size <= 0
]

print(f"[HF VALIDATE] model={root}")
print(f"[HF VALIDATE] num_keys={len(keys)}")
print(f"[HF VALIDATE] num_shards={len(shards)}")
print(f"[HF VALIDATE] bad_prefix={len(bad)}")
print(f"[HF VALIDATE] missing={len(missing)}")
print(f"[HF VALIDATE] empty={len(empty)}")

if bad:
    print("[HF VALIDATE] first_bad=", bad[:5])
    raise SystemExit(1)

if missing:
    print("[HF VALIDATE] missing_shards=", missing)
    raise SystemExit(1)

if empty:
    print("[HF VALIDATE] empty_shards=", empty)
    raise SystemExit(1)

print("[HF VALIDATE OK]")
PY
}

###############################################################################
# ZeRO -> HF prepare
###############################################################################

prepare_and_validate () {
    local TAG="$1"
    local MODEL_DIR="${OUTER_REPO}/outputs/${TAG}"
    local PREP_LOG="${LOG_ROOT}/${TAG}.prep.log"

    assert_raw_training_done "${MODEL_DIR}"

    if hf_ready "${MODEL_DIR}"; then
        log "[PREP SKIP] already HF-ready: ${TAG}"
        validate_hf_model "${MODEL_DIR}"
        return 0
    fi

    # Complete-but-prefixed HF shards are okay: PREP_SCRIPT will only
    # perform the prefix fix.
    if hf_files_complete "${MODEL_DIR}"; then
        log "[PREP] complete HF shards exist but prefix cleanup is required: ${TAG}"
    elif has_partial_hf_artifacts "${MODEL_DIR}"; then
        echo "======================================================================"
        echo "[ERROR] Partial HF conversion artifacts detected."
        echo "Refusing to let prepare_jacobi_model_for_eval.sh silently skip/overwrite."
        echo "MODEL_DIR=${MODEL_DIR}"
        echo "Inspect this directory manually."
        echo "======================================================================"
        ls -lh "${MODEL_DIR}" | grep -E 'pytorch_model|model.*safetensors|config' || true
        exit 1
    else
        log "[PREP START] ${TAG}"
    fi

    bash "${PREP_SCRIPT}" "${TAG}" 2>&1 | tee "${PREP_LOG}"

    if ! hf_ready "${MODEL_DIR}"; then
        die "HF preparation validation failed: ${MODEL_DIR}"
    fi

    validate_hf_model "${MODEL_DIR}"
    log "[PREP DONE] ${TAG}"
}

###############################################################################
# Code training
###############################################################################

train_code_stage () {
    local TAG="$1"
    local MODEL_PATH="$2"
    local DATA_PATH="$3"
    local BLOCK_LEN="$4"
    local PORT="$5"
    local SEED="$6"
    local ALPHA="$7"
    local CTM_MODE="$8"
    local CTM_LAMBDA="$9"
    local ANCHOR_CONTROL="${10}"
    local RANDOM_ANCHOR_SEED="${11}"

    local OUT_DIR="${OUTER_REPO}/outputs/${TAG}"
    local LOG_FILE="${LOG_ROOT}/${TAG}.train.log"

    if raw_training_done "${OUT_DIR}"; then
        log "[TRAIN SKIP] raw checkpoint already complete: ${TAG}"
        return 0
    fi

    if [[ -d "${OUT_DIR}" ]]; then
        if compgen -G "${OUT_DIR}/checkpoint-*" >/dev/null 2>&1; then
            log "[TRAIN RESUME] checkpoint found in ${OUT_DIR}"
            log "[TRAIN RESUME] train script will use resume_from_checkpoint=True"
            ls -d "${OUT_DIR}"/checkpoint-* 2>/dev/null | sort -V || true
        else
            echo "======================================================================"
            echo "[ERROR] Existing but incomplete output directory with NO checkpoint:"
            echo "${OUT_DIR}"
            echo
            echo "Cannot resume safely. Preserve/rename it before restarting."
            echo "======================================================================"
            exit 1
        fi
    fi

    mkdir -p "${OUT_DIR}"

    log "======================================================================"
    log "[CODE TRAIN START] ${TAG}"
    log "MODEL=${MODEL_PATH}"
    log "DATA=${DATA_PATH}"
    log "BLOCK_LEN=${BLOCK_LEN}"
    log "SEED=${SEED}"
    log "ALPHA=${ALPHA}"
    log "ANCHOR_CONS_TYPE=auto_influence"
    log "AUTO_ANCHOR_VOCAB=${AUTO_ANCHOR_VOCAB}"
    log "CTM_MODE=${CTM_MODE}"
    log "CTM_LAMBDA=${CTM_LAMBDA}"
    log "ANCHOR_CONTROL=${ANCHOR_CONTROL}"
    log "RANDOM_ANCHOR_SEED=${RANDOM_ANCHOR_SEED}"
    log "PORT=${PORT}"
    log "======================================================================"

    torchrun \
        --nnodes=1 \
        --nproc_per_node="${NPROC}" \
        --rdzv_id="${PORT}" \
        --rdzv_endpoint="localhost:${PORT}" \
        --master_port="${PORT}" \
        "${TRAIN_PY}" \
        --target_model_path "${MODEL_PATH}" \
        --tokenizer_path "${CODE_MODEL}" \
        --data_path "${DATA_PATH}" \
        --output_dir "${OUT_DIR}" \
        --deepspeed "${CODE_DS}" \
        --max_new_tokens "${BLOCK_LEN}" \
        --model_max_length 16384 \
        --bf16 True \
        --fp16 False \
        --report_to none \
        --do_train \
        --num_train_epochs 1 \
        --max_steps "${STEPS}" \
        --per_device_train_batch_size "${PER_DEVICE_BS}" \
        --gradient_accumulation_steps "${GRAD_ACCUM}" \
        --gradient_checkpointing True \
        --save_strategy steps \
        --save_steps "${SAVE_STEPS}" \
        --save_total_limit 2 \
        --learning_rate "${LR}" \
        --weight_decay 0.0 \
        --warmup_ratio 0.03 \
        --lr_scheduler_type cosine \
        --logging_steps 10 \
        --remove_unused_columns False \
        --lambda_anchor 0.0 \
        --anchor_weight 1.0 \
        --anchor_cons_type auto_influence \
        --anchor_auto_vocab_json "${AUTO_ANCHOR_VOCAB}" \
        --anchor_cons_alpha "${ALPHA}" \
        --anchor_cons_noise_gate mid \
        --anchor_cons_late_threshold 0.3 \
        --dist_cons_mode none \
        --lambda_contract 0.0 \
        --contract_margin 0.0 \
        --contract_on_anchor_only False \
        --prog_target_mode none \
        --lambda_prog_aux 0.0 \
        --ctm_aux_mode "${CTM_MODE}" \
        --lambda_ctm_aux "${CTM_LAMBDA}" \
        --ctm_top_ratio 0.20 \
        --ctm_residual_weight 0.0 \
        --ctm_score_power 1.0 \
        --ctm_max_weight 2.0 \
        --anchor_control_mode "${ANCHOR_CONTROL}" \
        --random_anchor_seed "${RANDOM_ANCHOR_SEED}" \
        --seed "${SEED}" \
        --data_seed "${SEED}" \
        --qlora False \
        2>&1 | tee "${LOG_FILE}"

    assert_raw_training_done "${OUT_DIR}"

    log "[CODE TRAIN DONE] ${TAG}"
}

###############################################################################
# Math matched-JF training
#
# Keeps main Math training recipe:
#   - Qwen2.5-Math-7B-Instruct
#   - n16w16 -> n64w32
#   - max length 2048
#   - constant_with_warmup, warmup_steps=100
#   - gradient checkpointing False
#   - dense/eager
#
# Only objective is changed to matched JF:
#   alpha=0, CTM=none, lambda=0
###############################################################################

train_math_matched_stage () {
    local TAG="$1"
    local MODEL_PATH="$2"
    local DATA_PATH="$3"
    local BLOCK_LEN="$4"
    local PORT="$5"
    local SEED="$6"

    local OUT_DIR="${OUTER_REPO}/outputs/${TAG}"
    local LOG_FILE="${LOG_ROOT}/${TAG}.train.log"

    if raw_training_done "${OUT_DIR}"; then
        log "[TRAIN SKIP] raw checkpoint already complete: ${TAG}"
        return 0
    fi

    if [[ -d "${OUT_DIR}" ]]; then
        if compgen -G "${OUT_DIR}/checkpoint-*" >/dev/null 2>&1; then
            log "[TRAIN RESUME] checkpoint found in ${OUT_DIR}"
            log "[TRAIN RESUME] train script will use resume_from_checkpoint=True"
            ls -d "${OUT_DIR}"/checkpoint-* 2>/dev/null | sort -V || true
        else
            echo "======================================================================"
            echo "[ERROR] Existing but incomplete output directory with NO checkpoint:"
            echo "${OUT_DIR}"
            echo
            echo "Cannot resume safely. Preserve/rename it before restarting."
            echo "======================================================================"
            exit 1
        fi
    fi

    mkdir -p "${OUT_DIR}"

    log "======================================================================"
    log "[MATH MATCHED-JF TRAIN START] ${TAG}"
    log "MODEL=${MODEL_PATH}"
    log "DATA=${DATA_PATH}"
    log "BLOCK_LEN=${BLOCK_LEN}"
    log "SEED=${SEED}"
    log "PORT=${PORT}"
    log "JF_LOAD_ATTN_IMPL=eager"
    log "JF_ATTN_BACKEND=dense"
    log "======================================================================"

    JF_LOAD_ATTN_IMPL=eager \
    JF_ATTN_BACKEND=dense \
    torchrun \
        --nnodes=1 \
        --nproc_per_node="${NPROC}" \
        --rdzv_id="${PORT}" \
        --rdzv_endpoint="localhost:${PORT}" \
        --master_port="${PORT}" \
        "${TRAIN_PY}" \
        --target_model_path "${MODEL_PATH}" \
        --tokenizer_path "${MATH_MODEL}" \
        --data_path "${DATA_PATH}" \
        --output_dir "${OUT_DIR}" \
        --deepspeed "${MATH_DS}" \
        --model_max_length 2048 \
        --max_new_tokens "${BLOCK_LEN}" \
        --per_device_train_batch_size "${PER_DEVICE_BS}" \
        --gradient_accumulation_steps "${GRAD_ACCUM}" \
        --learning_rate "${LR}" \
        --lr_scheduler_type constant_with_warmup \
        --warmup_steps 100 \
        --max_steps "${STEPS}" \
        --save_steps "${SAVE_STEPS}" \
        --save_total_limit 2 \
        --logging_steps 10 \
        --bf16 True \
        --fp16 False \
        --gradient_checkpointing False \
        --report_to none \
        --remove_unused_columns False \
        --anchor_cons_type math_plus_final \
        --anchor_cons_alpha 0.0 \
        --anchor_cons_noise_gate none \
        --anchor_cons_late_threshold 0.3 \
        --lambda_anchor 0.0 \
        --anchor_weight 1.0 \
        --dist_cons_mode none \
        --lambda_contract 0.0 \
        --contract_margin 0.0 \
        --contract_on_anchor_only False \
        --prog_target_mode none \
        --lambda_prog_aux 0.0 \
        --ctm_aux_mode none \
        --lambda_ctm_aux 0.0 \
        --anchor_control_mode semantic \
        --random_anchor_seed 42 \
        --seed "${SEED}" \
        --data_seed "${SEED}" \
        --qlora False \
        2>&1 | tee "${LOG_FILE}"

    assert_raw_training_done "${OUT_DIR}"

    log "[MATH MATCHED-JF TRAIN DONE] ${TAG}"
}

###############################################################################
# Variant wrappers
###############################################################################

declare -a FINAL_S2_TAGS=()

run_code_variant () {
    local SHORT="$1"
    local SEED="$2"
    local ALPHA="$3"
    local CTM_MODE="$4"
    local CTM_LAMBDA="$5"
    local ANCHOR_CONTROL="$6"
    local RANDOM_ANCHOR_SEED="$7"
    local PORT_BASE="$8"

    local S1_TAG="rebuttal_${SHORT}_gbs${GLOBAL_BATCH}_n16w16_s${STEPS}_lr1e6_${RUN_ID}"
    local S2_TAG="rebuttal_${SHORT}_gbs${GLOBAL_BATCH}_n16w16_to_n32w16_s${STEPS}_lr1e6_${RUN_ID}"

    echo "======================================================================"
    echo "[VARIANT START] ${SHORT}"
    echo "S1=${S1_TAG}"
    echo "S2=${S2_TAG}"
    echo "======================================================================"

    train_code_stage \
        "${S1_TAG}" \
        "${CODE_MODEL}" \
        "${CODE_S1_DATA}" \
        16 \
        "${PORT_BASE}" \
        "${SEED}" \
        "${ALPHA}" \
        "${CTM_MODE}" \
        "${CTM_LAMBDA}" \
        "${ANCHOR_CONTROL}" \
        "${RANDOM_ANCHOR_SEED}"

    prepare_and_validate "${S1_TAG}"

    train_code_stage \
        "${S2_TAG}" \
        "${OUTER_REPO}/outputs/${S1_TAG}" \
        "${CODE_S2_DATA}" \
        32 \
        "$((PORT_BASE + 1))" \
        "${SEED}" \
        "${ALPHA}" \
        "${CTM_MODE}" \
        "${CTM_LAMBDA}" \
        "${ANCHOR_CONTROL}" \
        "${RANDOM_ANCHOR_SEED}"

    FINAL_S2_TAGS+=("${S2_TAG}")

    log "[VARIANT TRAINING DONE] ${SHORT}"
}

run_math_t2 () {
    local SHORT="t2_math_matched_jf_seed42"
    local SEED=42
    local PORT_BASE=32120

    local S1_TAG="rebuttal_${SHORT}_gbs${GLOBAL_BATCH}_n16w16_s${STEPS}_lr1e6_${RUN_ID}"
    local S2_TAG="rebuttal_${SHORT}_gbs${GLOBAL_BATCH}_n16w16_to_n64w32_s${STEPS}_lr1e6_${RUN_ID}"

    echo "======================================================================"
    echo "[VARIANT START] ${SHORT}"
    echo "S1=${S1_TAG}"
    echo "S2=${S2_TAG}"
    echo "======================================================================"

    train_math_matched_stage \
        "${S1_TAG}" \
        "${MATH_MODEL}" \
        "${MATH_S1_DATA}" \
        16 \
        "${PORT_BASE}" \
        "${SEED}"

    prepare_and_validate "${S1_TAG}"

    train_math_matched_stage \
        "${S2_TAG}" \
        "${OUTER_REPO}/outputs/${S1_TAG}" \
        "${MATH_S2_DATA}" \
        64 \
        "$((PORT_BASE + 1))" \
        "${SEED}"

    FINAL_S2_TAGS+=("${S2_TAG}")

    log "[VARIANT TRAINING DONE] ${SHORT}"
}

###############################################################################
# Preflight
###############################################################################

preflight () {
    echo "======================================================================"
    echo "[PREFLIGHT]"
    echo "INNER_REPO=${INNER_REPO}"
    echo "OUTER_REPO=${OUTER_REPO}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    echo "NPROC=${NPROC}"
    echo "PER_DEVICE_BS=${PER_DEVICE_BS}"
    echo "GRAD_ACCUM=${GRAD_ACCUM}"
    echo "GLOBAL_BATCH=${GLOBAL_BATCH}"
    echo "STEPS=${STEPS}"
    echo "LR=${LR}"
    echo "RUN_ID=${RUN_ID}"
    echo "PREPARE_FINALS_AFTER_TRAIN=${PREPARE_FINALS_AFTER_TRAIN}"
    echo "LOG_ROOT=${LOG_ROOT}"
    echo "======================================================================"

    [[ "${NPROC}" -eq 4 ]] || die "NPROC must remain 4 for matched GBS=4 training."
    [[ "${PER_DEVICE_BS}" -eq 1 ]] || die "PER_DEVICE_BS must remain 1."
    [[ "${GRAD_ACCUM}" -eq 1 ]] || die "GRAD_ACCUM must remain 1."

    check_file "${TRAIN_PY}"
    check_file "${TRAINER_PY}"
    check_file "${PREP_SCRIPT}"
    check_file "${CODE_DS}"
    check_file "${MATH_DS}"

    check_dir "${CODE_MODEL}"
    check_dir "${MATH_MODEL}"

    check_file "${CODE_S1_DATA}"
    check_file "${CODE_S2_DATA}"
    check_dir "${MATH_S1_DATA}"
    check_dir "${MATH_S2_DATA}"

    if grep -q 'OUTER_REPO="/home/irteam/data-vol1/minsu/JacobiForcing"' "${PREP_SCRIPT}"; then
        die "PREP_SCRIPT still contains the OLD JacobiForcing root."
    fi

    python -m py_compile \
        "${TRAIN_PY}" \
        "${TRAINER_PY}"

    local HELP_FILE
    HELP_FILE="$(mktemp)"
    python "${TRAIN_PY}" --help > "${HELP_FILE}" 2>&1

    for ARG in \
        anchor_control_mode \
        random_anchor_seed \
        anchor_cons_alpha \
        anchor_cons_noise_gate \
        ctm_aux_mode \
        lambda_ctm_aux \
        ctm_top_ratio \
        ctm_residual_weight \
        ctm_score_power \
        ctm_max_weight \
        seed \
        data_seed
    do
        if ! grep -q -- "${ARG}" "${HELP_FILE}"; then
            rm -f "${HELP_FILE}"
            die "Required training argument missing: ${ARG}"
        fi
    done

    rm -f "${HELP_FILE}"

    grep -q 'def _random_same_count_anchor_mask' "${TRAINER_PY}" \
        || die "random_same_count helper missing"

    grep -q '"hybrid_anchor_persistence"' "${TRAINER_PY}" \
        || die "hybrid_anchor_persistence mode missing"

    grep -q '"persistence"' "${TRAINER_PY}" \
        || die "persistence CTM mode missing"

    grep -q '"anchor"' "${TRAINER_PY}" \
        || die "anchor CTM mode missing"

    echo
    echo "[DATA]"
    wc -l "${CODE_S1_DATA}" "${CODE_S2_DATA}" || true

    echo
    echo "[DISK]"
    df -h "${OUTER_REPO}"

    echo
    echo "[GPU]"
    nvidia-smi -i 1,2,3,4 \
        --query-gpu=index,name,memory.used,memory.total \
        --format=csv,noheader || true

    echo
    echo "[PREFLIGHT OK]"
}

###############################################################################
# Manifest
###############################################################################

write_manifest () {
    cat > "${MANIFEST}" <<EOF
order	domain	variant	seed	data_seed	alpha	ctm_mode	lambda_ctm	top_ratio	anchor_control	random_anchor_seed	stage1	stage2
T1	code	matched_jf	42	42	0.0	none	0.0	0.20	semantic	42	n16w16	n32w16
T2	math	matched_jf	42	42	0.0	none	0.0	NA	semantic	42	n16w16	n64w32
T5	code	alpha0_hybrid_ctm	42	42	0.0	hybrid_anchor_persistence	0.002	0.20	semantic	42	n16w16	n32w16
T6	code	alpha0_residual_ctm	42	42	0.0	persistence	0.002	0.20	semantic	42	n16w16	n32w16
T7	code	alpha0_anchor_ctm	42	42	0.0	anchor	0.002	0.20	semantic	42	n16w16	n32w16
T8	code	random_anchor_full	42	42	1.0	hybrid_anchor_persistence	0.002	0.20	random_same_count	42	n16w16	n32w16
T3	code	full_cajf	43	43	1.0	hybrid_anchor_persistence	0.002	0.20	semantic	42	n16w16	n32w16
T4	code	full_cajf	44	44	1.0	hybrid_anchor_persistence	0.002	0.20	semantic	42	n16w16	n32w16
EOF

    log "[MANIFEST] ${MANIFEST}"
    column -t -s $'\t' "${MANIFEST}" || cat "${MANIFEST}"
}

###############################################################################
# Main queue -- T9 Anchor-Aware Primary only
###############################################################################

main () {
    preflight

    echo
    echo "######################################################################"
    echo "# AUTO-INFLUENCE PRIMARY-ONLY TRAINING START"
    echo "# $(date)"
    echo "#"
    echo "# seed=42"
    echo "# anchor_cons_alpha=1.0"
    echo "# anchor_cons_type=auto_influence"
    echo "# automatic_vocab=global-median training-data discovery"
    echo "# ctm_aux_mode=none"
    echo "# lambda_ctm_aux=0.0"
    echo "# n16w16 10k -> n32w16 10k"
    echo "######################################################################"

    run_code_variant \
        "auto_globalmedian_primary_only_seed42" \
        42 \
        1.0 \
        none \
        0.0 \
        semantic \
        42 \
        32210

    if [[ "${PREPARE_FINALS_AFTER_TRAIN}" == "1" ]]; then
        echo
        echo "######################################################################"
        echo "# PREPARE FINAL S2 MODEL"
        echo "######################################################################"

        for TAG in "${FINAL_S2_TAGS[@]}"; do
            prepare_and_validate "${TAG}"
        done
    fi

    echo
    echo "######################################################################"
    echo "# T9 TRAINING FINISHED"
    echo "# $(date)"
    echo "######################################################################"
}

main "$@"
