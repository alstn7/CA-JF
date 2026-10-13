#!/usr/bin/env bash
set -Eeuo pipefail

###############################################################################
# Controlled JF vs CA-JF inference diagnostic
#
# Same physical GPU, same session, strictly serial:
#   JF -> CA-JF -> CA-JF -> JF
#
# Benchmarks:
#   HumanEval
#   MBPP-0shot + robust eval
#   GSM8K
#   MATH500
###############################################################################

INNER="/home/irteam/data-vol1/minsu/CA-JacobiForcing/JacobiForcing"
OUTER="/home/irteam/data-vol1/minsu/CA-JacobiForcing"

GPU="${GPU:-1}"
COOLDOWN="${COOLDOWN:-10}"

JF_CODE="${OUTER}/outputs/rebuttal_t1_code_matched_jf_seed42_gbs4_n16w16_to_n32w16_s10000_lr1e6_20260916_rebuttal_v1"

CA_CODE="${OUTER}/outputs/jf_trueacwctm_ctm_hybrid_anchor_persistence_aux002_top20_gbs4_n16w16_to_n32w16_s10000_lr1e6"

JF_MATH="${OUTER}/outputs/rebuttal_t2_math_matched_jf_seed42_gbs4_n16w16_to_n64w32_s10000_lr1e6_20260916_rebuttal_v1"

CA_MATH="${OUTER}/outputs/jf_math_round0_math_ctm_hybrid_anchor_persistence_aux002_top20_gbs4_n16w16_to_n64w32_full_s10000_lr1e6"

STAMP="$(date +%Y%m%d_%H%M%S)"
ROOT="${OUTER}/outputs/eval_0919_controlled_throughput_abba_${STAMP}"

mkdir -p "${ROOT}"
cd "${INNER}"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

log () {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

cooldown () {
    log "[COOLDOWN] ${COOLDOWN}s"
    sleep "${COOLDOWN}"
}

gpu_snapshot () {
    echo
    echo "===== GPU SNAPSHOT ====="
    nvidia-smi -i "${GPU}" \
      --query-gpu=index,name,temperature.gpu,power.draw,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader || true
    echo "========================"
    echo
}

###############################################################################
# Preflight
###############################################################################

for d in \
    "${JF_CODE}" \
    "${CA_CODE}" \
    "${JF_MATH}" \
    "${CA_MATH}"
do
    [[ -d "${d}" ]] || {
        echo "[ERROR] Missing model: ${d}"
        exit 1
    }
done

for f in \
    jacobi_forcing_inference_codegen_evalchemy.py \
    jacobi_forcing_inference_codegen_evalchemy_mbpp0.py \
    repair_mbpp0_jsonl_robust_and_eval.py \
    jacobi_forcing_inference_GSM8K_minsu_evalchemy.py \
    jacobi_forcing_inference_MATH500_minsu_evalchemy.py
do
    [[ -f "${f}" ]] || {
        echo "[ERROR] Missing file: ${f}"
        exit 1
    }
done

echo "======================================================================"
echo "CONTROLLED THROUGHPUT / CONVERGENCE DIAGNOSTIC"
echo "GPU=${GPU}"
echo "ROOT=${ROOT}"
echo
echo "JF_CODE=${JF_CODE}"
echo "CA_CODE=${CA_CODE}"
echo "JF_MATH=${JF_MATH}"
echo "CA_MATH=${CA_MATH}"
echo "======================================================================"

gpu_snapshot

###############################################################################
# HumanEval
###############################################################################

run_he () {
    local tag="$1"
    local model="$2"

    local od="${ROOT}/${tag}"
    mkdir -p "${od}"

    log "[HE START] ${tag}"
    gpu_snapshot

    CUDA_VISIBLE_DEVICES="${GPU}" \
    python jacobi_forcing_inference_codegen_evalchemy.py \
      --task humaneval \
      --model_path "${model}" \
      --tokenizer_path "${model}" \
      --output "${od}/humaneval.jsonl" \
      --profile_csv "${od}/humaneval.profile.csv" \
      --limit -1 \
      --block_len 64 \
      --max_new_tokens 1024 \
      --max_calls 1024 \
      --attn_implementation flash_attention_2 \
      > "${od}/humaneval.log" 2>&1

    [[ -f "${od}/humaneval.summary.json" ]] || {
        echo "[ERROR] HE summary missing for ${tag}"
        tail -100 "${od}/humaneval.log" || true
        exit 1
    }

    log "[HE DONE] ${tag}"
    cat "${od}/humaneval.summary.json"
}

###############################################################################
# MBPP
###############################################################################

run_mbpp () {
    local tag="$1"
    local model="$2"

    local od="${ROOT}/${tag}"
    mkdir -p "${od}"

    log "[MBPP START] ${tag}"
    gpu_snapshot

    CUDA_VISIBLE_DEVICES="${GPU}" \
    python jacobi_forcing_inference_codegen_evalchemy_mbpp0.py \
      --task mbpp \
      --model_path "${model}" \
      --tokenizer_path "${model}" \
      --output "${od}/mbpp0.jsonl" \
      --profile_csv "${od}/mbpp0.profile.csv" \
      --mbpp_num_examples 0 \
      --mbpp_start_idx 10 \
      --mbpp_end_idx 510 \
      --limit -1 \
      --block_len 32 \
      --max_new_tokens 512 \
      --max_calls 512 \
      --attn_implementation flash_attention_2 \
      > "${od}/mbpp0.log" 2>&1

    [[ -f "${od}/mbpp0.summary.json" ]] || {
        echo "[ERROR] MBPP generation summary missing for ${tag}"
        tail -100 "${od}/mbpp0.log" || true
        exit 1
    }

    python repair_mbpp0_jsonl_robust_and_eval.py \
      --input "${od}/mbpp0.jsonl" \
      --output "${od}/mbpp0.robust.jsonl" \
      --summary "${od}/mbpp0.robust.summary.json" \
      > "${od}/mbpp0.robust.log" 2>&1

    [[ -f "${od}/mbpp0.robust.summary.json" ]] || {
        echo "[ERROR] MBPP robust summary missing for ${tag}"
        tail -100 "${od}/mbpp0.robust.log" || true
        exit 1
    }

    log "[MBPP DONE] ${tag}"
    cat "${od}/mbpp0.summary.json"
    cat "${od}/mbpp0.robust.summary.json"
}

###############################################################################
# GSM8K
###############################################################################

run_gsm () {
    local tag="$1"
    local model="$2"

    local od="${ROOT}/${tag}"
    mkdir -p "${od}"

    log "[GSM START] ${tag}"
    gpu_snapshot

    CUDA_VISIBLE_DEVICES="${GPU}" \
    JF_LOAD_ATTN_IMPL=eager \
    JF_ATTN_BACKEND=dense \
    python jacobi_forcing_inference_GSM8K_minsu_evalchemy.py \
      --model_path "${model}" \
      --tokenizer_path "${model}" \
      --output "${od}/gsm8k.jsonl" \
      --profile_csv "${od}/gsm8k.profile.csv" \
      --limit -1 \
      --prompt_mode boxed \
      --scoring_mode robust_math \
      --block_len 64 \
      --max_new_tokens 1024 \
      --max_calls 1024 \
      --attn_implementation flash_attention_2 \
      > "${od}/gsm8k.log" 2>&1

    [[ -f "${od}/gsm8k.summary.json" ]] || {
        echo "[ERROR] GSM summary missing for ${tag}"
        tail -100 "${od}/gsm8k.log" || true
        exit 1
    }

    log "[GSM DONE] ${tag}"
    cat "${od}/gsm8k.summary.json"
}

###############################################################################
# MATH500
###############################################################################

run_math500 () {
    local tag="$1"
    local model="$2"

    local od="${ROOT}/${tag}"
    mkdir -p "${od}"

    log "[MATH500 START] ${tag}"
    gpu_snapshot

    CUDA_VISIBLE_DEVICES="${GPU}" \
    JF_LOAD_ATTN_IMPL=eager \
    JF_ATTN_BACKEND=dense \
    python jacobi_forcing_inference_MATH500_minsu_evalchemy.py \
      --model_path "${model}" \
      --tokenizer_path "${model}" \
      --output "${od}/math500.jsonl" \
      --profile_csv "${od}/math500.profile.csv" \
      --limit -1 \
      --block_len 64 \
      --max_new_tokens 2048 \
      --max_calls 2048 \
      --attn_implementation flash_attention_2 \
      > "${od}/math500.log" 2>&1

    [[ -f "${od}/math500.summary.json" ]] || {
        echo "[ERROR] MATH500 summary missing for ${tag}"
        tail -100 "${od}/math500.log" || true
        exit 1
    }

    log "[MATH500 DONE] ${tag}"
    cat "${od}/math500.summary.json"
}

###############################################################################
# A/B/B/A runs
###############################################################################

log "================ HumanEval A/B/B/A ================"

run_he "he_jf_r1" "${JF_CODE}"
cooldown
run_he "he_ca_r1" "${CA_CODE}"
cooldown
run_he "he_ca_r2" "${CA_CODE}"
cooldown
run_he "he_jf_r2" "${JF_CODE}"
cooldown

log "================ MBPP A/B/B/A ====================="

run_mbpp "mbpp_jf_r1" "${JF_CODE}"
cooldown
run_mbpp "mbpp_ca_r1" "${CA_CODE}"
cooldown
run_mbpp "mbpp_ca_r2" "${CA_CODE}"
cooldown
run_mbpp "mbpp_jf_r2" "${JF_CODE}"
cooldown

log "================ GSM8K A/B/B/A ===================="

run_gsm "gsm_jf_r1" "${JF_MATH}"
cooldown
run_gsm "gsm_ca_r1" "${CA_MATH}"
cooldown
run_gsm "gsm_ca_r2" "${CA_MATH}"
cooldown
run_gsm "gsm_jf_r2" "${JF_MATH}"
cooldown

log "================ MATH500 A/B/B/A =================="

run_math500 "math_jf_r1" "${JF_MATH}"
cooldown
run_math500 "math_ca_r1" "${CA_MATH}"
cooldown
run_math500 "math_ca_r2" "${CA_MATH}"
cooldown
run_math500 "math_jf_r2" "${JF_MATH}"

###############################################################################
# Aggregate
###############################################################################

python - "${ROOT}" <<'PY'
import csv
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])

spec = {
    "HumanEval": {
        "JF": ["he_jf_r1", "he_jf_r2"],
        "CA-JF": ["he_ca_r1", "he_ca_r2"],
        "summary": "humaneval.summary.json",
        "quality_summary": "humaneval.summary.json",
    },
    "MBPP": {
        "JF": ["mbpp_jf_r1", "mbpp_jf_r2"],
        "CA-JF": ["mbpp_ca_r1", "mbpp_ca_r2"],
        "summary": "mbpp0.summary.json",
        "quality_summary": "mbpp0.robust.summary.json",
    },
    "GSM8K": {
        "JF": ["gsm_jf_r1", "gsm_jf_r2"],
        "CA-JF": ["gsm_ca_r1", "gsm_ca_r2"],
        "summary": "gsm8k.summary.json",
        "quality_summary": "gsm8k.summary.json",
    },
    "MATH500": {
        "JF": ["math_jf_r1", "math_jf_r2"],
        "CA-JF": ["math_ca_r1", "math_ca_r2"],
        "summary": "math500.summary.json",
        "quality_summary": "math500.summary.json",
    },
}

def load(path):
    return json.loads(path.read_text())

def nested_find(x, keys):
    if isinstance(x, dict):
        for k in keys:
            if k in x and isinstance(x[k], (int, float)):
                return float(x[k])
        for v in x.values():
            y = nested_find(v, keys)
            if y is not None:
                return y
    elif isinstance(x, list):
        for v in x:
            y = nested_find(v, keys)
            if y is not None:
                return y
    return None

def get_metrics(bench, tag, cfg):
    s = load(root / tag / cfg["summary"])
    q = load(root / tag / cfg["quality_summary"])

    new_tokens = s.get("avg_new_tokens_eos_or_all")
    calls = s.get("avg_calls_eos_or_all")

    calls_per_token = None
    if new_tokens not in (None, 0) and calls is not None:
        calls_per_token = calls / new_tokens

    quality = nested_find(
        q,
        [
            "pass@1",
            "solve_rate",
            "accuracy",
            "acc",
            "exact_match",
        ],
    )

    return {
        "quality": quality,
        "n_eos": s.get("n_eos"),
        "new_tokens": new_tokens,
        "calls": calls,
        "calls_per_token": calls_per_token,
        "iter_per_call": s.get("avg_iter_per_call_eos_or_all"),
        "iter_per_token": s.get("avg_iter_per_token_eos_or_all"),
        "tps": s.get("avg_toks_per_sec_eos_or_all"),
    }

def med(vals):
    vals = [x for x in vals if x is not None]
    return statistics.median(vals) if vals else None

def mean(vals):
    vals = [x for x in vals if x is not None]
    return statistics.mean(vals) if vals else None

def fmt(x, digits=4):
    if x is None:
        return "NA"
    return f"{x:.{digits}f}"

rows = []
all_runs = {}

for bench, cfg in spec.items():
    all_runs[bench] = {}

    for method, tags in (("JF", cfg["JF"]), ("CA-JF", cfg["CA-JF"])):
        runs = [get_metrics(bench, tag, cfg) for tag in tags]
        all_runs[bench][method] = runs

        row = {
            "benchmark": bench,
            "method": method,
            "quality_mean": mean([r["quality"] for r in runs]),
            "tps_r1": runs[0]["tps"],
            "tps_r2": runs[1]["tps"],
            "tps_median": med([r["tps"] for r in runs]),
            "new_tokens_median": med([r["new_tokens"] for r in runs]),
            "calls_median": med([r["calls"] for r in runs]),
            "calls_per_token_median": med([r["calls_per_token"] for r in runs]),
            "iter_per_call_median": med([r["iter_per_call"] for r in runs]),
            "iter_per_token_median": med([r["iter_per_token"] for r in runs]),
            "n_eos_median": med([r["n_eos"] for r in runs]),
        }
        rows.append(row)

csv_path = root / "controlled_diagnostic.csv"

with csv_path.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

lines = []
lines.append("=" * 110)
lines.append("CONTROLLED JF vs CA-JF A/B/B/A DIAGNOSTIC")
lines.append("=" * 110)
lines.append("")

for bench in spec:
    jf = next(r for r in rows if r["benchmark"] == bench and r["method"] == "JF")
    ca = next(r for r in rows if r["benchmark"] == bench and r["method"] == "CA-JF")

    def delta(key):
        a = jf[key]
        b = ca[key]
        if a is None or b is None:
            return None
        return b - a

    def pct_delta(key):
        a = jf[key]
        b = ca[key]
        if a in (None, 0) or b is None:
            return None
        return 100 * (b / a - 1)

    lines.append(f"[{bench}]")
    lines.append(
        f"{'':18s}"
        f"{'JF':>14s}"
        f"{'CA-JF':>14s}"
        f"{'CA-JF - JF':>16s}"
        f"{'% change':>14s}"
    )
    lines.append("-" * 76)

    metrics = [
        ("TPS", "tps_median"),
        ("new tokens", "new_tokens_median"),
        ("calls", "calls_median"),
        ("calls/token", "calls_per_token_median"),
        ("iter/call", "iter_per_call_median"),
        ("iter/token", "iter_per_token_median"),
        ("n_eos", "n_eos_median"),
    ]

    for label, key in metrics:
        lines.append(
            f"{label:18s}"
            f"{fmt(jf[key]):>14s}"
            f"{fmt(ca[key]):>14s}"
            f"{fmt(delta(key)):>16s}"
            f"{fmt(pct_delta(key), 2):>13s}%"
        )

    lines.append(
        f"{'quality':18s}"
        f"{fmt(jf['quality_mean']):>14s}"
        f"{fmt(ca['quality_mean']):>14s}"
        f"{fmt((ca['quality_mean'] - jf['quality_mean']) if "
        f"ca['quality_mean'] is not None and jf['quality_mean'] is not None else None):>16s}"
    )

    lines.append("")
    lines.append(
        f"TPS raw JF:    {fmt(jf['tps_r1'])}, {fmt(jf['tps_r2'])}"
    )
    lines.append(
        f"TPS raw CA-JF: {fmt(ca['tps_r1'])}, {fmt(ca['tps_r2'])}"
    )
    lines.append("")

txt_path = root / "controlled_diagnostic.txt"
txt_path.write_text("\n".join(lines) + "\n")

print("\n".join(lines))
print(f"[SAVED] {csv_path}")
print(f"[SAVED] {txt_path}")
PY

echo
echo "======================================================================"
echo "ALL CONTROLLED DIAGNOSTICS COMPLETE"
echo "ROOT=${ROOT}"
echo "======================================================================"
cat "${ROOT}/controlled_diagnostic.txt"
