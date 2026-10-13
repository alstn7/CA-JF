#!/usr/bin/env python3
"""
Evalchemy-compatible MATH500 Jacobi inference/evaluation.

This script keeps the user's Jacobi decoding loop and profiling schema, but uses
Evalchemy's MATH500 prompt and answer scoring:
  - Prompt: user-only chat message with
      Problem: {problem}\nMark your solution with \\boxed\nAnswer:
  - Answer extraction: lm_eval.tasks.hendrycks_math.utils.last_boxed_only_string/remove_boxed
  - Scoring: lm_eval.tasks.hendrycks_math.utils.is_equiv
"""

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, Qwen2ForCausalLM

try:
    from transformers import Qwen3ForCausalLM
except Exception:  # pragma: no cover
    Qwen3ForCausalLM = None

repo_root = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(repo_root))

from modeling.cllm2_qwen2_modeling_kv_terminate_on_eos_improved import jacobi_forward_greedy

Qwen2ForCausalLM.jacobi_forward_greedy = jacobi_forward_greedy
if Qwen3ForCausalLM is not None:
    Qwen3ForCausalLM.jacobi_forward_greedy = jacobi_forward_greedy

try:
    from lm_eval.tasks.hendrycks_math.utils import is_equiv, last_boxed_only_string, remove_boxed
    HENDRYCKS_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    HENDRYCKS_AVAILABLE = False
    HENDRYCKS_IMPORT_ERROR = repr(exc)

    def last_boxed_only_string(text: str):
        if not text:
            return None
        idx = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
        if idx < 0:
            return None
        brace = text.find("{", idx)
        if brace < 0:
            return None
        depth = 0
        for j in range(brace, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    return text[idx : j + 1]
        return None

    def remove_boxed(s: Optional[str]) -> str:
        if s is None:
            return ""
        for prefix in ("\\boxed", "\\fbox"):
            if s.startswith(prefix):
                start = s.find("{")
                if start >= 0 and s.endswith("}"):
                    return s[start + 1 : -1]
        return s

    def is_equiv(a: str, b: str) -> bool:
        def norm(x):
            return str(x).strip().replace(" ", "").replace("$", "")
        return norm(a) == norm(b)


EVALCHEMY_MATH500_PROMPT = """Problem: {problem}\nMark your solution with \\boxed\nAnswer:"""
DEFAULT_EVALCHEMY_MATH500 = "/home/irteam/data-vol1/minsu/evalchemy/eval/chat_benchmarks/MATH500/data/math500.jsonl"


def load_math500_records(dataset_path: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    path_to_try = dataset_path
    if not path_to_try and Path(DEFAULT_EVALCHEMY_MATH500).exists():
        path_to_try = DEFAULT_EVALCHEMY_MATH500

    if path_to_try:
        p = Path(path_to_try)
        if not p.exists():
            raise FileNotFoundError(f"dataset_path not found: {path_to_try}")
        if p.suffix == ".jsonl":
            df = pd.read_json(str(p), lines=True)
            records = df.to_dict(orient="records")
        elif p.suffix == ".json":
            with open(p, "r") as f:
                obj = json.load(f)
            records = obj if isinstance(obj, list) else obj.get("data", [])
        else:
            raise ValueError(f"Unsupported dataset_path suffix: {p.suffix}")
    else:
        records = list(load_dataset("HuggingFaceH4/MATH-500", split="test"))

    if limit is not None and limit > 0:
        records = records[:limit]
    return records


def get_problem(row: Dict[str, Any]) -> str:
    for key in ["problem", "question", "input"]:
        if key in row and row[key] is not None:
            return str(row[key])
    raise KeyError(f"Could not find problem/question field in row keys={list(row.keys())}")


def get_gold(row: Dict[str, Any]) -> str:
    for key in ["answer", "expected_answer", "gold", "target", "solution"]:
        if key in row and row[key] is not None:
            return str(row[key])
    return ""


def build_evalchemy_math500_prompt(tokenizer, problem: str) -> str:
    messages = [{"role": "user", "content": EVALCHEMY_MATH500_PROMPT.format(problem=problem.strip())}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_math500_answer_evalchemy(output: str) -> str:
    try:
        return remove_boxed(last_boxed_only_string(output))
    except Exception:
        return ""


def load_model(model_path: str, tokenizer_path: Optional[str], attn_implementation: str):
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_type = getattr(cfg, "model_type", "")
    if model_type == "qwen3" and Qwen3ForCausalLM is not None:
        cls = Qwen3ForCausalLM
    else:
        cls = Qwen2ForCausalLM

    model = cls.from_pretrained(
        model_path,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path or model_path,
        trust_remote_code=True,
        use_fast=False,
    )
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_one(
    model,
    tokenizer,
    prompt_text: str,
    n_token_seq_len: int,
    max_new_tokens: int,
    max_calls: int,
    seed: Optional[int],
    prefill_full_attention_mask: bool = False,
    debug_accept: bool = False,
    debug_max_calls: int = 10,
):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    model_inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)
    input_ids = model_inputs["input_ids"]
    attention_mask = torch.ones_like(input_ids, device=model.device)

    eos_id = tokenizer.eos_token_id
    alt_eos_id = 151645

    iters = []
    total_new_tokens = 0
    calls = 0
    prev_len = input_ids.shape[1]
    prompt_len = prev_len
    stop_reason = None
    prefill_phase = True
    generated_ids = input_ids
    prefill_drafted_n_gram = None
    first_correct_token = None
    past_key_values = None

    gen_only_time = 0.0
    t_start = time.perf_counter()

    while True:
        generated_part = generated_ids[0, prompt_len:]
        hit_eos = False
        if generated_part.numel() > 0:
            if eos_id is not None:
                hit_eos = (generated_part == eos_id).any().item()
            if not hit_eos:
                hit_eos = (generated_part == alt_eos_id).any().item()

        if hit_eos:
            stop_reason = "eos"
            break
        if total_new_tokens >= max_new_tokens:
            stop_reason = "max_new_tokens"
            break
        if calls >= max_calls:
            stop_reason = "max_calls"
            break

        if prefill_phase:
            q_sampled = []
            src = generated_ids[0].tolist()
            for _ in range(n_token_seq_len):
                q_sample = torch.tensor([random.choice(src)], dtype=torch.long, device=model.device).unsqueeze(0)
                q_sampled.append(q_sample)
            prefill_draft_token_ids = torch.cat(q_sampled, dim=1)
            prefill_input_ids = torch.cat((input_ids, prefill_draft_token_ids), dim=-1)
            prefill_attention_mask = torch.ones_like(prefill_input_ids, device=model.device) if prefill_full_attention_mask else attention_mask

            past_key_values, first_correct_token, prefill_drafted_n_gram, iter_count = model.jacobi_forward_greedy(
                input_ids=prefill_input_ids,
                attention_mask=prefill_attention_mask,
                past_key_values=None,
                use_cache=True,
                prefill_phase=True,
                n_token_seq_len=n_token_seq_len,
                tokenizer=tokenizer,
                eos_token_id=eos_id,
            )
            prefill_phase = False
            generated_ids = input_ids
            itr_count = 0
        else:
            if calls == 1:
                input_ids = prefill_drafted_n_gram
            else:
                q_sampled = []
                src = generated_ids[0].tolist()
                for _ in range(n_token_seq_len - 1):
                    q_sample = torch.tensor([random.choice(src)], dtype=torch.long, device=model.device).unsqueeze(0)
                    q_sampled.append(q_sample)
                q_sampled = torch.cat(q_sampled, dim=1)
                input_ids = torch.cat((first_correct_token.view(1, -1), q_sampled), dim=-1)

            t_gen = time.perf_counter()
            past_key_values, first_correct_token, accepted_n_gram, itr_count = model.jacobi_forward_greedy(
                input_ids=input_ids,
                attention_mask=None,
                past_key_values=past_key_values,
                use_cache=True,
                prefill_phase=False,
                n_token_seq_len=n_token_seq_len,
                tokenizer=tokenizer,
                eos_token_id=eos_id,
            )
            gen_only_time += time.perf_counter() - t_gen

            if debug_accept and calls <= debug_max_calls:
                try:
                    acc_text = tokenizer.decode(accepted_n_gram[0], skip_special_tokens=False)
                    first_tok_text = tokenizer.decode(first_correct_token.view(-1), skip_special_tokens=False)
                    print("=" * 80, flush=True)
                    print(f"[DEBUG_ACCEPT] call={calls} itr={itr_count} accepted_shape={tuple(accepted_n_gram.shape)}", flush=True)
                    print(f"[DEBUG_ACCEPT] first_correct_token={repr(first_tok_text)}", flush=True)
                    print(f"[DEBUG_ACCEPT] accepted_text={repr(acc_text[:1000])}", flush=True)
                    print("=" * 80, flush=True)
                except Exception as e:
                    print(f"[DEBUG_ACCEPT_ERROR] {e}", flush=True)

            generated_ids = torch.cat((generated_ids, accepted_n_gram), dim=-1)

        calls += 1
        iters.append(int(itr_count))

        added = generated_ids.shape[1] - prev_len
        if added > 0:
            total_new_tokens += int(added)
        prev_len = generated_ids.shape[1]

    wall_time = time.perf_counter() - t_start
    total_new_tokens = max(0, generated_ids.shape[1] - prompt_len)

    raw_completion_ids = generated_ids[0, prompt_len:]
    completion_ids = raw_completion_ids
    eos_cut_pos = None
    cut_positions = []
    if completion_ids.numel() > 0:
        if eos_id is not None:
            pos = (completion_ids == eos_id).nonzero(as_tuple=False)
            if pos.numel() > 0:
                cut_positions.append(int(pos[0].item()))
        pos = (completion_ids == alt_eos_id).nonzero(as_tuple=False)
        if pos.numel() > 0:
            cut_positions.append(int(pos[0].item()))
    if cut_positions:
        eos_cut_pos = min(cut_positions)
        completion_ids = completion_ids[:eos_cut_pos]

    completion = tokenizer.decode(completion_ids, skip_special_tokens=False)

    total_iterations = sum(iters)
    avg_iter_per_call = total_iterations / calls if calls else float("nan")
    avg_iter_per_token = total_iterations / total_new_tokens if total_new_tokens else float("nan")
    toks_per_sec = total_new_tokens / gen_only_time if gen_only_time > 0 else float("nan")

    profile = {
        "prompt_tokens": int(prompt_len),
        "new_tokens": int(total_new_tokens),
        "decoded_new_tokens": int(completion_ids.numel()),
        "raw_new_tokens": int(raw_completion_ids.numel()),
        "eos_cut_pos": eos_cut_pos,
        "calls": int(calls),
        "total_iterations": int(total_iterations),
        "avg_iter_per_call": avg_iter_per_call,
        "avg_iter_per_token": avg_iter_per_token,
        "wall_time_sec": wall_time,
        "gen_only_time_sec": gen_only_time,
        "toks_per_sec": toks_per_sec,
        "stop_reason": stop_reason,
    }
    return completion, profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--dataset_path", default=None, help="Optional MATH500 jsonl/json. Defaults to Evalchemy local math500.jsonl if present, else HF MATH-500.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile_csv", default=None)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--block_len", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--max_calls", type=int, default=2048)
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefill_full_attention_mask", action="store_true")
    parser.add_argument("--debug_accept", action="store_true")
    parser.add_argument("--debug_max_calls", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(str(Path(args.output).parent), exist_ok=True)

    limit = args.limit if args.limit and args.limit > 0 else None
    records = load_math500_records(args.dataset_path, limit=limit)
    print(f"Loaded MATH500 records: {len(records)}")
    print(f"Hendrycks math scorer available: {HENDRYCKS_AVAILABLE}")
    if not HENDRYCKS_AVAILABLE:
        print(f"[WARN] Falling back to simple equality because lm_eval import failed: {HENDRYCKS_IMPORT_ERROR}")

    model, tokenizer = load_model(args.model_path, args.tokenizer_path or args.model_path, args.attn_implementation)

    rows = []
    correct = 0
    total = 0
    t0 = time.perf_counter()

    with open(args.output, "w") as fout:
        for idx, row in tqdm(list(enumerate(records))):
            task_id = row.get("task_id", row.get("id", f"MATH500/{idx}"))
            problem = get_problem(row)
            gold_raw = get_gold(row)
            prompt = build_evalchemy_math500_prompt(tokenizer, problem)

            completion, profile = generate_one(
                model=model,
                tokenizer=tokenizer,
                prompt_text=prompt,
                n_token_seq_len=args.block_len,
                max_new_tokens=args.max_new_tokens,
                max_calls=args.max_calls,
                seed=args.seed + idx if args.seed is not None else None,
                prefill_full_attention_mask=args.prefill_full_attention_mask,
                debug_accept=args.debug_accept and idx == 0,
                debug_max_calls=args.debug_max_calls,
            )

            pred = extract_math500_answer_evalchemy(completion)
            ok = bool(is_equiv(str(gold_raw), pred))
            total += 1
            correct += int(ok)

            out = {
                "index": idx,
                "task_id": task_id,
                "problem": problem,
                "gold_raw": gold_raw,
                "gold": str(gold_raw),
                "prediction": pred,
                "correct": ok,
                "prompt": prompt,
                "completion": completion,
                **profile,
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()

            rows.append({"index": idx, "task_id": task_id, "correct": ok, **profile})

            if (idx + 1) % 10 == 0 or (idx + 1) == len(records):
                print(
                    f"[{idx+1}/{len(records)}] acc={correct}/{total}={correct/total:.4f} "
                    f"last_stop={profile['stop_reason']} toks/sec={profile['toks_per_sec']:.2f} pred={pred!r} gold={gold_raw!r}",
                    flush=True,
                )

    elapsed = time.perf_counter() - t0
    df = pd.DataFrame(rows)
    if args.profile_csv is None:
        args.profile_csv = str(Path(args.output).with_suffix(".profile.csv"))
    df.to_csv(args.profile_csv, index=False)
    eos_df = df[df["stop_reason"] == "eos"].copy()

    def mean_col(name):
        if name not in eos_df or len(eos_df) == 0:
            return float("nan")
        return float(pd.to_numeric(eos_df[name], errors="coerce").mean())

    summary = {
        "total": total,
        "correct": correct,
        "solve_rate": correct / total if total else 0.0,
        "n_eos": int((df["stop_reason"] == "eos").sum()) if len(df) else 0,
        "elapsed_sec": elapsed,
        "avg_new_tokens_eos": mean_col("new_tokens"),
        "avg_calls_eos": mean_col("calls"),
        "avg_iter_per_call_eos": mean_col("avg_iter_per_call"),
        "avg_iter_per_token_eos": mean_col("avg_iter_per_token"),
        "avg_toks_per_sec_eos": mean_col("toks_per_sec"),
        "output": args.output,
        "profile_csv": args.profile_csv,
        "prompt_style": "evalchemy_math500",
        "scoring": "hendrycks_math.is_equiv" if HENDRYCKS_AVAILABLE else "fallback_simple_equality",
        "block_len": args.block_len,
        "max_new_tokens": args.max_new_tokens,
        "max_calls": args.max_calls,
        "prefill_full_attention_mask": args.prefill_full_attention_mask,
        "attn_implementation": args.attn_implementation,
    }

    print("\n=== MATH500 Jacobi Evalchemy-Compatible Eval Summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    summary_path = str(Path(args.output).with_suffix(".summary.json"))
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[summary saved] {summary_path}")


if __name__ == "__main__":
    main()
