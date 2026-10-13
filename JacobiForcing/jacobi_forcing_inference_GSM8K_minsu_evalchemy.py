#!/usr/bin/env python3
"""
ZeroEval/Evalchemy-compatible GSM8K Jacobi inference/evaluation.

This script keeps the user's Jacobi decoding loop and profiling schema, but uses
zeroeval's GSM8K/OEQA prompt and math_eval-style scoring:
  - Prompt: OEQA JSON prompt as user-only chat message
  - Extraction: first complete JSON -> regex JSON-value extraction with allow_no_quotes -> model-specific fallback
  - Scoring: sanitize answers and compare first numeric value as float
"""

import argparse
import json
import os
import random
import re
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

OEQA_PROMPT = """
## Question: 

{question}


## Instruction 

Please answer this question by first reasoning and then providing your answer.
Present your reasoning and solution in the following json format. 
Please show your final answer in the `answer` field, e.g.,`"answer": "42"`.

```json
{{
    "reasoning": "___",
    "answer": "___"
}}
```
"""


def load_gsm8k_records(dataset_path: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    if dataset_path:
        p = Path(dataset_path)
        if not p.exists():
            raise FileNotFoundError(f"dataset_path not found: {dataset_path}")
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
        records = list(load_dataset("openai/gsm8k", "main", split="test"))
    if limit is not None and limit > 0:
        records = records[:limit]
    return records


def get_problem(row: Dict[str, Any]) -> str:
    for key in ["question", "problem", "input"]:
        if key in row and row[key] is not None:
            return str(row[key])
    raise KeyError(f"Could not find question/problem field in row keys={list(row.keys())}")


def get_gold_raw(row: Dict[str, Any]) -> str:
    for key in ["answer", "solution", "gold", "target"]:
        if key in row and row[key] is not None:
            return str(row[key])
    return ""


def zeroeval_gold_answer(answer_text: str) -> str:
    text = str(answer_text or "")
    if "###" in text:
        text = text.split("###", 1)[1].strip()
    elif "####" in text:
        text = text.split("####", 1)[1].strip()
    return text.replace("#", "").strip()


def sanitize_math_answers(answer: str) -> str:
    answer = str(answer).replace("$", "").strip()
    answer = answer.replace(",", "")
    if "/" in answer:
        try:
            answer = str(float(eval(answer)))
        except Exception:
            pass
    return answer


def model_specific_extraction(model_name: str, prediction_str: str):
    if "Llama-3.1" in model_name:
        if "boxed" in prediction_str[-30:]:
            match = re.search(r"\\boxed{([\w\d]+)}", prediction_str)
            if match:
                return match.group(1)
    return None


def extract_values_from_json(json_string: str, keys=["reasoning", "answer"], allow_no_quotes: bool = False) -> Dict[str, str]:
    extracted_values = {}
    for key in keys:
        pattern = f'"{key}"\\s*:\\s*"([^"]*?)"'
        match = re.search(pattern, json_string)
        if match:
            extracted_values[key] = match.group(1)
        else:
            pattern = f'"{key}"\\s*:\\s*"(.*?)"'
            match = re.search(pattern, json_string, re.DOTALL)
            if match:
                extracted_values[key] = match.group(1)
        if not match and allow_no_quotes:
            pattern = f'"{key}"\\s*:\\s*([^,\\s]*)'
            match = re.search(pattern, json_string)
            if match:
                extracted_values[key] = match.group(1)
            else:
                pattern = f"{key}\\s*:\\s*([^,\\s]*)"
                match = re.search(pattern, json_string)
                if match:
                    extracted_values[key] = match.group(1)
    return extracted_values


def extract_first_complete_json(s: str):
    stack = []
    first_json_start = None
    for i, char in enumerate(s):
        if char == "{":
            stack.append(i)
            if first_json_start is None:
                first_json_start = i
        elif char == "}":
            if stack:
                stack.pop()
                if not stack:
                    first_json_str = s[first_json_start : i + 1]
                    try:
                        return json.loads(first_json_str.replace("\n", ""))
                    except json.JSONDecodeError:
                        return None
                    finally:
                        first_json_start = None
    return None

def extract_boxed(text: str):
    if not text:
        return None
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    brace = text.find("{", idx)
    if brace < 0:
        return None

    depth = 0
    out = []
    for ch in text[brace:]:
        if ch == "{":
            depth += 1
            if depth == 1:
                continue
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out).strip()
        out.append(ch)
    return None

def extract_gsm8k_answer_evalchemy(prediction_str: str, model_name: str = "") -> Dict[str, Any]:
    prediction_json = extract_first_complete_json(prediction_str)
    extraction_method = "first_complete_json"
    if prediction_json is None or "answer" not in prediction_json:
        prediction_json = extract_values_from_json(prediction_str, allow_no_quotes=True)
        extraction_method = "regex_values_allow_no_quotes"
    if prediction_json is None or "answer" not in prediction_json:
        try_extracted_answer = model_specific_extraction(model_name, prediction_str)
        if try_extracted_answer:
            prediction_json = {"answer": try_extracted_answer}
            extraction_method = "model_specific_extraction"
        else:
            return {"answer": None, "reasoning": "", "method": "no_answer"}
    return {
        "answer": prediction_json.get("answer"),
        "reasoning": prediction_json.get("reasoning", ""),
        "method": extraction_method,
    }

def extract_gsm8k_answer_robust_math(prediction_str: str, model_name: str = "") -> Dict[str, Any]:
    """
    GSM8K answer extraction for math correctness evaluation.

    Priority:
      1. zeroeval JSON answer extraction
      2. \\boxed{...}
      3. #### ...
      4. final-answer phrases
      5. last number fallback
    """
    # 1. First, preserve zeroeval extraction behavior.
    z = extract_gsm8k_answer_evalchemy(prediction_str, model_name=model_name)
    if z.get("answer") is not None:
        z["method"] = "zeroeval_" + str(z.get("method"))
        return z

    # 2. Boxed answer.
    boxed = extract_boxed(prediction_str)
    if boxed is not None:
        return {
            "answer": boxed,
            "reasoning": "",
            "method": "boxed_fallback",
        }

    # 3. GSM8K #### marker.
    m = re.search(r"####\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)", prediction_str)
    if m:
        return {
            "answer": m.group(1),
            "reasoning": "",
            "method": "gsm8k_hash_fallback",
        }

    # 4. Final-answer phrases.
    patterns = [
        r"(?:final answer is|answer is|the answer is|therefore,? the answer is)\s*[:：]?\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)",
        r"(?:final answer|answer)\s*[:：]\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)",
    ]
    for pat in patterns:
        m = re.search(pat, prediction_str, flags=re.IGNORECASE)
        if m:
            return {
                "answer": m.group(1),
                "reasoning": "",
                "method": "final_phrase_fallback",
            }

    # 5. Last-number fallback.
    nums = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", prediction_str)
    if nums:
        return {
            "answer": nums[-1],
            "reasoning": "",
            "method": "last_number_fallback",
        }

    return {
        "answer": None,
        "reasoning": "",
        "method": "no_answer",
    }

def zeroeval_gsm8k_match(model_answer: Any, correct_answer: Any) -> Dict[str, Any]:
    if model_answer is None:
        return {
            "correct": False,
            "model_answer_sanitized": None,
            "correct_answer_sanitized": sanitize_math_answers(str(correct_answer)),
            "model_first_number": None,
            "correct_first_number": None,
        }
    raw_model_answer = str(model_answer)
    model_sanitized = sanitize_math_answers(raw_model_answer)
    correct_sanitized = sanitize_math_answers(str(correct_answer))
    first_model = re.search(r"-?\d+(\.\d+)?", model_sanitized)
    first_correct = re.search(r"-?\d+(\.\d+)?", correct_sanitized)
    correct = False
    if first_model and first_correct:
        try:
            correct = float(first_model.group()) == float(first_correct.group())
        except Exception:
            correct = False
    return {
        "correct": correct,
        "model_answer_sanitized": model_sanitized,
        "correct_answer_sanitized": correct_sanitized,
        "model_first_number": first_model.group() if first_model else None,
        "correct_first_number": first_correct.group() if first_correct else None,
    }


def build_prompt(tokenizer, problem: str, prompt_mode: str) -> str:
    if prompt_mode == "zeroeval_oeqa":
        content = OEQA_PROMPT.format(question=problem.strip())
        messages = [{"role": "user", "content": content}]

    elif prompt_mode == "boxed":
        messages = [
            {
                "role": "system",
                "content": "Please reason step by step, and put your final answer within \\boxed{}.",
            },
            {
                "role": "user",
                "content": problem.strip(),
            },
        ]

    elif prompt_mode == "gsm8k_cot":
        messages = [
            {
                "role": "system",
                "content": "Please reason step by step, and put your final answer after ####.",
            },
            {
                "role": "user",
                "content": problem.strip(),
            },
        ]

    else:
        raise ValueError(f"Unknown prompt_mode={prompt_mode}")

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

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
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile_csv", default=None)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--block_len", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--max_calls", type=int, default=1024)
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefill_full_attention_mask", action="store_true")
    parser.add_argument("--debug_accept", action="store_true")
    parser.add_argument("--debug_max_calls", type=int, default=10)
    parser.add_argument(
        "--scoring_mode",
        default="robust_math",
        choices=["strict_zeroeval", "robust_math"],
        help="strict_zeroeval follows zeroeval JSON-only extraction; robust_math adds boxed/hash/final/last-number fallbacks.",
    )
    parser.add_argument(
        "--prompt_mode",
        default="zeroeval_oeqa",
        choices=["zeroeval_oeqa", "boxed", "gsm8k_cot"],
    )
    args = parser.parse_args()

    os.makedirs(str(Path(args.output).parent), exist_ok=True)
    limit = args.limit if args.limit and args.limit > 0 else None
    records = load_gsm8k_records(args.dataset_path, limit=limit)
    print(f"Loaded GSM8K records: {len(records)}")

    model, tokenizer = load_model(args.model_path, args.tokenizer_path or args.model_path, args.attn_implementation)

    rows = []
    correct = 0
    total = 0
    no_answer = 0
    reason_lens = []
    t0 = time.perf_counter()

    with open(args.output, "w") as fout:
        for idx, row in tqdm(list(enumerate(records))):
            task_id = row.get("task_id", row.get("id", f"gsm8k-main-test-#{idx}"))
            problem = get_problem(row)
            gold_raw = get_gold_raw(row)
            gold = zeroeval_gold_answer(gold_raw)
            prompt = build_prompt(tokenizer, problem, args.prompt_mode)

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

            if args.scoring_mode == "strict_zeroeval":
                parsed = extract_gsm8k_answer_evalchemy(completion, args.model_path)
            elif args.scoring_mode == "robust_math":
                parsed = extract_gsm8k_answer_robust_math(completion, args.model_path)
            else:
                raise ValueError(f"Unknown scoring_mode={args.scoring_mode}")

            if parsed["answer"] is None:
                no_answer += 1

            reason_lens.append(len(str(parsed.get("reasoning", ""))))
            match = zeroeval_gsm8k_match(parsed["answer"], gold)
            ok = match["correct"]
            
            total += 1
            correct += int(ok)

            out = {
                "index": idx,
                "task_id": task_id,
                "problem": problem,
                "gold_raw": gold_raw,
                "gold": gold,
                "prompt_mode": args.prompt_mode,
                "scoring_mode": args.scoring_mode,
                "extraction_method": parsed.get("method"),
                "prediction": parsed["answer"],
                "prediction_extraction_method": parsed["method"],
                "reasoning": parsed.get("reasoning", ""),
                "correct": ok,
                "model_answer": {
                    "raw": parsed["answer"],
                    "sanitized": match["model_answer_sanitized"],
                    "first_number": match["model_first_number"],
                },
                "correct_answer": {
                    "raw": gold,
                    "sanitized": match["correct_answer_sanitized"],
                    "first_number": match["correct_first_number"],
                },
                "prompt": prompt,
                "completion": completion,
                **profile,
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()

            rows.append({
                "index": idx,
                "task_id": task_id,
                "correct": ok,
                "no_answer": parsed["answer"] is None,
                "prompt_mode": args.prompt_mode,
                "scoring_mode": args.scoring_mode,
                "extraction_method": parsed.get("method"),
                **profile,
            })
            if (idx + 1) % 10 == 0 or (idx + 1) == len(records):
                print(
                    f"[{idx+1}/{len(records)}] acc={correct}/{total}={correct/total:.4f} "
                    f"no_answer={no_answer}/{total} last_stop={profile['stop_reason']} "
                    f"toks/sec={profile['toks_per_sec']:.2f} pred={parsed['answer']} gold={gold}",
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
        "no_answer": no_answer,
        "no_answer_rate": no_answer / total if total else 0.0,
        "n_eos": int((df["stop_reason"] == "eos").sum()) if len(df) else 0,
        "elapsed_sec": elapsed,
        "avg_new_tokens_eos": mean_col("new_tokens"),
        "avg_calls_eos": mean_col("calls"),
        "avg_iter_per_call_eos": mean_col("avg_iter_per_call"),
        "avg_iter_per_token_eos": mean_col("avg_iter_per_token"),
        "avg_toks_per_sec_eos": mean_col("toks_per_sec"),
        "reason_lens_avg": sum(reason_lens) / len(reason_lens) if reason_lens else 0.0,
        "output": args.output,
        "profile_csv": args.profile_csv,
        "prompt_style": args.prompt_mode,
        "prompt_mode": args.prompt_mode,
        "scoring": f"zeroeval_math_eval_first_number_float_match/{args.scoring_mode}",
        "scoring_mode": args.scoring_mode,
        "block_len": args.block_len,
        "max_new_tokens": args.max_new_tokens,
        "max_calls": args.max_calls,
        "prefill_full_attention_mask": args.prefill_full_attention_mask,
        "attn_implementation": args.attn_implementation,
    }

    print("\n=== GSM8K Jacobi Evalchemy/ZeroEval-Compatible Eval Summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    summary_path = str(Path(args.output).with_suffix(".summary.json"))
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[summary saved] {summary_path}")


if __name__ == "__main__":
    main()
