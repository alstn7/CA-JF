#!/usr/bin/env python3
"""
Evalchemy-compatible Jacobi inference/evaluation for code generation tasks.

Supported tasks:
  - humaneval: Evalchemy HumanEval Python prompt/postprocess/evaluation
  - mbpp: Evalchemy MBPP 3-shot prompt/postprocess/evaluation

This file should be run from the JacobiForcing/JacobiForcing directory so that
`modeling/...` imports resolve correctly.
"""

import argparse
import json
import os
import random
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, Qwen2ForCausalLM

try:
    import pandas as pd
except Exception:
    pd = None

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from modeling.cllm2_qwen2_modeling_kv_terminate_on_eos_improved import jacobi_forward_greedy

Qwen2ForCausalLM.jacobi_forward_greedy = jacobi_forward_greedy


# -----------------------------------------------------------------------------
# Evalchemy HumanEval utils copied/adapted from:
# eval/chat_benchmarks/HumanEval/utils/utils.py
# -----------------------------------------------------------------------------

LANGUAGE_SETTINGS = {
    "python": {"full_name": "Python", "indent": 4},
    "sh": {"full_name": "Bash", "indent": 0},
}


def get_function_name(question: str, lang: str):
    func_lines = [x for x in question.strip().split("\n") if x.strip()]
    if lang.lower() == "python":
        func_idx = [i for i in range(len(func_lines)) if func_lines[i].startswith("def ")][-1]
        func_name = func_lines[func_idx].split("(")[0].strip()
        func_prefix = "\n".join(func_lines[:func_idx])
        return func_name, func_prefix

    func_name = func_lines[-1].split("{")[0].strip()
    func_prefix = "\n".join(func_lines[:-1])
    return func_name, func_prefix


def get_code_block(input_text: str, lang: str):
    lines = input_text.split("\n")
    beg = 0
    while beg < len(lines) - 2 and not lines[beg].strip().startswith("```"):
        beg += 1
    beg += 1

    end = len(lines)
    while end > 1 and not lines[end - 1].strip().endswith("```"):
        end -= 1
    end -= 1

    return "\n".join(lines[beg:end])


def extract_generation_code(example: Dict[str, Any], lang_code: str, verbose: bool = False):
    task_id = example["task_id"]
    output = example.get("output", example.get("gpt_completion", ""))
    question = example["prompt"].strip()
    setting = LANGUAGE_SETTINGS[lang_code]
    lang = setting["full_name"]
    indent = setting["indent"]

    try:
        func_name, func_prefix = get_function_name(question, lang)

        if func_name not in output:
            output = "```" + lang + "\n" + question + "\n" + get_code_block(output, lang_code) + "\n```"

        code_block: str = re.findall(
            f"```{lang.lower()}\n(.*?)```",
            output,
            re.DOTALL | re.IGNORECASE,
        )[0]
        if verbose:
            print(f">>> Task: {task_id}\n{code_block}")

        if setting.get("main", None) and setting["main"] in code_block:
            main_start = code_block.index(setting["main"])
            code_block = code_block[:main_start]

        try:
            start = code_block.lower().index(func_name.lower())
            indent = 0
            while start - indent >= 0 and code_block[start - indent - 1] == " ":
                indent += 1
            try:
                end = code_block.rindex("\n" + " " * indent + "}")
            except Exception:
                end = len(code_block)
        except Exception:
            start = 0
            try:
                end = code_block.rindex("\n" + " " * indent + "}")
            except Exception:
                end = len(code_block)

        body = code_block[start:end]
        generation = func_prefix + "\n" + body + "\n"
        example["generation"] = generation

    except Exception as ex:
        print(
            "Failed to extract code block with error `{}`:\n>>> Task: {}\n>>> Output:\n{}".format(
                ex, task_id, output
            )
        )
        example["generation"] = example["prompt"] + "\n" + output

    return example


# -----------------------------------------------------------------------------
# Evalchemy prompt builders and MBPP extraction
# -----------------------------------------------------------------------------


def build_humaneval_instruction(language: str, question: str) -> str:
    return """
Please continue to complete the function. You are not allowed to modify the given code and do the completion only. Please return all completed function in a codeblock. Here is the given code to do completion:
```{}
{}
```
""".strip().format(language.lower(), question.strip())


def mbpp_format_test_example(question: str, tests: List[str], code: Optional[str] = None) -> str:
    prompt = ">>> Problem:\n{}\n>>> Test Cases:\n{}\n".format(question.strip(), "\n".join(tests))
    if code:
        code = code.replace("\r", "").replace("\t", "    ")
        prompt += "\n>>> Code:\n```python\n{}\n```".format(code)
    return prompt


def read_mbpp_evalchemy_examples(
    mbpp_jsonl: str,
    num_examples: int = 3,
    start_idx: int = 10,
    end_idx: int = 510,
    limit: int = -1,
) -> List[Dict[str, Any]]:
    with open(mbpp_jsonl, "r", encoding="utf-8") as f:
        raw_examples = [json.loads(x) for x in f if x.strip()]

    examples_str = []
    for i in range(1, num_examples + 1):
        ex = raw_examples[i]
        example_prompt = "- Example {}:\n{}".format(
            i, mbpp_format_test_example(ex["text"], ex["test_list"], ex["code"])
        )
        examples_str.append(example_prompt)

    records = []
    for i in range(start_idx, min(end_idx, len(raw_examples))):
        ex = raw_examples[i]
        prompt = mbpp_format_test_example(ex["text"], ex["test_list"])

        if num_examples <= 0:
            prompt_with_shots = """
Generate a Python function for the following problem.
Return only the completed Python code in a markdown code block.

{}
""".strip().format(prompt)
        else:
            prompt_with_shots = """
Please refer the given examples and generate a python function for my problem.
Examples are listed as follows:
{}

Here is my problem:
{}
""".strip().format("\n\n".join(examples_str), prompt)

        records.append({
            "task_id": ex["task_id"],
            "raw_index": i,
            "prompt": prompt_with_shots,
            "text": ex.get("text", ""),
            "test_list": ex.get("test_list", []),
            "test_imports": ex.get("test_imports", []),
            "canonical_solution": ex.get("code", ""),
            "code": ex.get("code", ""),
        })

    if limit is not None and limit > 0:
        records = records[:limit]
    return records


def extract_mbpp_code(completion: str) -> str:
    try:
        return re.findall(r"```python\n(.*?)```", completion, re.DOTALL | re.IGNORECASE)[0]
    except Exception as e:
        print(f"[warn] Failed to extract MBPP python code block; using full completion. Error: {e}")
        return completion


def save_jsonl(records: List[Dict[str, Any]], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def maybe_evaluate(args, task: str) -> Dict[str, Any]:
    if args.no_evaluate:
        return {"evaluation_skipped": True}

    try:
        import sys
        import importlib

        EVALCHEMY_ROOT = os.environ.get(
            "EVALCHEMY_ROOT",
            "/home/irteam/data-vol1/minsu/evalchemy",
        )

        if args.task == "humaneval":
            vendor_root = os.path.join(EVALCHEMY_ROOT, "eval/chat_benchmarks/HumanEval")
        elif args.task == "mbpp":
            vendor_root = os.path.join(EVALCHEMY_ROOT, "eval/chat_benchmarks/MBPP")
        else:
            raise ValueError(f"Unknown task for functional eval: {args.task}")

        if vendor_root in sys.path:
            sys.path.remove(vendor_root)
        sys.path.insert(0, vendor_root)

        for _name in list(sys.modules.keys()):
            if _name == "human_eval" or _name.startswith("human_eval."):
                del sys.modules[_name]

        from human_eval.evaluation import evaluate_functional_correctness
        import inspect
        print(f"[functional_eval] task={args.task}")
        print(f"[functional_eval] vendor_root={vendor_root}")
        print(f"[functional_eval] evaluator={evaluate_functional_correctness}")
        print(f"[functional_eval] signature={inspect.signature(evaluate_functional_correctness)}")
    except Exception as e:
        return {"evaluation_error": f"failed to import human_eval.evaluation: {repr(e)}"}

    tmp_dir_obj = tempfile.TemporaryDirectory()
    tmp_dir = tmp_dir_obj.name
    try:
        if task == "humaneval":
            result = evaluate_functional_correctness(
                input_file=args.output,
                tmp_dir=tmp_dir,
                n_workers=args.num_workers,
                timeout=args.timeout,
                problem_file=os.path.join(args.humaneval_data_dir, "humaneval-python.jsonl"),
                language="python",
            )
        elif task == "mbpp":
            result = evaluate_functional_correctness(
                input_file=args.output,
                tmp_dir=tmp_dir,
                problem_file=os.path.join(args.mbpp_data_dir, "mbpp_test.jsonl"),
                language="python",
                is_mbpp=True,
            )
        else:
            raise ValueError(task)
        return dict(result)
    except Exception as e:
        return {"evaluation_error": repr(e)}
    finally:
        tmp_dir_obj.cleanup()


def build_records(args) -> List[Dict[str, Any]]:
    if args.task == "humaneval":
        problem_file = os.path.join(args.humaneval_data_dir, "humaneval-python.jsonl")
        records = [json.loads(x) for x in open(problem_file, encoding="utf-8") if x.strip()]
        if args.limit is not None and args.limit > 0:
            records = records[: args.limit]
        return records

    if args.task == "mbpp":
        return read_mbpp_evalchemy_examples(
            os.path.join(args.mbpp_data_dir, "mbpp.jsonl"),
            num_examples=args.mbpp_num_examples,
            start_idx=args.mbpp_start_idx,
            end_idx=args.mbpp_end_idx,
            limit=args.limit,
        )

    raise ValueError(args.task)


# -----------------------------------------------------------------------------
# Jacobi generation
# -----------------------------------------------------------------------------


def jacobi_generate_one(
    model,
    tokenizer,
    prompt_text: str,
    block_len: int,
    max_new_tokens: int,
    max_calls: int,
    eos_id: Optional[int],
    alt_eos_id: int = 151645,
    decode_skip_special_tokens: bool = True,
):
    model_inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)
    input_ids = model_inputs["input_ids"]
    attention_mask = torch.full_like(input_ids, 1, device=model.device)

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
        if eos_id is not None and generated_part.numel() > 0:
            hit_eos = (generated_part == eos_id).any().item()
        if not hit_eos and generated_part.numel() > 0:
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
            source_tokens = generated_ids[0].tolist()
            for _ in range(block_len):
                q_sample = torch.tensor(
                    [random.choice(source_tokens)],
                    dtype=torch.long,
                    device=model.device,
                ).unsqueeze(0)
                q_sampled.append(q_sample)
            prefill_draft_token_ids = torch.cat(q_sampled, dim=1)
            prefill_input_ids = torch.cat((input_ids, prefill_draft_token_ids), dim=-1)

            past_key_values, first_correct_token, prefill_drafted_n_gram, iter_count = model.jacobi_forward_greedy(
                input_ids=prefill_input_ids,
                attention_mask=attention_mask,
                past_key_values=None,
                use_cache=True,
                prefill_phase=prefill_phase,
                n_token_seq_len=block_len,
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
                source_tokens = generated_ids[0].tolist()
                for _ in range(block_len - 1):
                    q_sample = torch.tensor(
                        [random.choice(source_tokens)],
                        dtype=torch.long,
                        device=model.device,
                    ).unsqueeze(0)
                    q_sampled.append(q_sample)
                q_sampled = torch.cat(q_sampled, dim=1)
                input_ids = torch.cat((first_correct_token.view(1, -1), q_sampled), dim=-1)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_gen_start = time.perf_counter()
            past_key_values, first_correct_token, accepted_n_gram, itr_count = model.jacobi_forward_greedy(
                input_ids=input_ids,
                attention_mask=None,
                past_key_values=past_key_values,
                use_cache=True,
                prefill_phase=prefill_phase,
                n_token_seq_len=block_len,
                tokenizer=tokenizer,
                eos_token_id=eos_id,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            gen_only_time += time.perf_counter() - t_gen_start
            generated_ids = torch.cat((generated_ids, accepted_n_gram), dim=-1)

        calls += 1
        iters.append(itr_count)
        added = generated_ids.shape[1] - prev_len
        if added > 0:
            total_new_tokens += added
        prev_len = generated_ids.shape[1]

    # Keep compatibility with previous scripts: prefill adds one bookkeeping token.
    total_new_tokens = max(total_new_tokens - 1, 0)
    wall_time = time.perf_counter() - t_start
    total_iterations = int(sum(iters))
    safe_calls = max(calls, 1)
    safe_new_tokens = max(total_new_tokens, 1)
    safe_gen_time = max(gen_only_time, 1e-6)

    completion = tokenizer.decode(
        generated_ids[0, prompt_len:],
        skip_special_tokens=decode_skip_special_tokens,
    )

    profile = {
        "prompt_tokens": int(prompt_len),
        "new_tokens": int(total_new_tokens),
        "calls": int(calls),
        "total_iterations": total_iterations,
        "avg_iter_per_call": total_iterations / safe_calls,
        "avg_iter_per_token": total_iterations / safe_new_tokens,
        "time_sec": wall_time,
        "gen_only_time": gen_only_time,
        "toks_per_sec": safe_new_tokens / safe_gen_time,
        "stop_reason": stop_reason,
    }
    return completion, profile


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["humaneval", "mbpp"], required=True)
    p.add_argument("--model_path", "--model-path", dest="model_path", required=True)
    p.add_argument("--tokenizer_path", "--tokenizer-path", dest="tokenizer_path", required=True)
    p.add_argument("--output", "--output-path", dest="output", required=True)
    p.add_argument("--profile_csv", "--profile-path", dest="profile_csv", default=None)

    p.add_argument("--humaneval_data_dir", default="/home/irteam/data-vol1/minsu/evalchemy/eval/chat_benchmarks/HumanEval/data")
    p.add_argument("--mbpp_data_dir", default="/home/irteam/data-vol1/minsu/evalchemy/eval/chat_benchmarks/MBPP/data")
    p.add_argument("--mbpp_num_examples", type=int, default=3)
    p.add_argument("--mbpp_start_idx", type=int, default=10)
    p.add_argument("--mbpp_end_idx", type=int, default=510)

    p.add_argument("--limit", type=int, default=-1)
    p.add_argument("--block_len", "--jacobi-block-len", dest="block_len", type=int, default=16)
    p.add_argument("--max_new_tokens", "--max-new-tokens", dest="max_new_tokens", type=int, default=None)
    p.add_argument("--max_calls", "--max-calls", dest="max_calls", type=int, default=1024)
    p.add_argument("--attn_implementation", default="flash_attention_2")
    p.add_argument("--prefill_full_attention_mask", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--timeout", type=float, default=3.0)
    p.add_argument("--no_evaluate", action="store_true")
    p.add_argument("--decode_skip_special_tokens", action="store_true", default=True)
    args = p.parse_args()

    if args.max_new_tokens is None:
        args.max_new_tokens = 1024 if args.task == "humaneval" else 512

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    records = build_records(args)
    print(f"Loaded {args.task} records: {len(records)}")

    model = Qwen2ForCausalLM.from_pretrained(
        args.model_path,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    eos_id = tokenizer.eos_token_id
    generated_examples = []
    profile_rows = []
    total_wall_start = time.perf_counter()

    for idx, row in enumerate(tqdm(records)):
        if args.task == "humaneval":
            prompt = build_humaneval_instruction("Python", row["prompt"])
            task_id = row.get("task_id", f"Python/{idx}")
        else:
            prompt = row["prompt"]
            task_id = row.get("task_id", f"MBPP/{idx}")

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        completion, profile = jacobi_generate_one(
            model=model,
            tokenizer=tokenizer,
            prompt_text=text,
            block_len=args.block_len,
            max_new_tokens=args.max_new_tokens,
            max_calls=args.max_calls,
            eos_id=eos_id,
            decode_skip_special_tokens=args.decode_skip_special_tokens,
        )

        if args.task == "humaneval":
            ex = row.copy()
            ex["output"] = completion
            ex = extract_generation_code(ex, lang_code="python")
        else:
            ex = {
                "task_id": task_id,
                "prompt": row.get("prompt", ""),
                "gpt_completion": completion,
                "output": completion,
                "generation": extract_mbpp_code(completion),
            }

        ex["evalchemy_task"] = args.task
        ex["block_len"] = args.block_len
        ex["max_new_tokens"] = args.max_new_tokens
        generated_examples.append(ex)

        row_profile = {
            "index": idx,
            "task_id": task_id,
            **profile,
        }
        profile_rows.append(row_profile)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(records):
            print(
                f"[{idx+1}/{len(records)}] task_id={task_id} "
                f"new_toks={profile['new_tokens']} calls={profile['calls']} "
                f"avg_iter/call={profile['avg_iter_per_call']:.2f} "
                f"toks/sec={profile['toks_per_sec']:.2f} reason={profile['stop_reason']}"
            )

    save_jsonl(generated_examples, args.output)
    elapsed = time.perf_counter() - total_wall_start

    if args.profile_csv is None:
        args.profile_csv = os.path.splitext(args.output)[0] + "_profile.csv"
    os.makedirs(os.path.dirname(args.profile_csv), exist_ok=True)

    if pd is not None:
        df = pd.DataFrame(profile_rows)
        df.to_csv(args.profile_csv, index=False)
        df_eos = df[df["stop_reason"] == "eos"].copy()
        df_eos.to_csv(os.path.splitext(args.profile_csv)[0] + "_eos.csv", index=False)
        use_df = df_eos if len(df_eos) else df
        avg_new_tokens = float(use_df["new_tokens"].mean()) if len(use_df) else 0.0
        avg_calls = float(use_df["calls"].mean()) if len(use_df) else 0.0
        avg_iter_per_call = float(use_df["avg_iter_per_call"].mean()) if len(use_df) else 0.0
        avg_iter_per_token = float(use_df["avg_iter_per_token"].mean()) if len(use_df) else 0.0
        avg_toks_per_sec = float(use_df["toks_per_sec"].mean()) if len(use_df) else 0.0
        n_eos = int((df["stop_reason"] == "eos").sum())
    else:
        avg_new_tokens = sum(r["new_tokens"] for r in profile_rows) / max(len(profile_rows), 1)
        avg_calls = sum(r["calls"] for r in profile_rows) / max(len(profile_rows), 1)
        avg_iter_per_call = sum(r["avg_iter_per_call"] for r in profile_rows) / max(len(profile_rows), 1)
        avg_iter_per_token = sum(r["avg_iter_per_token"] for r in profile_rows) / max(len(profile_rows), 1)
        avg_toks_per_sec = sum(r["toks_per_sec"] for r in profile_rows) / max(len(profile_rows), 1)
        n_eos = sum(1 for r in profile_rows if r["stop_reason"] == "eos")

    eval_result = maybe_evaluate(args, args.task)

    summary = {
        "task": args.task,
        "total": len(records),
        "n_eos": n_eos,
        "elapsed_sec": elapsed,
        "avg_new_tokens_eos_or_all": avg_new_tokens,
        "avg_calls_eos_or_all": avg_calls,
        "avg_iter_per_call_eos_or_all": avg_iter_per_call,
        "avg_iter_per_token_eos_or_all": avg_iter_per_token,
        "avg_toks_per_sec_eos_or_all": avg_toks_per_sec,
        "output": args.output,
        "profile_csv": args.profile_csv,
        "prompt_style": "evalchemy_humaneval_deepseekcoder" if args.task == "humaneval" else ("evalchemy_mbpp_0shot" if args.task == "mbpp" and args.mbpp_num_examples <= 0 else "evalchemy_mbpp_3shot"),
        "postprocess": "evalchemy_extract_generation_code" if args.task == "humaneval" else "evalchemy_extract_first_python_codeblock",
        "block_len": args.block_len,
        "max_new_tokens": args.max_new_tokens,
        "max_calls": args.max_calls,
        "prefill_full_attention_mask": bool(args.prefill_full_attention_mask),
        "attn_implementation": args.attn_implementation,
        "evaluation": eval_result,
    }
    summary_path = os.path.splitext(args.output)[0] + ".summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Codegen Jacobi Evalchemy-Compatible Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[summary saved] {summary_path}")


if __name__ == "__main__":
    main()
