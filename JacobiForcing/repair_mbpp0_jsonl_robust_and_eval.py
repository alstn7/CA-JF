#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path


def robust_extract_mbpp_code(completion: str) -> str:
    text = completion.replace("\r\n", "\n").replace("\r", "\n").strip()

    # 1) Standard fenced python block: ```python\n...\n```
    m = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m[0].strip() + "\n"

    # 2) Common broken style: ```\npython\n...\n```
    m = re.findall(r"```\s*\npython\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m[0].strip() + "\n"

    # 3) Generic fenced block: ```\n...\n```
    m = re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
    if m:
        code = m[0].strip()
        if code.lower().startswith("python\n"):
            code = code.split("\n", 1)[1].strip()
        return code + "\n"

    # 4) No fence: start from first likely code line.
    lines = text.split("\n")
    start = None
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith(("def ", "import ", "from ", "class ")):
            start = i
            break

    if start is not None:
        code = "\n".join(lines[start:]).strip()
        stop_markers = [
            "\nExplanation:",
            "\nThe function",
            "\nThis function",
            "\nNote:",
            "\nTest Cases:",
            "\nOutput:",
        ]
        for marker in stop_markers:
            pos = code.find(marker)
            if pos > 0:
                code = code[:pos].strip()
        return code + "\n"

    return text + "\n"


def repair_jsonl(input_path: Path, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    changed = 0
    with input_path.open("r", encoding="utf-8") as f, output_path.open("w", encoding="utf-8") as g:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            completion = ex.get("gpt_completion", ex.get("output", ""))
            old_generation = ex.get("generation", "")
            new_generation = robust_extract_mbpp_code(completion)
            ex["generation"] = new_generation
            ex["postprocess"] = "robust_mbpp0_extract_code"
            if new_generation != old_generation:
                changed += 1
            n += 1
            g.write(json.dumps(ex, ensure_ascii=False) + "\n")

    return n, changed


def eval_mbpp(output_path: Path, mbpp_data_dir: str, timeout: float = 3.0):
    EVALCHEMY_ROOT = os.environ.get("EVALCHEMY_ROOT", "/home/irteam/data-vol1/minsu/evalchemy")
    vendor_root = os.path.join(EVALCHEMY_ROOT, "eval/chat_benchmarks/MBPP")
    if vendor_root in sys.path:
        sys.path.remove(vendor_root)
    sys.path.insert(0, vendor_root)

    for name in list(sys.modules.keys()):
        if name == "human_eval" or name.startswith("human_eval."):
            del sys.modules[name]

    from human_eval.evaluation import evaluate_functional_correctness

    with tempfile.TemporaryDirectory() as tmp_dir:
        result = evaluate_functional_correctness(
            input_file=str(output_path),
            tmp_dir=tmp_dir,
            problem_file=os.path.join(mbpp_data_dir, "mbpp_test.jsonl"),
            language="python",
            is_mbpp=True,
        )
    return dict(result)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--mbpp_data_dir", default="/home/irteam/data-vol1/minsu/evalchemy/eval/chat_benchmarks/MBPP/data")
    ap.add_argument("--timeout", type=float, default=3.0)
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    summary = Path(args.summary)

    n, changed = repair_jsonl(inp, out)
    result = eval_mbpp(out, args.mbpp_data_dir, timeout=args.timeout)

    s = {
        "input": str(inp),
        "output": str(out),
        "num_records": n,
        "num_generation_changed": changed,
        "evaluation": result,
    }
    summary.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(s, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
