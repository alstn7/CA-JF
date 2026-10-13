#!/usr/bin/env python3

import argparse
import csv
import io
import json
import random
import sys
import tokenize as py_tokenize
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from utils.hierarchical_targets import make_batch_anchor_mask


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--data",
        default=(
            "OpenCodeInstruct_training_data_n16w16/"
            "merged_data_v2_8_30_opencodeinstruct_progressive_noise_cyclic_all.jsonl"
        ),
    )
    p.add_argument(
        "--model",
        default="models/Qwen2.5-Coder-7B-Instruct",
    )

    p.add_argument("--block-len", type=int, default=16)
    p.add_argument("--num-examples", type=int, default=100)
    p.add_argument("--positions-per-example", type=int, default=8)
    p.add_argument(
        "--matched-per-group",
        type=int,
        default=8,
        help="Unused in discovery mode; retained for compatibility.",
    )
    p.add_argument(
        "--discovery-positions-per-example",
        type=int,
        default=16,
        help="Uniformly sampled executable-code positions per example.",
    )
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--topk-alternatives", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--max-seq-len", type=int, default=2048)

    p.add_argument(
        "--output",
        default="analysis/code_counterfactual_influence_discovery.tsv",
    )

    return p.parse_args()


def find_last_subseq(seq, pat):
    last = -1
    n = len(pat)

    for i in range(len(seq) - n + 1):
        if seq[i:i+n] == pat:
            last = i

    return last


def reconstruct_clean_example(row, tok, block_len):
    seq = row["complete_training_sequence_ids"]
    prompt_len = int(row["prompt_ids_len"])
    traj = row["traj_position_indices"]
    T = len(traj)

    expected = prompt_len + 2 * T * block_len

    if len(seq) != expected:
        return None

    prompt = seq[:prompt_len]
    cont = seq[prompt_len:]

    clean = []

    for j in range(T):
        s = j * 2 * block_len
        clean.extend(
            cont[s + block_len:s + 2 * block_len]
        )

    # Stop at first assistant end marker.
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")

    if im_end_id is not None and im_end_id in clean:
        clean = clean[:clean.index(im_end_id)]

    assistant_marker = tok.encode(
        "<|im_start|>assistant\n",
        add_special_tokens=False,
    )

    marker_pos = find_last_subseq(
        prompt,
        assistant_marker,
    )

    if marker_pos < 0:
        return None

    generation_prefix = prompt[
        :marker_pos + len(assistant_marker)
    ]

    return {
        "prefix": generation_prefix,
        "response": clean,
        "T": T,
    }


def find_python_code_and_comment_positions(response_ids, tok):
    """
    Find token positions belonging to the first fenced Python code body,
    and separately identify positions overlapping Python comments.

    Returns:
      code_positions: set[int]
      comment_positions: set[int]
    """
    text = tok.decode(
        response_ids,
        skip_special_tokens=False,
    )

    fence = "`" * 3

    open_candidates = [
        fence + "python",
        fence + "Python",
        fence,
    ]

    open_start = -1
    open_end = -1

    for marker in open_candidates:
        pos = text.find(marker)
        if pos >= 0:
            open_start = pos
            open_end = pos + len(marker)

            while (
                open_end < len(text)
                and text[open_end] in "\r\n"
            ):
                open_end += 1
            break

    if open_start < 0:
        return set(), set()

    close_start = text.find(
        fence,
        open_end,
    )

    if close_start < 0:
        close_start = len(text)

    code_text = text[open_end:close_start]

    comment_char_spans = []

    try:
        lines = code_text.splitlines(keepends=True)

        line_starts = []
        cur = 0

        for line in lines:
            line_starts.append(cur)
            cur += len(line)

        def to_abs_char(row, col):
            if row <= 0:
                return 0

            if row > len(line_starts):
                return len(code_text)

            return line_starts[row - 1] + col

        tokens = py_tokenize.generate_tokens(
            io.StringIO(code_text).readline
        )

        for token_info in tokens:
            if token_info.type != py_tokenize.COMMENT:
                continue

            srow, scol = token_info.start
            erow, ecol = token_info.end

            c0 = to_abs_char(srow, scol)
            c1 = to_abs_char(erow, ecol)

            comment_char_spans.append(
                (
                    open_end + c0,
                    open_end + c1,
                )
            )

    except (
        py_tokenize.TokenError,
        IndentationError,
        SyntaxError,
    ):
        comment_char_spans = []

    code_positions = set()
    comment_positions = set()

    cursor = 0

    for i, tid in enumerate(response_ids):
        piece = tok.decode(
            [tid],
            skip_special_tokens=False,
        )

        start = cursor
        end = cursor + len(piece)
        cursor = end

        if end > open_end and start < close_start:
            code_positions.add(i)

            for c0, c1 in comment_char_spans:
                if end > c0 and start < c1:
                    comment_positions.add(i)
                    break

    return code_positions, comment_positions

def build_manual_anchor_mask(response_ids, tok):
    if not response_ids:
        return []

    labels = torch.tensor(
        [response_ids],
        dtype=torch.long,
    )

    mask = make_batch_anchor_mask(
        tok,
        labels,
        ignore_index=-100,
    ).squeeze(0)

    return mask.bool().cpu().tolist()


def choose_valid_positions(
    response_ids,
    tok,
    code_positions,
    comment_positions,
    required_horizon,
):
    specials = set(tok.all_special_ids)

    valid = []

    for i, tid in enumerate(response_ids):
        if i not in code_positions:
            continue

        if i in comment_positions:
            continue

        if i == 0:
            continue

        # For a clean multi-horizon comparison, every sampled position
        # must have the full requested downstream horizon available.
        #
        # We measure tokens:
        #   i+1, ..., i+required_horizon
        #
        # so the final required index is i + required_horizon.
        if i + required_horizon >= len(response_ids):
            continue

        if tid in specials:
            continue

        piece = tok.decode(
            [tid],
            skip_special_tokens=False,
        )

        if piece.strip() == "":
            continue

        valid.append(i)

    return valid


def build_random_same_count_mask(
    valid_positions,
    manual_mask,
    rng,
):
    """
    Offline audit control:
    choose exactly the same number of random valid code positions
    as manual semantic anchors in this example.
    """
    valid_set = set(valid_positions)

    manual_valid = [
        i for i in valid_positions
        if manual_mask[i]
    ]

    k = len(manual_valid)

    random_mask = [False] * len(manual_mask)

    if k == 0:
        return random_mask

    candidates = list(valid_set)

    if k >= len(candidates):
        chosen = candidates
    else:
        chosen = rng.sample(
            candidates,
            k,
        )

    for i in chosen:
        random_mask[i] = True

    return random_mask



def choose_matched_groups(
    valid_positions,
    manual_mask,
    max_per_group,
    rng,
):
    """
    Select equal-size manual, non-manual, and random groups
    within each example.

    Random is sampled from all valid executable-code positions,
    matching the number of selected semantic anchors.
    """
    manual_positions = [
        i
        for i in valid_positions
        if manual_mask[i]
    ]

    nonmanual_positions = [
        i
        for i in valid_positions
        if not manual_mask[i]
    ]

    n = min(
        max_per_group,
        len(manual_positions),
        len(nonmanual_positions),
    )

    if n <= 0:
        return None

    manual_selected = set(
        rng.sample(
            manual_positions,
            n,
        )
    )

    nonmanual_selected = set(
        rng.sample(
            nonmanual_positions,
            n,
        )
    )

    random_selected = set(
        rng.sample(
            valid_positions,
            n,
        )
    )

    union_positions = sorted(
        manual_selected
        | nonmanual_selected
        | random_selected
    )

    return {
        "n_per_group": n,
        "manual": manual_selected,
        "nonmanual": nonmanual_selected,
        "random": random_selected,
        "union": union_positions,
    }


def norm_piece(piece):
    return (
        piece.replace("Ġ", "")
             .replace("▁", "")
             .replace("Ċ", "\n")
             .strip()
    )


def get_plausible_alternative(
    logits_for_position,
    target_id,
    tok,
    topk=50,
):
    """
    Choose the highest-probability valid non-target alternative.

    Also return perturbation-plausibility diagnostics:
      target_logprob
      alternative_logprob
      logprob_gap = target_logprob - alternative_logprob
      target_rank
      alternative_rank

    Ranks are raw vocabulary ranks under the base model:
      rank 1 = highest-logit token.
    """
    logits = logits_for_position.float()

    k = min(
        topk,
        logits.shape[-1],
    )

    top_values, top_ids = torch.topk(
        logits,
        k=k,
    )

    specials = set(tok.all_special_ids)

    target_piece = tok.decode(
        [target_id],
        skip_special_tokens=False,
    )

    target_norm = norm_piece(
        target_piece
    )

    log_probs = torch.log_softmax(
        logits,
        dim=-1,
    )

    target_logprob = float(
        log_probs[target_id].item()
    )

    target_logit = logits[target_id]

    # Full-vocabulary rank without sorting the entire vocabulary.
    target_rank = int(
        (logits > target_logit)
        .sum()
        .item()
    ) + 1

    for raw_rank, alt_id in enumerate(
        top_ids.tolist(),
        start=1,
    ):
        if alt_id == target_id:
            continue

        if alt_id in specials:
            continue

        alt_piece = tok.decode(
            [alt_id],
            skip_special_tokens=False,
        )

        if alt_piece.strip() == "":
            continue

        alt_norm = norm_piece(
            alt_piece
        )

        # Skip whitespace/tokenizer-equivalent variants.
        if alt_norm == target_norm:
            continue

        alt_logprob = float(
            log_probs[alt_id].item()
        )

        logprob_gap = (
            target_logprob
            - alt_logprob
        )

        return {
            "alt_id": int(alt_id),
            "target_logprob": target_logprob,
            "alternative_logprob": alt_logprob,
            "logprob_gap": logprob_gap,
            "target_rank": target_rank,
            "alternative_rank": int(raw_rank),
        }

    return None


@torch.inference_mode()
def forward_logits(model, input_ids):
    out = model(
        input_ids=input_ids,
        use_cache=False,
        return_dict=True,
    )

    return out.logits


def main():
    args = parse_args()

    rng = random.Random(args.seed)

    print(
        "Loading tokenizer:",
        args.model,
        flush=True,
    )

    tok = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )

    print(
        "Loading model:",
        args.model,
        flush=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    model.eval()

    device = next(model.parameters()).device

    out_path = Path(args.output)
    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "data_id",
        "raw_index",
        "response_pos",
        "response_pos_frac",
        "abs_pos",
        "target_id",
        "target_piece",
        "alternative_id",
        "alternative_piece",
        "target_logprob",
        "alternative_logprob",
        "logprob_gap",
        "target_rank",
        "alternative_rank",
        "manual_anchor",
        "discovery_sample",
        "matched_manual",
        "matched_nonmanual",
        "random_anchor",
        "matched_group_size",
        "is_comment",
        "horizon_used",
        "clean_mean_nll",
        "perturbed_mean_nll",
        "delta_nll",
        "delta_nll_h1",
        "delta_nll_h4",
        "delta_nll_h8",
        "delta_nll_h16",
        "T",
        "response_len",
        "valid_code_positions",
        "manual_anchor_count",
        "random_anchor_count",
    ]

    rows_written = 0
    examples_used = 0
    skipped_no_code = 0
    skipped_no_positions = 0

    with open(
        args.data,
        "r",
        encoding="utf-8",
    ) as fin, open(
        out_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as fout:

        writer = csv.DictWriter(
            fout,
            fieldnames=fieldnames,
            delimiter="\t",
        )

        writer.writeheader()

        for raw_idx, line in enumerate(fin):
            if raw_idx < args.offset:
                continue

            if examples_used >= args.num_examples:
                break

            row = json.loads(line)

            rec = reconstruct_clean_example(
                row,
                tok,
                args.block_len,
            )

            if rec is None:
                continue

            prefix = rec["prefix"]
            response = rec["response"]

            if len(response) < 4:
                continue

            full_ids = prefix + response

            if len(full_ids) > args.max_seq_len:
                continue

            (
                code_positions,
                comment_positions,
            ) = find_python_code_and_comment_positions(
                response,
                tok,
            )

            if not code_positions:
                skipped_no_code += 1
                continue

            manual_mask = build_manual_anchor_mask(
                response,
                tok,
            )

            valid_positions = choose_valid_positions(
                response,
                tok,
                code_positions,
                comment_positions,
                args.horizon,
            )

            if not valid_positions:
                skipped_no_positions += 1
                continue

            if (
                len(valid_positions)
                <= args.discovery_positions_per_example
            ):
                positions = list(valid_positions)
            else:
                positions = sorted(
                    rng.sample(
                        valid_positions,
                        args.discovery_positions_per_example,
                    )
                )

            if not positions:
                skipped_no_positions += 1
                continue

            manual_anchor_count = sum(
                1
                for i in valid_positions
                if manual_mask[i]
            )

            random_anchor_count = 0

            response_start = len(prefix)

            clean_input = torch.tensor(
                [full_ids],
                dtype=torch.long,
                device=device,
            )

            clean_logits = forward_logits(
                model,
                clean_input,
            )

            perturb_specs = []

            for rpos in positions:
                abs_pos = response_start + rpos
                target_id = int(response[rpos])

                if abs_pos <= 0:
                    continue

                alt_info = get_plausible_alternative(
                    clean_logits[0, abs_pos - 1],
                    target_id,
                    tok,
                    args.topk_alternatives,
                )

                if alt_info is None:
                    continue

                alt_id = alt_info["alt_id"]

                end_pos = (
                    abs_pos
                    + 1
                    + args.horizon
                )

                response_end = (
                    response_start
                    + len(response)
                )

                assert end_pos <= response_end, (
                    "valid-position horizon invariant violated: "
                    f"end_pos={end_pos}, response_end={response_end}"
                )

                perturb_specs.append(
                    (
                        rpos,
                        abs_pos,
                        target_id,
                        alt_id,
                        end_pos,
                        alt_info,
                    )
                )

            if not perturb_specs:
                continue

            perturbed = clean_input.repeat(
                len(perturb_specs),
                1,
            )

            for b, spec in enumerate(
                perturb_specs
            ):
                (
                    _,
                    abs_pos,
                    _,
                    alt_id,
                    _,
                    _,
                ) = spec

                perturbed[b, abs_pos] = alt_id

            pert_logits = forward_logits(
                model,
                perturbed,
            )

            for b, spec in enumerate(
                perturb_specs
            ):
                (
                    rpos,
                    abs_pos,
                    target_id,
                    alt_id,
                    end_pos,
                    alt_info,
                ) = spec

                clean_nlls = []
                pert_nlls = []

                for q in range(
                    abs_pos + 1,
                    end_pos,
                ):
                    target = clean_input[0, q]

                    clean_nll = F.cross_entropy(
                        clean_logits[
                            0,
                            q - 1,
                        ].float().unsqueeze(0),
                        target.unsqueeze(0),
                        reduction="mean",
                    )

                    pert_nll = F.cross_entropy(
                        pert_logits[
                            b,
                            q - 1,
                        ].float().unsqueeze(0),
                        target.unsqueeze(0),
                        reduction="mean",
                    )

                    clean_nlls.append(
                        float(
                            clean_nll.item()
                        )
                    )

                    pert_nlls.append(
                        float(
                            pert_nll.item()
                        )
                    )

                if not clean_nlls:
                    continue

                clean_mean = (
                    sum(clean_nlls)
                    / len(clean_nlls)
                )

                pert_mean = (
                    sum(pert_nlls)
                    / len(pert_nlls)
                )

                delta = (
                    pert_mean
                    - clean_mean
                )

                def horizon_delta(h):
                    if len(clean_nlls) < h:
                        return None

                    c = (
                        sum(clean_nlls[:h])
                        / h
                    )

                    p = (
                        sum(pert_nlls[:h])
                        / h
                    )

                    return p - c

                delta_h1 = horizon_delta(1)
                delta_h4 = horizon_delta(4)
                delta_h8 = horizon_delta(8)
                delta_h16 = horizon_delta(16)

                writer.writerow({
                    "data_id":
                        row["data_id"],

                    "raw_index":
                        raw_idx,

                    "response_pos":
                        rpos,

                    "response_pos_frac":
                        f"{rpos / max(1, len(response) - 1):.8f}",

                    "abs_pos":
                        abs_pos,

                    "target_id":
                        target_id,

                    "target_piece":
                        repr(
                            tok.decode(
                                [target_id],
                                skip_special_tokens=False,
                            )
                        ),

                    "alternative_id":
                        alt_id,

                    "alternative_piece":
                        repr(
                            tok.decode(
                                [alt_id],
                                skip_special_tokens=False,
                            )
                        ),

                    "target_logprob":
                        f"{alt_info['target_logprob']:.8f}",

                    "alternative_logprob":
                        f"{alt_info['alternative_logprob']:.8f}",

                    "logprob_gap":
                        f"{alt_info['logprob_gap']:.8f}",

                    "target_rank":
                        alt_info["target_rank"],

                    "alternative_rank":
                        alt_info["alternative_rank"],

                    "manual_anchor":
                        int(
                            bool(
                                manual_mask[rpos]
                            )
                        ),

                    "discovery_sample":
                        1,

                    "matched_manual":
                        0,

                    "matched_nonmanual":
                        0,

                    "random_anchor":
                        0,

                    "matched_group_size":
                        0,

                    "is_comment":
                        int(
                            rpos
                            in comment_positions
                        ),

                    "horizon_used":
                        len(clean_nlls),

                    "clean_mean_nll":
                        f"{clean_mean:.8f}",

                    "perturbed_mean_nll":
                        f"{pert_mean:.8f}",

                    "delta_nll":
                        f"{delta:.8f}",

                    "delta_nll_h1":
                        (
                            ""
                            if delta_h1 is None
                            else f"{delta_h1:.8f}"
                        ),

                    "delta_nll_h4":
                        (
                            ""
                            if delta_h4 is None
                            else f"{delta_h4:.8f}"
                        ),

                    "delta_nll_h8":
                        (
                            ""
                            if delta_h8 is None
                            else f"{delta_h8:.8f}"
                        ),

                    "delta_nll_h16":
                        (
                            ""
                            if delta_h16 is None
                            else f"{delta_h16:.8f}"
                        ),

                    "T":
                        rec["T"],

                    "response_len":
                        len(response),

                    "valid_code_positions":
                        len(valid_positions),

                    "manual_anchor_count":
                        manual_anchor_count,

                    "random_anchor_count":
                        random_anchor_count,
                })

                rows_written += 1

            fout.flush()

            examples_used += 1

            if examples_used % 10 == 0:
                print(
                    "[progress]"
                    f" examples={examples_used}"
                    f" positions={rows_written}"
                    f" no_code={skipped_no_code}",
                    flush=True,
                )

    print()
    print("===== DONE =====")
    print("examples_used:", examples_used)
    print("positions_written:", rows_written)
    print("skipped_no_code:", skipped_no_code)
    print(
        "skipped_no_positions:",
        skipped_no_positions,
    )
    print("output:", out_path)


if __name__ == "__main__":
    main()
