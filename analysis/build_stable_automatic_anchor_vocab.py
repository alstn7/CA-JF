#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--input",
        default="analysis/code_counterfactual_influence_discovery_2000.tsv",
    )

    p.add_argument(
        "--output-prefix",
        default="analysis/automatic_anchor_stable",
    )

    p.add_argument(
        "--min-total-count",
        type=int,
        default=20,
    )

    p.add_argument(
        "--min-split-count",
        type=int,
        default=8,
    )

    p.add_argument(
        "--min-positive-fraction",
        type=float,
        default=0.80,
    )

    return p.parse_args()


def stable_half(data_id):
    h = hashlib.blake2b(
        data_id.encode("utf-8"),
        digest_size=8,
        person=b"CAJFauto",
    ).digest()

    x = int.from_bytes(
        h,
        byteorder="little",
        signed=False,
    )

    return x % 2


def median(xs):
    return statistics.median(xs)


def mean(xs):
    return statistics.mean(xs)


def positive_fraction(xs):
    return sum(x > 0 for x in xs) / len(xs)


def collect_stats(rows):
    by_token = defaultdict(list)

    for r in rows:
        by_token[r["target_id"]].append(r)

    out = {}

    for tid, rs in by_token.items():
        vals = [
            x["delta"]
            for x in rs
        ]

        out[tid] = {
            "count": len(vals),
            "median": median(vals),
            "mean": mean(vals),
            "positive_fraction": positive_fraction(vals),
            "token_piece": rs[0]["target_piece"],
            "manual_anchor": rs[0]["manual_anchor"],
        }

    return out


def main():
    args = parse_args()

    rows = []

    with open(
        args.input,
        encoding="utf-8",
    ) as f:
        for r in csv.DictReader(
            f,
            delimiter="\t",
        ):
            rows.append({
                "data_id": r["data_id"],
                "target_id": int(r["target_id"]),
                "target_piece": r["target_piece"],
                "manual_anchor": int(r["manual_anchor"]),
                "delta": float(r["delta_nll_h16"]),
            })

    split_a = [
        r for r in rows
        if stable_half(r["data_id"]) == 0
    ]

    split_b = [
        r for r in rows
        if stable_half(r["data_id"]) == 1
    ]

    all_stats = collect_stats(rows)
    a_stats = collect_stats(split_a)
    b_stats = collect_stats(split_b)

    # ------------------------------------------------------------
    # Discovery-sample manual coverage.
    # This is the target occurrence coverage for automatic anchors.
    # ------------------------------------------------------------

    manual_occ = sum(
        r["manual_anchor"]
        for r in rows
    )

    manual_coverage = (
        manual_occ / len(rows)
    )

    # ------------------------------------------------------------
    # Stable automatic token candidates.
    #
    # No hand-written semantic filtering:
    #   - sufficient total support
    #   - sufficient support in both halves
    #   - positive median in both halves
    #   - high positive fraction overall
    #
    # Ranking score is conservative:
    # minimum split median.
    # ------------------------------------------------------------

    candidates = []

    for tid, st in all_stats.items():
        if st["count"] < args.min_total_count:
            continue

        if tid not in a_stats or tid not in b_stats:
            continue

        sa = a_stats[tid]
        sb = b_stats[tid]

        if sa["count"] < args.min_split_count:
            continue

        if sb["count"] < args.min_split_count:
            continue

        if sa["median"] <= 0:
            continue

        if sb["median"] <= 0:
            continue

        if (
            st["positive_fraction"]
            < args.min_positive_fraction
        ):
            continue

        robust_score = min(
            sa["median"],
            sb["median"],
        )

        candidates.append({
            "token_id": tid,
            "token_piece": st["token_piece"],
            "manual_anchor": st["manual_anchor"],
            "count": st["count"],
            "count_a": sa["count"],
            "count_b": sb["count"],
            "median_all": st["median"],
            "median_a": sa["median"],
            "median_b": sb["median"],
            "mean_all": st["mean"],
            "positive_fraction":
                st["positive_fraction"],
            "robust_score": robust_score,
        })

    candidates.sort(
        key=lambda x: (
            x["robust_score"],
            x["median_all"],
            x["count"],
        ),
        reverse=True,
    )

    # ------------------------------------------------------------
    # Coverage matching.
    #
    # Add token TYPES in robust-score order until their occurrence
    # coverage in this uniform discovery sample reaches the manual
    # anchor occurrence coverage.
    # ------------------------------------------------------------

    selected_ids = set()
    selected_occ = 0

    for c in candidates:
        selected_ids.add(
            c["token_id"]
        )

        selected_occ += c["count"]

        current_coverage = (
            selected_occ / len(rows)
        )

        if current_coverage >= manual_coverage:
            break

    auto_coverage = (
        selected_occ / len(rows)
    )

    selected = [
        c
        for c in candidates
        if c["token_id"] in selected_ids
    ]

    selected_manual = sum(
        c["manual_anchor"]
        for c in selected
    )

    selected_nonmanual = (
        len(selected)
        - selected_manual
    )

    # ------------------------------------------------------------
    # Write all stable candidates.
    # ------------------------------------------------------------

    prefix = Path(
        args.output_prefix
    )

    candidate_path = Path(
        str(prefix)
        + "_candidates.tsv"
    )

    selected_path = Path(
        str(prefix)
        + "_selected.tsv"
    )

    vocab_path = Path(
        str(prefix)
        + "_selected_token_ids.json"
    )

    fields = [
        "rank",
        "token_id",
        "token_piece",
        "manual_anchor",
        "count",
        "count_a",
        "count_b",
        "median_all",
        "median_a",
        "median_b",
        "mean_all",
        "positive_fraction",
        "robust_score",
    ]

    with candidate_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=fields,
            delimiter="\t",
        )

        w.writeheader()

        for rank, c in enumerate(
            candidates,
            start=1,
        ):
            row = dict(c)
            row["rank"] = rank
            w.writerow(row)

    with selected_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=fields,
            delimiter="\t",
        )

        w.writeheader()

        for rank, c in enumerate(
            selected,
            start=1,
        ):
            row = dict(c)
            row["rank"] = rank
            w.writerow(row)

    with vocab_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "source":
                    args.input,
                "score":
                    "min(splitA_median_delta_h16, splitB_median_delta_h16)",
                "min_total_count":
                    args.min_total_count,
                "min_split_count":
                    args.min_split_count,
                "min_positive_fraction":
                    args.min_positive_fraction,
                "manual_occurrence_coverage":
                    manual_coverage,
                "automatic_occurrence_coverage":
                    auto_coverage,
                "num_selected_token_types":
                    len(selected),
                "num_selected_manual_types":
                    selected_manual,
                "num_selected_nonmanual_types":
                    selected_nonmanual,
                "token_ids": [
                    c["token_id"]
                    for c in selected
                ],
            },
            f,
            indent=2,
        )

    # ------------------------------------------------------------
    # Console summary.
    # ------------------------------------------------------------

    print("input rows:", len(rows))
    print(
        "split A rows:",
        len(split_a),
    )
    print(
        "split B rows:",
        len(split_b),
    )
    print()

    print(
        "manual occurrence coverage:",
        f"{manual_coverage:.4f}",
    )

    print(
        "stable candidate types:",
        len(candidates),
    )

    print(
        "selected automatic types:",
        len(selected),
    )

    print(
        "automatic occurrence coverage:",
        f"{auto_coverage:.4f}",
    )

    print(
        "selected manual token types:",
        selected_manual,
    )

    print(
        "selected NEW nonmanual token types:",
        selected_nonmanual,
    )

    print()
    print(
        "===== TOP 50 STABLE CANDIDATES ====="
    )

    for rank, c in enumerate(
        candidates[:50],
        start=1,
    ):
        print(
            f"{rank:3d}",
            f"score={c['robust_score']:.4f}",
            f"med={c['median_all']:.4f}",
            f"A={c['median_a']:.4f}",
            f"B={c['median_b']:.4f}",
            f"n={c['count']:4d}",
            f"M={c['manual_anchor']}",
            c["token_piece"],
        )

    print()
    print(
        "===== SELECTED AUTOMATIC VOCAB ====="
    )

    for rank, c in enumerate(
        selected,
        start=1,
    ):
        print(
            f"{rank:3d}",
            f"score={c['robust_score']:.4f}",
            f"n={c['count']:4d}",
            f"M={c['manual_anchor']}",
            c["token_piece"],
        )

    print()
    print("candidate file:", candidate_path)
    print("selected file:", selected_path)
    print("vocab json:", vocab_path)


if __name__ == "__main__":
    main()
