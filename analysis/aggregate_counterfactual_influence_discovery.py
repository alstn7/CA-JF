#!/usr/bin/env python3

import argparse
import csv
import statistics
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--input",
        default="analysis/code_counterfactual_influence_discovery_2000.tsv",
    )

    p.add_argument(
        "--output",
        default="analysis/code_counterfactual_influence_token_aggregate.tsv",
    )

    p.add_argument(
        "--min-count",
        type=int,
        default=20,
    )

    return p.parse_args()


def percentile(xs, q):
    xs = sorted(xs)

    if not xs:
        return float("nan")

    if len(xs) == 1:
        return xs[0]

    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo

    return (
        xs[lo] * (1.0 - frac)
        + xs[hi] * frac
    )


def main():
    args = parse_args()

    by_token = defaultdict(list)
    pieces = {}
    manual_flags = defaultdict(list)

    with open(
        args.input,
        encoding="utf-8",
    ) as f:
        reader = csv.DictReader(
            f,
            delimiter="\t",
        )

        for r in reader:
            tid = int(r["target_id"])
            delta = float(r["delta_nll_h16"])
            manual = int(r["manual_anchor"])

            by_token[tid].append(delta)
            pieces[tid] = r["target_piece"]
            manual_flags[tid].append(manual)

    out_rows = []

    for tid, vals in by_token.items():
        count = len(vals)

        if count < args.min_count:
            continue

        manual_rate = (
            sum(manual_flags[tid])
            / len(manual_flags[tid])
        )

        row = {
            "token_id": tid,
            "token_piece": pieces[tid],
            "count": count,
            "manual_anchor_rate": manual_rate,
            "manual_anchor_token":
                int(manual_rate >= 0.5),
            "mean_delta": statistics.mean(vals),
            "median_delta": statistics.median(vals),
            "p75_delta": percentile(vals, 0.75),
            "p90_delta": percentile(vals, 0.90),
            "positive_fraction":
                sum(x > 0 for x in vals) / count,
        }

        out_rows.append(row)

    out_rows.sort(
        key=lambda x: (
            x["median_delta"],
            x["count"],
        ),
        reverse=True,
    )

    fieldnames = [
        "rank",
        "token_id",
        "token_piece",
        "count",
        "manual_anchor_token",
        "manual_anchor_rate",
        "mean_delta",
        "median_delta",
        "p75_delta",
        "p90_delta",
        "positive_fraction",
    ]

    with open(
        args.output,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            delimiter="\t",
        )

        writer.writeheader()

        for rank, row in enumerate(
            out_rows,
            start=1,
        ):
            writer.writerow({
                "rank": rank,
                "token_id": row["token_id"],
                "token_piece": row["token_piece"],
                "count": row["count"],
                "manual_anchor_token":
                    row["manual_anchor_token"],
                "manual_anchor_rate":
                    f"{row['manual_anchor_rate']:.6f}",
                "mean_delta":
                    f"{row['mean_delta']:.8f}",
                "median_delta":
                    f"{row['median_delta']:.8f}",
                "p75_delta":
                    f"{row['p75_delta']:.8f}",
                "p90_delta":
                    f"{row['p90_delta']:.8f}",
                "positive_fraction":
                    f"{row['positive_fraction']:.6f}",
            })

    print("input:", args.input)
    print("eligible token types:", len(out_rows))
    print("min_count:", args.min_count)
    print("output:", args.output)

    print()
    print("===== TOP 50 BY MEDIAN DELTA =====")

    for row in out_rows[:50]:
        print(
            f"{row['median_delta']:8.4f}",
            f"n={row['count']:4d}",
            f"M={row['manual_anchor_token']}",
            f"mean={row['mean_delta']:8.4f}",
            f"p75={row['p75_delta']:8.4f}",
            f"pos={row['positive_fraction']:.3f}",
            row["token_piece"],
        )

    print()
    print("===== TOP NON-MANUAL CANDIDATES =====")

    nonmanual = [
        r
        for r in out_rows
        if r["manual_anchor_token"] == 0
    ]

    for row in nonmanual[:50]:
        print(
            f"{row['median_delta']:8.4f}",
            f"n={row['count']:4d}",
            f"mean={row['mean_delta']:8.4f}",
            f"p75={row['p75_delta']:8.4f}",
            f"pos={row['positive_fraction']:.3f}",
            row["token_piece"],
        )


if __name__ == "__main__":
    main()
