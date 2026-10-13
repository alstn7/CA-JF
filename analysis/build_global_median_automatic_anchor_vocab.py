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
        default="analysis/automatic_anchor_global_median",
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

    return p.parse_args()


def stable_half(data_id):
    h = hashlib.blake2b(
        data_id.encode("utf-8"),
        digest_size=8,
        person=b"CAJFauto",
    ).digest()

    return (
        int.from_bytes(
            h,
            byteorder="little",
            signed=False,
        )
        % 2
    )


def summarize_by_token(rows):
    by_token = defaultdict(list)

    for r in rows:
        by_token[r["target_id"]].append(r)

    stats = {}

    for tid, rs in by_token.items():
        vals = [
            r["delta"]
            for r in rs
        ]

        stats[tid] = {
            "count": len(vals),
            "median": statistics.median(vals),
            "mean": statistics.mean(vals),
            "positive_fraction":
                sum(v > 0 for v in vals) / len(vals),
            "token_piece":
                rs[0]["target_piece"],
            "manual_anchor":
                rs[0]["manual_anchor"],
        }

    return stats


def main():
    args = parse_args()

    rows = []

    with open(
        args.input,
        encoding="utf-8",
    ) as f:
        reader = csv.DictReader(
            f,
            delimiter="\t",
        )

        for r in reader:
            rows.append({
                "data_id":
                    r["data_id"],
                "target_id":
                    int(r["target_id"]),
                "target_piece":
                    r["target_piece"],
                "manual_anchor":
                    int(r["manual_anchor"]),
                "delta":
                    float(r["delta_nll_h16"]),
            })

    split_a = [
        r for r in rows
        if stable_half(r["data_id"]) == 0
    ]

    split_b = [
        r for r in rows
        if stable_half(r["data_id"]) == 1
    ]

    # ------------------------------------------------------------
    # Data-derived thresholds.
    # ------------------------------------------------------------

    global_median_all = statistics.median(
        r["delta"]
        for r in rows
    )

    global_median_a = statistics.median(
        r["delta"]
        for r in split_a
    )

    global_median_b = statistics.median(
        r["delta"]
        for r in split_b
    )

    all_stats = summarize_by_token(
        rows
    )

    a_stats = summarize_by_token(
        split_a
    )

    b_stats = summarize_by_token(
        split_b
    )

    selected = []
    rejected_support = 0
    rejected_a = 0
    rejected_b = 0

    for tid, st in all_stats.items():
        if st["count"] < args.min_total_count:
            rejected_support += 1
            continue

        if (
            tid not in a_stats
            or tid not in b_stats
        ):
            rejected_support += 1
            continue

        sa = a_stats[tid]
        sb = b_stats[tid]

        if (
            sa["count"] < args.min_split_count
            or sb["count"] < args.min_split_count
        ):
            rejected_support += 1
            continue

        if sa["median"] <= global_median_a:
            rejected_a += 1
            continue

        if sb["median"] <= global_median_b:
            rejected_b += 1
            continue

        robust_margin = min(
            sa["median"] - global_median_a,
            sb["median"] - global_median_b,
        )

        selected.append({
            "token_id":
                tid,
            "token_piece":
                st["token_piece"],
            "manual_anchor":
                st["manual_anchor"],

            "count":
                st["count"],
            "count_a":
                sa["count"],
            "count_b":
                sb["count"],

            "median_all":
                st["median"],
            "median_a":
                sa["median"],
            "median_b":
                sb["median"],

            "mean_all":
                st["mean"],

            "positive_fraction":
                st["positive_fraction"],

            "margin_a":
                sa["median"]
                - global_median_a,

            "margin_b":
                sb["median"]
                - global_median_b,

            "robust_margin":
                robust_margin,
        })

    selected.sort(
        key=lambda x: (
            x["robust_margin"],
            x["median_all"],
            x["count"],
        ),
        reverse=True,
    )

    selected_ids = {
        x["token_id"]
        for x in selected
    }

    # Realized occurrence coverage in the uniform discovery sample.
    selected_occ = sum(
        1
        for r in rows
        if r["target_id"] in selected_ids
    )

    automatic_coverage = (
        selected_occ / len(rows)
    )

    manual_occ = sum(
        r["manual_anchor"]
        for r in rows
    )

    manual_coverage = (
        manual_occ / len(rows)
    )

    both_occ = sum(
        1
        for r in rows
        if (
            r["manual_anchor"]
            and r["target_id"] in selected_ids
        )
    )

    auto_only_occ = sum(
        1
        for r in rows
        if (
            not r["manual_anchor"]
            and r["target_id"] in selected_ids
        )
    )

    selected_manual_types = sum(
        x["manual_anchor"]
        for x in selected
    )

    selected_nonmanual_types = (
        len(selected)
        - selected_manual_types
    )

    prefix = Path(
        args.output_prefix
    )

    tsv_path = Path(
        str(prefix)
        + "_selected.tsv"
    )

    json_path = Path(
        str(prefix)
        + "_token_ids.json"
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
        "margin_a",
        "margin_b",
        "robust_margin",
    ]

    with tsv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            delimiter="\t",
        )

        writer.writeheader()

        for rank, row in enumerate(
            selected,
            start=1,
        ):
            out = dict(row)
            out["rank"] = rank
            writer.writerow(out)

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "selection_rule": {
                    "min_total_count":
                        args.min_total_count,
                    "min_split_count":
                        args.min_split_count,
                    "split_a_global_median":
                        global_median_a,
                    "split_b_global_median":
                        global_median_b,
                    "criterion":
                        (
                            "median_A(token) > global_median_A "
                            "AND median_B(token) > global_median_B"
                        ),
                },

                "num_selected_token_types":
                    len(selected),

                "num_selected_manual_types":
                    selected_manual_types,

                "num_selected_nonmanual_types":
                    selected_nonmanual_types,

                "manual_occurrence_coverage":
                    manual_coverage,

                "automatic_occurrence_coverage":
                    automatic_coverage,

                "manual_and_auto_occurrences":
                    both_occ,

                "auto_nonmanual_occurrences":
                    auto_only_occ,

                "token_ids": [
                    x["token_id"]
                    for x in selected
                ],
            },
            f,
            indent=2,
        )

    print(
        "input rows:",
        len(rows),
    )

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
        "global median ALL:",
        f"{global_median_all:.6f}",
    )

    print(
        "global median A:",
        f"{global_median_a:.6f}",
    )

    print(
        "global median B:",
        f"{global_median_b:.6f}",
    )

    print()

    print(
        "manual occurrence coverage:",
        f"{manual_coverage:.4f}",
    )

    print(
        "automatic occurrence coverage:",
        f"{automatic_coverage:.4f}",
    )

    print(
        "selected token types:",
        len(selected),
    )

    print(
        "selected manual types:",
        selected_manual_types,
    )

    print(
        "selected NEW nonmanual types:",
        selected_nonmanual_types,
    )

    print()

    print(
        "manual+auto overlapping occurrences:",
        both_occ,
    )

    print(
        "auto-selected nonmanual occurrences:",
        auto_only_occ,
    )

    print()

    print(
        "rejected support:",
        rejected_support,
    )

    print(
        "rejected split-A threshold:",
        rejected_a,
    )

    print(
        "rejected split-B threshold:",
        rejected_b,
    )

    print()
    print(
        "===== SELECTED GLOBAL-MEDIAN AUTO ANCHORS ====="
    )

    for rank, row in enumerate(
        selected,
        start=1,
    ):
        print(
            f"{rank:3d}",
            f"margin={row['robust_margin']:.4f}",
            f"med={row['median_all']:.4f}",
            f"A={row['median_a']:.4f}",
            f"B={row['median_b']:.4f}",
            f"n={row['count']:4d}",
            f"M={row['manual_anchor']}",
            row["token_piece"],
        )

    print()
    print(
        "selected TSV:",
        tsv_path,
    )

    print(
        "token IDs JSON:",
        json_path,
    )


if __name__ == "__main__":
    main()
