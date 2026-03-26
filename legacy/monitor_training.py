#!/usr/bin/env python3
"""Simple monitor for benchmark outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _find_latest_results(root: Path) -> Path | None:
    candidates = sorted(root.glob("**/benchmark_all_runs.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor benchmark progress/results")
    parser.add_argument("--root", type=Path, default=Path("experiments"))
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    latest = _find_latest_results(args.root)
    if latest is None:
        print(f"No benchmark results found under {args.root}")
        return

    df = pd.read_csv(latest)
    print(f"Latest result file: {latest}")
    print(f"Total runs: {len(df)}")
    if "status" in df.columns:
        print(df["status"].value_counts(dropna=False).to_string())

    ok = df[df.get("status", "failed") == "ok"].copy() if "status" in df.columns else df.copy()
    if ok.empty:
        print("No successful runs yet.")
        return

    metric = "test_map50_95" if "test_map50_95" in ok.columns else "val_map50_95"
    board = ok.sort_values(metric, ascending=False).head(args.top_k)
    keep_cols = [
        c
        for c in [
            "model_id",
            "seed",
            "val_map50_95",
            "test_map50_95",
            "train_eval_minutes",
            "wandb_run_url",
            "best_model",
        ]
        if c in board.columns
    ]
    print("\nTop runs:")
    print(board[keep_cols].to_string(index=False))


if __name__ == "__main__":
    main()
