from __future__ import annotations

import csv
import os


FIELDNAMES = [
    "model", "benchmark", "ckpt_type", "pooling", "large_head", "fold",
    "block_idx", "block_name", "n_blocks",
    "probe_dim",
    "bacc", "f1_weighted", "f1_macro",
    "train_bacc", "train_f1_weighted", "train_f1_macro",
    "best_epoch",
]


def open_writer(output_csv: str):
    """Open `output_csv` for append. Writes header iff the file is new/empty."""
    write_header = (not os.path.exists(output_csv)
                    or os.path.getsize(output_csv) == 0)
    f = open(output_csv, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDNAMES)
    if write_header:
        w.writeheader()
    return w, f


def write_rows(writer, csv_file, rows: list[dict]) -> None:
    writer.writerows(rows)
    csv_file.flush()


def build_row(*, model, benchmark, ckpt_type, pooling, large_head, fold,
              block_idx, block_name, n_blocks, metrics: dict) -> dict:
    return {
        "model": model, "benchmark": benchmark,
        "ckpt_type": ckpt_type, "pooling": pooling,
        "large_head": large_head, "fold": fold,
        "block_idx": block_idx, "block_name": block_name,
        "n_blocks": n_blocks,
        **{k: round(v, 6) for k, v in metrics.items()},
    }


def load_completed_keys(probing_dir: str) -> set:
    """Scan every CSV under `probing_dir` and collect
    (model, benchmark, ckpt_type, pooling, large_head, fold) tuples already
    present. Used to skip work on resume.
    """
    completed = set()
    if not os.path.isdir(probing_dir):
        return completed
    for dirpath, _, filenames in os.walk(probing_dir):
        for fname in filenames:
            if not fname.endswith(".csv"):
                continue
            fpath = os.path.join(dirpath, fname)
            try:
                with open(fpath, newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            completed.add((
                                row["model"], row["benchmark"],
                                row["ckpt_type"], row.get("pooling", "mean"),
                                row.get("large_head", "False") == "True",
                                int(row["fold"]),
                            ))
                        except (KeyError, ValueError):
                            pass
            except Exception:
                pass
    return completed
