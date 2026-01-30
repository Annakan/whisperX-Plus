from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd

from evaluator.models import ComparisonResult


def write_result_csv(result: ComparisonResult, output_dir: Path) -> Path:
    """
    Write a single comparison result to CSV.

    Args:
        result: ComparisonResult to write
        output_dir: Directory to write result.csv

    Returns:
        Path to the created result.csv file
    """
    output_path = output_dir / "result.csv"

    row: dict[str, Any] = {
        "source_name": result.source_name,
        "variation_name": result.variation_name,
    }

    for key, value in sorted(result.params.to_dict().items()):
        row[key] = value

    row.update({
        "levenshtein_va": result.levenshtein_va,
        "levenshtein_vb": result.levenshtein_vb,
        "levenshtein_vc": result.levenshtein_vc,
        "levenshtein_vd": result.levenshtein_vd,
        "wer_va": result.wer_va,
        "wer_vb": result.wer_vb,
        "wer_vc": result.wer_vc,
        "wer_vd": result.wer_vd,
        "cer_va": result.cer_va,
        "cer_vb": result.cer_vb,
        "cer_vc": result.cer_vc,
        "cer_vd": result.cer_vd,
        "runtime_seconds": result.runtime_seconds,
        "error": result.error,
    })

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)

    return output_path


def aggregate_results(run_dir: Path) -> Path:
    """
    Aggregate all result.csv files into a single CSV.

    Recursively finds all result.csv files in run_dir and combines them.

    Args:
        run_dir: Root run directory containing subdirectories with results

    Returns:
        Path to aggregated_results.csv
    """
    result_files = list(run_dir.rglob("result.csv"))

    if not result_files:
        raise ValueError(f"No result.csv files found in {run_dir}")

    dfs: list[pd.DataFrame] = []
    for result_file in result_files:
        try:
            df = pd.read_csv(result_file)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not read {result_file}: {e}")

    if not dfs:
        raise ValueError("No valid result.csv files could be read")

    combined = pd.concat(dfs, ignore_index=True)

    sort_cols = ["source_name"]
    for col in ["model", "beam_size", "patience", "preprocess", "compute_type"]:
        if col in combined.columns:
            sort_cols.append(col)

    combined = combined.sort_values(
        by=sort_cols,
        ignore_index=True,
    )

    output_path = run_dir / "aggregated_results.csv"
    combined.to_csv(output_path, index=False)

    return output_path


def load_aggregated_results(run_dir: Path) -> pd.DataFrame:
    """
    Load the aggregated results CSV into a DataFrame.

    Args:
        run_dir: Run directory containing aggregated_results.csv

    Returns:
        DataFrame with all results
    """
    path = run_dir / "aggregated_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Aggregated results not found: {path}")

    return pd.read_csv(path)
