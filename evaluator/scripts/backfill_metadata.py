#!/usr/bin/env python3
"""
Backfill transcriber_run.metadata files from an existing aggregated_results CSV.

One-time fix script to populate metadata files for runs that were executed
before the metadata feature was added.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def backfill_metadata(
    csv_path: Path,
    run_dir: Path,
    force: bool = False,
) -> tuple[int, int, list[str]]:
    """
    Read aggregated results CSV and create transcriber_run.metadata files.

    Args:
        csv_path: Path to the aggregated_results CSV file
        run_dir: Base run directory containing source_name/variation_name subdirs
        force: If True, overwrite existing metadata files

    Returns:
        Tuple of (created_count, skipped_count, errors)
    """
    created = 0
    skipped = 0
    errors: list[str] = []

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            source_name = row.get("source_name", "")
            variation_name = row.get("variation_name", "")

            if not source_name or not variation_name:
                errors.append(f"Row missing source_name or variation_name: {row}")
                continue

            variation_dir = run_dir / source_name / variation_name
            metadata_path = variation_dir / "transcriber_run.metadata"

            if not variation_dir.exists():
                errors.append(f"Directory not found: {variation_dir}")
                continue

            if metadata_path.exists() and not force:
                skipped += 1
                continue

            runtime_str = row.get("runtime_seconds", "")
            error_str = row.get("error", "")

            try:
                runtime_seconds = float(runtime_str) if runtime_str else None
            except ValueError:
                runtime_seconds = None

            timestamp = None
            audio_file = None
            params_path = variation_dir / "params.json"
            if params_path.exists():
                import datetime
                mtime = params_path.stat().st_mtime
                timestamp = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%dT%H:%M:%S")
                try:
                    with params_path.open(encoding="utf-8") as pf:
                        params_data = json.load(pf)
                    audio_file = params_data.get("audio_file")
                except Exception:
                    pass

            metadata = {
                "runtime_seconds": runtime_seconds,
                "timestamp": timestamp,
                "audio_file": audio_file,
                "error": error_str if error_str else None,
            }

            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            created += 1

    return created, skipped, errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill transcriber_run.metadata from aggregated results CSV"
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to aggregated_results CSV file",
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Base run directory containing source_name/variation_name subdirs",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Overwrite existing metadata files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without writing files",
    )

    args = parser.parse_args()

    if not args.csv_path.exists():
        print(f"Error: CSV file not found: {args.csv_path}")
        return

    if not args.run_dir.exists():
        print(f"Error: Run directory not found: {args.run_dir}")
        return

    if args.dry_run:
        print(f"DRY RUN - no files will be written")
        print(f"CSV: {args.csv_path}")
        print(f"Run dir: {args.run_dir}")
        print(f"Force: {args.force}")

        with args.csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            count = sum(1 for _ in reader)
        print(f"Rows in CSV: {count}")
        return

    created, skipped, errors = backfill_metadata(
        args.csv_path,
        args.run_dir,
        args.force,
    )

    print(f"Created: {created}")
    print(f"Skipped (already exists): {skipped}")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for err in errors[:20]:
            print(f"  - {err}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")


if __name__ == "__main__":
    main()
