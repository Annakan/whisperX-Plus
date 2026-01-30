from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import pytest

from evaluator.aggregator import write_result_csv
from evaluator.models import ComparisonResult, TranscriptionParams


class TestWriteResultCSV:
    def test_dynamic_params_become_columns(self):
        """Verify that all params in TranscriptionParams become CSV columns."""
        params = TranscriptionParams(
            params={
                "model": "large-v3",
                "beam_size": 7,
                "patience": 2.0,
                "preprocess": 1,
                "compute_type": "float16",
            }
        )

        result = ComparisonResult(
            source_name="test_source",
            variation_name=params.to_dirname(),
            params=params,
            levenshtein_va=0.95,
            levenshtein_vb=0.90,
            levenshtein_vc=0.85,
            levenshtein_vd=0.80,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            csv_path = write_result_csv(result, output_dir)

            with csv_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 1
            row = rows[0]

            assert row["source_name"] == "test_source"
            assert row["model"] == "large-v3"
            assert row["beam_size"] == "7"
            assert row["patience"] == "2.0"
            assert row["preprocess"] == "1"
            assert row["compute_type"] == "float16"
            assert row["levenshtein_va"] == "0.95"

    def test_params_columns_sorted_alphabetically(self):
        """Verify params columns are in alphabetical order."""
        params = TranscriptionParams(
            params={
                "z_param": "last",
                "a_param": "first",
                "model": "tiny",
            }
        )

        result = ComparisonResult(
            source_name="test",
            variation_name=params.to_dirname(),
            params=params,
            levenshtein_va=0.9,
            levenshtein_vb=0.9,
            levenshtein_vc=0.9,
            levenshtein_vd=0.9,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            csv_path = write_result_csv(result, output_dir)

            with csv_path.open("r", encoding="utf-8") as f:
                reader = csv.reader(f)
                headers = next(reader)

            param_start = headers.index("a_param")
            param_end = headers.index("z_param")
            model_idx = headers.index("model")

            assert param_start < model_idx < param_end
