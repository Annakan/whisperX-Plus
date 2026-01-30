"""WhisperX Evaluator - Benchmark transcription quality across parameter variations."""

from evaluator.aggregator import aggregate_results, write_result_csv
from evaluator.comparator import compare, compare_all_variants
from evaluator.models import (
    ComparisonResult,
    RunPlan,
    Segment,
    SourceFiles,
    TranscriptionParams,
)
from evaluator.parser import normalize_speakers, parse_source, parse_whisperx_json
from evaluator.prepare import (
    VARIANTS,
    prepare,
    prepare_all_variants,
    prepare_va,
    prepare_vb,
    prepare_vc,
    prepare_vd,
)
from evaluator.runner import (
    check_variation_complete,
    copy_source_to_run_dir,
    discover_source_files,
    generate_command_file,
    generate_parameter_grid,
    run_transcription,
)

__all__ = [
    "Segment",
    "RunPlan",
    "TranscriptionParams",
    "ComparisonResult",
    "SourceFiles",
    "parse_source",
    "parse_whisperx_json",
    "normalize_speakers",
    "prepare",
    "prepare_all_variants",
    "prepare_va",
    "prepare_vb",
    "prepare_vc",
    "prepare_vd",
    "VARIANTS",
    "compare",
    "compare_all_variants",
    "discover_source_files",
    "copy_source_to_run_dir",
    "generate_command_file",
    "generate_parameter_grid",
    "run_transcription",
    "check_variation_complete",
    "write_result_csv",
    "aggregate_results",
]
