from __future__ import annotations

import contextlib
import itertools
import json
import random
import shutil
import sys
import time
from pathlib import Path
from typing import IO, Any, Iterator

from evaluator.models import RunPlan, SourceFiles, TranscriptionParams


def discover_source_files(source_dir: Path, subdir_name: str) -> SourceFiles:
    """
    Discover audio and reference files in a source subdirectory.

    Args:
        source_dir: Root source data directory
        subdir_name: Name of subdirectory to search

    Returns:
        SourceFiles with paths to audio and reference files

    Raises:
        FileNotFoundError: If required files not found
    """
    subdir = source_dir / subdir_name

    if not subdir.exists():
        raise FileNotFoundError(f"Subdirectory not found: {subdir}")

    audio_files = list(subdir.glob("audio_*.*"))
    if not audio_files:
        raise FileNotFoundError(f"No audio file (audio_*.*) found in {subdir}")
    if len(audio_files) > 1:
        raise ValueError(f"Multiple audio files found in {subdir}: {audio_files}")
    audio_path = audio_files[0]

    ref_files = list(subdir.glob("*_REF.txt"))
    if not ref_files:
        raise FileNotFoundError(f"No reference file (*_REF.txt) found in {subdir}")
    if len(ref_files) > 1:
        raise ValueError(f"Multiple reference files found in {subdir}: {ref_files}")
    reference_path = ref_files[0]

    return SourceFiles(
        audio_path=audio_path,
        reference_path=reference_path,
        subdir_name=subdir_name,
    )


def copy_source_to_run_dir(source_files: SourceFiles, run_dir: Path) -> Path:
    """
    Copy source files to the run directory.

    Args:
        source_files: Source audio and reference paths
        run_dir: Target run directory

    Returns:
        Path to the created subdirectory in run_dir
    """
    target_dir = run_dir / source_files.subdir_name
    target_dir.mkdir(parents=True, exist_ok=True)

    target_audio = target_dir / source_files.audio_path.name
    target_ref = target_dir / source_files.reference_path.name

    if not target_audio.exists():
        shutil.copy2(source_files.audio_path, target_audio)

    if not target_ref.exists():
        shutil.copy2(source_files.reference_path, target_ref)

    return target_dir


def generate_parameter_grid(
    run_plan: RunPlan,
    sample: int | None = None,
) -> list[TranscriptionParams]:
    """
    Generate all parameter combinations from varying_args.

    Dynamically handles any parameters defined in varying_args.

    Args:
        run_plan: RunPlan with varying_args configuration
        sample: If set, randomly sample N combinations

    Returns:
        List of TranscriptionParams for each combination
    """
    varying = run_plan.varying_args

    if not varying:
        return [TranscriptionParams(params={})]

    keys = sorted(varying.keys())
    value_lists = [varying[k] for k in keys]

    combinations = list(itertools.product(*value_lists))

    if sample and sample < len(combinations):
        combinations = random.sample(combinations, sample)

    params_list: list[TranscriptionParams] = []
    for combo in combinations:
        params_dict = dict(zip(keys, combo))
        params_list.append(TranscriptionParams(params=params_dict))

    return params_list


def generate_command_file(
    audio_path: Path,
    output_dir: Path,
    merged_args: dict[str, Any],
) -> Path:
    """
    Generate a shell script with the exact command line to reproduce the transcription.

    Args:
        audio_path: Path to audio file
        output_dir: Directory to save the command file
        merged_args: All merged arguments for transcription

    Returns:
        Path to the generated command.sh file
    """
    cmd_parts = ["whisperx", "transcribe"]

    for key, value in sorted(merged_args.items()):
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                cmd_parts.append(f"--{key}")
        elif isinstance(value, (list, tuple)):
            for v in value:
                cmd_parts.append(f"--{key}")
                cmd_parts.append(str(v))
        else:
            cmd_parts.append(f"--{key}")
            cmd_parts.append(str(value))

    cmd_parts.append(str(audio_path))

    cmd_line = " \\\n    ".join(cmd_parts)
    script_content = f"#!/bin/bash\n# Command to reproduce this transcription\n{cmd_line}\n"

    cmd_file = output_dir / "command.sh"
    cmd_file.write_text(script_content, encoding="utf-8")
    cmd_file.chmod(0o755)

    return cmd_file


def _run_with_output_capture(
    func: Any,
    params: Any,
    audio_files: list[str],
    output_dir: Path,
) -> None:
    """
    Run a function while capturing stdout/stderr to a log file.

    Args:
        func: The transcription function to call
        params: TranscribeParams to pass
        audio_files: List of audio file paths
        output_dir: Directory to save the out.log file
    """
    log_file = output_dir / "out.log"

    with log_file.open("a", encoding="utf-8") as log:
        log.write(f"\n{'='*60}\n")
        log.write(f"Transcription started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"{'='*60}\n\n")
        log.flush()

        old_stdout = sys.stdout
        old_stderr = sys.stderr

        try:
            sys.stdout = log
            sys.stderr = log
            func(params, audio_files)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def run_transcription(
    audio_path: Path,
    output_dir: Path,
    params: TranscriptionParams,
    base_args: dict[str, Any],
) -> tuple[Path | None, float, str | None]:
    """
    Run WhisperX transcription with given parameters.

    Args:
        audio_path: Path to audio file
        output_dir: Directory to save output (variation subdirectory)
        params: Transcription parameters
        base_args: Base arguments from runplan

    Returns:
        Tuple of (transcript_path, runtime_seconds, error_message)
        transcript_path is None if failed
    """
    from whisperx.api_models import TranscribeParams
    from whisperx.transcribe import run_transcription as whisperx_transcribe

    output_dir.mkdir(parents=True, exist_ok=True)

    merged_args = {
        **base_args,
        **params.to_dict(),
        "output_dir": str(output_dir),
        "output_format": "json",
        "verbose": False,
    }

    generate_command_file(audio_path, output_dir, merged_args)

    params_file = output_dir / "params.json"
    params_file.write_text(
        json.dumps(
            {
                "audio_file": audio_path.name,
                "params": params.to_dict(),
                "base_args": base_args,
            },
            indent=2,
        )
    )

    def write_run_metadata(runtime: float, error: str | None) -> None:
        """Write transcriber run metadata to file."""
        metadata = {
            "runtime_seconds": runtime,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "audio_file": audio_path.name,
            "error": error,
        }
        metadata_file = output_dir / "transcriber_run.metadata"
        metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    try:
        transcribe_params = TranscribeParams(**merged_args)
        start_time = time.time()
        _run_with_output_capture(whisperx_transcribe, transcribe_params, [str(audio_path)], output_dir)
        runtime = time.time() - start_time

        json_files = list(output_dir.glob("*.json"))
        json_files = [f for f in json_files if f.name not in ("params.json", "transcriber_run.metadata")]

        if json_files:
            write_run_metadata(runtime, None)
            return json_files[0], runtime, None
        else:
            error_msg = "No transcript JSON output generated"
            write_run_metadata(0.0, error_msg)
            return None, 0.0, error_msg

    except Exception as e:
        error_msg = str(e)
        write_run_metadata(0.0, error_msg)
        return None, 0.0, error_msg


def check_variation_complete(variation_dir: Path) -> bool:
    """
    Check if a variation has already been successfully processed.

    A variation is complete if a transcript JSON file exists (not params.json).
    This ensures we don't skip failed runs that only created params.json.

    Args:
        variation_dir: Path to variation subdirectory

    Returns:
        True if successfully processed (transcript exists)
    """
    if not variation_dir.exists():
        return False

    json_files = list(variation_dir.glob("*.json"))
    transcript_files = [f for f in json_files if f.name != "params.json"]
    return len(transcript_files) > 0
