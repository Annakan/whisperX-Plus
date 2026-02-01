from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import typer
import yaml
from tqdm import tqdm

from evaluator.aggregator import aggregate_results, write_result_csv
from evaluator.comparator import compare_all_variants
from evaluator.models import ComparisonResult, RunPlan, TranscriptionParams
from evaluator.parser import parse_source, parse_whisperx_json
from evaluator.prepare import prepare_all_variants
from evaluator.runner import (
    check_variation_complete,
    copy_source_to_run_dir,
    discover_source_files,
    generate_parameter_grid,
    run_transcription,
)

def _cleanup_gpu_memory() -> None:
    """Release GPU memory after each transcription task."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


app = typer.Typer(
    name="whisperx-eval",
    help="WhisperX transcription quality evaluator",
    add_completion=True,
)


@app.command("run")
def run_evaluation(
    source_dir: Path = typer.Option(
        ...,
        "--source-dir",
        "-s",
        help="Directory containing source data (audio + reference transcripts)",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    run_dir: Path = typer.Option(
        ...,
        "--run-dir",
        "-r",
        help="Directory containing runplan.yaml and for storing outputs",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    stop_on_fail: bool = typer.Option(
        False,
        "--stop-on-fail",
        help="Abort on transcription error instead of skipping",
    ),
    sample: Optional[int] = typer.Option(
        None,
        "--sample",
        "-n",
        help="Run only N random parameter combinations",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate configuration and show plan without executing",
    ),
    post_task_delay: int = typer.Option(
        2000,
        "--post-task-delay",
        help="Delay in milliseconds between transcription tasks to allow GPU memory cleanup (default: 2000ms)",
    ),
) -> None:
    """
    Run WhisperX evaluation with parameter variations.

    Reads runplan.yaml from run_dir, processes each source in to_process,
    runs transcriptions with all parameter combinations, and compares results.
    """
    runplan_path = run_dir / "runplan.yaml"
    if not runplan_path.exists():
        typer.secho(f"Error: runplan.yaml not found in {run_dir}", fg=typer.colors.RED)
        raise typer.Exit(1)

    with runplan_path.open() as f:
        runplan_data = yaml.safe_load(f)

    run_plan = RunPlan(**runplan_data)
    params_grid = generate_parameter_grid(run_plan, sample=sample)

    typer.echo(f"Source directory: {source_dir}")
    typer.echo(f"Run directory: {run_dir}")
    typer.echo(f"Sources to process: {len(run_plan.to_process)}")
    typer.echo(f"Parameter combinations: {len(params_grid)}")
    typer.echo(
        f"Total transcriptions: {len(run_plan.to_process) * len(params_grid)}"
    )

    if dry_run:
        typer.echo("\n--- DRY RUN MODE ---")
        typer.echo("\nSources:")
        for subdir in run_plan.to_process:
            try:
                source_files = discover_source_files(source_dir, subdir)
                typer.echo(f"  ✓ {subdir}: {source_files.audio_path.name}")
            except Exception as e:
                typer.secho(f"  ✗ {subdir}: {e}", fg=typer.colors.RED)

        typer.echo("\nBase args:")
        for k, v in run_plan.base_args.items():
            typer.echo(f"  {k}: {v}")

        typer.echo("\nVarying args:")
        for k, v in run_plan.varying_args.items():
            typer.echo(f"  {k}: {v}")

        typer.echo("\nParameter combinations (first 5):")
        for p in params_grid[:5]:
            typer.echo(f"  {p.to_dirname()}")
        if len(params_grid) > 5:
            typer.echo(f"  ... and {len(params_grid) - 5} more")

        raise typer.Exit(0)

    errors: list[str] = []

    for subdir_name in tqdm(run_plan.to_process, desc="Sources", unit="source"):
        try:
            source_files = discover_source_files(source_dir, subdir_name)
        except Exception as e:
            msg = f"Failed to discover files in {subdir_name}: {e}"
            if stop_on_fail:
                typer.secho(f"Error: {msg}", fg=typer.colors.RED)
                raise typer.Exit(1)
            errors.append(msg)
            continue

        work_dir = copy_source_to_run_dir(source_files, run_dir)

        reference_segments = parse_source(work_dir / source_files.reference_path.name)
        reference_variants = prepare_all_variants(reference_segments)

        audio_path = work_dir / source_files.audio_path.name

        for params in tqdm(
            params_grid,
            desc=f"  {subdir_name}",
            unit="var",
            leave=False,
        ):
            variation_dir = work_dir / params.to_dirname()

            if check_variation_complete(variation_dir):
                continue

            transcript_path, runtime, error = run_transcription(
                audio_path=audio_path,
                output_dir=variation_dir,
                params=params,
                base_args=run_plan.base_args,
            )

            if error:
                msg = f"{subdir_name}/{params.to_dirname()}: {error}"
                typer.secho(f"✗ {msg}", fg=typer.colors.RED, err=True)
                if stop_on_fail:
                    raise typer.Exit(1)
                errors.append(msg)

                result = ComparisonResult(
                    source_name=subdir_name,
                    variation_name=params.to_dirname(),
                    params=params,
                    levenshtein_va=0.0,
                    levenshtein_vb=0.0,
                    levenshtein_vc=0.0,
                    levenshtein_vd=0.0,
                    error=error,
                )
                write_result_csv(result, variation_dir)
                continue

            generated_segments = parse_whisperx_json(transcript_path)
            generated_variants = prepare_all_variants(generated_segments)

            metrics = compare_all_variants(generated_variants, reference_variants)

            result = ComparisonResult(
                source_name=subdir_name,
                variation_name=params.to_dirname(),
                params=params,
                levenshtein_va=metrics.get("VA").levenshtein_ratio if metrics.get("VA") else 0.0,
                levenshtein_vb=metrics.get("VB").levenshtein_ratio if metrics.get("VB") else 0.0,
                levenshtein_vc=metrics.get("VC").levenshtein_ratio if metrics.get("VC") else 0.0,
                levenshtein_vd=metrics.get("VD").levenshtein_ratio if metrics.get("VD") else 0.0,
                wer_va=metrics.get("VA").wer if metrics.get("VA") else None,
                wer_vb=metrics.get("VB").wer if metrics.get("VB") else None,
                wer_vc=metrics.get("VC").wer if metrics.get("VC") else None,
                wer_vd=metrics.get("VD").wer if metrics.get("VD") else None,
                cer_va=metrics.get("VA").cer if metrics.get("VA") else None,
                cer_vb=metrics.get("VB").cer if metrics.get("VB") else None,
                cer_vc=metrics.get("VC").cer if metrics.get("VC") else None,
                cer_vd=metrics.get("VD").cer if metrics.get("VD") else None,
                runtime_seconds=runtime,
            )

            for variant_name, text in generated_variants.items():
                (variation_dir / f"{variant_name}.txt").write_text(text, encoding="utf-8")

            write_result_csv(result, variation_dir)

            _cleanup_gpu_memory()
            if post_task_delay > 0:
                time.sleep(post_task_delay / 1000.0)

    typer.echo("\nAggregating results...")
    try:
        aggregated_path = aggregate_results(run_dir)
        typer.secho(f"✓ Aggregated results: {aggregated_path}", fg=typer.colors.GREEN)
    except ValueError as e:
        typer.secho(f"Warning: {e}", fg=typer.colors.YELLOW)

    if errors:
        typer.echo(f"\n{len(errors)} errors occurred:")
        for err in errors[:10]:
            typer.secho(f"  • {err}", fg=typer.colors.YELLOW)
        if len(errors) > 10:
            typer.echo(f"  ... and {len(errors) - 10} more")

    typer.secho("\nEvaluation complete.", fg=typer.colors.GREEN)


@app.command("aggregate")
def aggregate_only(
    run_dir: Path = typer.Option(
        ...,
        "--run-dir",
        "-r",
        help="Directory containing result.csv files to aggregate",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
) -> None:
    """
    Aggregate existing result.csv files into aggregated_results.csv.

    Use this if you want to re-aggregate results without re-running transcriptions.
    """
    try:
        aggregated_path = aggregate_results(run_dir)
        typer.secho(f"✓ Aggregated results: {aggregated_path}", fg=typer.colors.GREEN)
    except ValueError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED)
        raise typer.Exit(1)


@app.command("generate-variants")
def generate_variants(
    target_dir: Path = typer.Option(
        ...,
        "--target-dir",
        "-t",
        help="Directory to process (rundir root or leaf subdir with transcript)",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
) -> None:
    """
    Regenerate V*.txt variants from existing transcript JSON files.

    Use this after changing variant calculation logic or after manual
    edits to transcript JSON files.

    If pointed at the root rundir, regenerates variants in all subdirs
    with transcripts. If pointed at a leaf subdir, only processes that directory.
    """
    from evaluator.parser import parse_whisperx_json
    from evaluator.prepare import prepare_all_variants

    def find_transcript_json(directory: Path) -> Path | None:
        """Find transcript JSON file in directory (excluding params.json)."""
        json_files = list(directory.glob("*.json"))
        transcript_files = [f for f in json_files if f.name != "params.json"]
        return transcript_files[0] if transcript_files else None

    def process_directory(directory: Path) -> bool:
        """Process a single directory, return True if variants were generated."""
        transcript_path = find_transcript_json(directory)
        if not transcript_path:
            return False

        try:
            segments = parse_whisperx_json(transcript_path)
            variants = prepare_all_variants(segments)

            for variant_name, text in variants.items():
                variant_file = directory / f"{variant_name}.txt"
                variant_file.write_text(text, encoding="utf-8")

            return True
        except Exception as e:
            typer.secho(f"  ✗ {directory.name}: {e}", fg=typer.colors.RED, err=True)
            return False

    typer.echo(f"Scanning for transcripts in: {target_dir}")
    processed = 0

    if process_directory(target_dir):
        typer.echo(f"  ✓ {target_dir.name}")
        processed += 1

    for subdir in sorted(target_dir.rglob("*")):
        if not subdir.is_dir():
            continue
        if process_directory(subdir):
            typer.echo(f"  ✓ {subdir.relative_to(target_dir)}")
            processed += 1

    if processed == 0:
        typer.secho("No transcript files found.", fg=typer.colors.YELLOW)
    else:
        typer.secho(f"\n✓ Generated variants in {processed} directories", fg=typer.colors.GREEN)


@app.command("generate-ref-variants")
def generate_ref_variants(
    target_dir: Path = typer.Option(
        ...,
        "--target-dir",
        "-t",
        help="Directory to process (source dir or subdir with *_ref.txt files)",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    recursive: bool = typer.Option(
        False,
        "--recursive",
        "-R",
        help="Recursively process subdirectories of target directory",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force recursive descent even when target directory has no ref files",
    ),
) -> None:
    """
    Regenerate V*.txt variants from existing reference transcript files (*_ref.txt).

    Use this after changing variant calculation logic or after manual
    edits to reference transcript files.

    By default, only processes *_ref.txt files in the target directory.
    Use -R to also process subdirectories. Use -f to force recursion even
    when the target directory has no ref files.
    """
    from evaluator.parser import parse_source
    from evaluator.prepare import prepare_all_variants

    def find_ref_files(directory: Path) -> list[Path]:
        """Find all *_ref.txt files in directory (non-recursive)."""
        return list(directory.glob("*_ref.txt", case_sensitive=False))

    def process_ref_file(ref_path: Path) -> bool:
        """Process a single reference file, return True if variants were generated."""
        try:
            segments = parse_source(ref_path)
            variants = prepare_all_variants(segments)

            base_name = ref_path.stem.replace("_ref", "")
            output_dir = ref_path.parent

            for variant_name, text in variants.items():
                variant_file = output_dir / f"{base_name}_{variant_name}.txt"
                variant_file.write_text(text, encoding="utf-8")

            return True
        except Exception as e:
            typer.secho(f"  ✗ {ref_path.name}: {e}", fg=typer.colors.RED, err=True)
            return False

    typer.echo(f"Scanning for reference files in: {target_dir}")
    processed = 0

    for ref_file in sorted(find_ref_files(target_dir)):
        if process_ref_file(ref_file):
            typer.echo(f"  ✓ {ref_file.name}")
            processed += 1

    should_recurse = recursive or force
    if should_recurse:
        for subdir in sorted(target_dir.rglob("*")):
            if not subdir.is_dir():
                continue
            for ref_file in sorted(find_ref_files(subdir)):
                if process_ref_file(ref_file):
                    typer.echo(f"  ✓ {ref_file.relative_to(target_dir)}")
                    processed += 1

    if processed == 0:
        typer.secho("No reference files (*_ref.txt) found.", fg=typer.colors.YELLOW)
    else:
        typer.secho(f"\n✓ Generated variants for {processed} reference files", fg=typer.colors.GREEN)


@app.command("update-results")
def update_results(
    run_dir: Path = typer.Option(
        ...,
        "--run-dir",
        "-r",
        help="Root run directory containing variation subdirectories",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
) -> None:
    """
    Recompute result.csv files from existing transcripts and references.

    Walks subdirectories from run_dir, finds variation directories (those with
    params.json), recomputes metrics from transcript JSON and reference files,
    updates each result.csv, then regenerates aggregated_results.csv.

    Use this after changing comparison/variant logic or to repair missing results.
    """
    from evaluator.parser import parse_source, parse_whisperx_json
    from evaluator.prepare import prepare_all_variants

    def find_transcript_json(directory: Path) -> Path | None:
        """Find transcript JSON file in directory (excluding params.json)."""
        json_files = list(directory.glob("*.json"))
        transcript_files = [f for f in json_files if f.name != "params.json"]
        return transcript_files[0] if transcript_files else None

    def find_ref_file(source_dir: Path) -> Path | None:
        """Find reference file (*_ref.txt) in source directory."""
        ref_files = list(source_dir.glob("*_ref.txt")) + list(source_dir.glob("*_REF.txt"))
        return ref_files[0] if ref_files else None

    def load_run_metadata(variation_dir: Path) -> dict | None:
        """Load transcriber_run.metadata if it exists."""
        metadata_path = variation_dir / "transcriber_run.metadata"
        if not metadata_path.exists():
            return None
        try:
            import json
            with metadata_path.open() as f:
                return json.load(f)
        except Exception:
            return None

    def update_single_result(variation_dir: Path) -> tuple[bool, str | None]:
        """
        Update result.csv for a single variation directory.

        Returns:
            Tuple of (success, error_message)
        """
        params_path = variation_dir / "params.json"
        if not params_path.exists():
            return False, "No params.json found"

        source_dir = variation_dir.parent
        ref_path = find_ref_file(source_dir)
        if not ref_path:
            return False, f"No reference file found in {source_dir.name}"

        run_metadata = load_run_metadata(variation_dir)
        if run_metadata is None:
            typer.secho(
                f"  ⚠ {variation_dir.name}: No transcriber_run.metadata, timing data will be missing",
                fg=typer.colors.YELLOW,
                err=True,
            )
            run_metadata = {}

        stored_error = run_metadata.get("error")
        runtime_seconds = run_metadata.get("runtime_seconds")

        transcript_path = find_transcript_json(variation_dir)
        if not transcript_path:
            if stored_error:
                try:
                    import json
                    with params_path.open() as f:
                        params_data = json.load(f)
                    params = TranscriptionParams(params=params_data.get("params", params_data))

                    result = ComparisonResult(
                        source_name=source_dir.name,
                        variation_name=variation_dir.name,
                        params=params,
                        levenshtein_va=0.0,
                        levenshtein_vb=0.0,
                        levenshtein_vc=0.0,
                        levenshtein_vd=0.0,
                        runtime_seconds=runtime_seconds,
                        error=stored_error,
                    )
                    write_result_csv(result, variation_dir)
                    return True, None
                except Exception as e:
                    return False, str(e)
            return False, "No transcript JSON found"

        try:
            import json
            with params_path.open() as f:
                params_data = json.load(f)
            params = TranscriptionParams(params=params_data.get("params", params_data))

            generated_segments = parse_whisperx_json(transcript_path)
            generated_variants = prepare_all_variants(generated_segments)

            reference_segments = parse_source(ref_path)
            reference_variants = prepare_all_variants(reference_segments)

            metrics = compare_all_variants(generated_variants, reference_variants)

            source_name = source_dir.name
            variation_name = variation_dir.name

            result = ComparisonResult(
                source_name=source_name,
                variation_name=variation_name,
                params=params,
                levenshtein_va=metrics.get("VA").levenshtein_ratio if metrics.get("VA") else 0.0,
                levenshtein_vb=metrics.get("VB").levenshtein_ratio if metrics.get("VB") else 0.0,
                levenshtein_vc=metrics.get("VC").levenshtein_ratio if metrics.get("VC") else 0.0,
                levenshtein_vd=metrics.get("VD").levenshtein_ratio if metrics.get("VD") else 0.0,
                wer_va=metrics.get("VA").wer if metrics.get("VA") else None,
                wer_vb=metrics.get("VB").wer if metrics.get("VB") else None,
                wer_vc=metrics.get("VC").wer if metrics.get("VC") else None,
                wer_vd=metrics.get("VD").wer if metrics.get("VD") else None,
                cer_va=metrics.get("VA").cer if metrics.get("VA") else None,
                cer_vb=metrics.get("VB").cer if metrics.get("VB") else None,
                cer_vc=metrics.get("VC").cer if metrics.get("VC") else None,
                cer_vd=metrics.get("VD").cer if metrics.get("VD") else None,
                runtime_seconds=runtime_seconds,
                error=stored_error,
            )

            for variant_name, text in generated_variants.items():
                (variation_dir / f"{variant_name}.txt").write_text(text, encoding="utf-8")

            write_result_csv(result, variation_dir)
            return True, None

        except Exception as e:
            return False, str(e)

    typer.echo(f"Scanning for variation directories in: {run_dir}")

    params_files = sorted(run_dir.rglob("params.json"))
    if not params_files:
        typer.secho("No variation directories found (no params.json files).", fg=typer.colors.YELLOW)
        raise typer.Exit(1)

    typer.echo(f"Found {len(params_files)} variation directories")

    updated = 0
    errors: list[str] = []

    for params_path in tqdm(params_files, desc="Updating results", unit="var"):
        variation_dir = params_path.parent
        success, error = update_single_result(variation_dir)

        if success:
            updated += 1
        else:
            rel_path = variation_dir.relative_to(run_dir)
            errors.append(f"{rel_path}: {error}")

    if errors:
        typer.echo(f"\n{len(errors)} errors:")
        for err in errors[:10]:
            typer.secho(f"  ✗ {err}", fg=typer.colors.RED)
        if len(errors) > 10:
            typer.echo(f"  ... and {len(errors) - 10} more")

    typer.echo(f"\n✓ Updated {updated} result.csv files")

    typer.echo("\nAggregating results...")
    try:
        aggregated_path = aggregate_results(run_dir)
        typer.secho(f"✓ Aggregated results: {aggregated_path}", fg=typer.colors.GREEN)
    except ValueError as e:
        typer.secho(f"Warning: {e}", fg=typer.colors.YELLOW)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
