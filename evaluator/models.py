from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class Segment(BaseModel):
    """A single segment from a transcript with speaker and text."""

    locuteur: str = Field(description="Speaker identifier (A, B, C, ...)")
    text: str = Field(description="The transcribed text for this segment")


class RunPlan(BaseModel):
    """Configuration for an evaluation run loaded from runplan.yaml."""

    to_process: list[str] = Field(
        description="List of subdirectory names to process from source_data_dir"
    )
    base_args: dict[str, Any] = Field(
        default_factory=dict,
        description="Fixed arguments for transcription (override defaults but don't vary)",
    )
    varying_args: dict[str, list[Any]] = Field(
        default_factory=dict,
        description="Parameters to vary: key is arg name, value is list of values to try",
    )


class TranscriptionParams(BaseModel):
    """Parameters for a single transcription run."""

    params: dict[str, Any] = Field(
        default_factory=dict,
        description="All varying parameters as key-value pairs",
    )

    def to_dirname(self) -> str:
        """Generate a directory name from parameters."""
        parts = []
        for key in sorted(self.params.keys()):
            value = self.params[key]
            if isinstance(value, float) and value == int(value):
                value = int(value)
            parts.append(f"{key}={value}")
        return "_".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return self.params.copy()

    def get(self, key: str, default: Any = None) -> Any:
        """Get a parameter value."""
        return self.params.get(key, default)


class ComparisonResult(BaseModel):
    """Result of comparing a generated transcript against reference."""

    source_name: str = Field(description="Name of the source subdirectory")
    variation_name: str = Field(description="Parameter variation identifier")
    params: TranscriptionParams = Field(description="Parameters used for transcription")

    levenshtein_va: float = Field(description="Levenshtein ratio for VA variant")
    levenshtein_vb: float = Field(description="Levenshtein ratio for VB variant")
    levenshtein_vc: float = Field(description="Levenshtein ratio for VC variant")
    levenshtein_vd: float = Field(description="Levenshtein ratio for VD variant")

    wer_va: float | None = Field(default=None, description="WER for VA variant")
    wer_vb: float | None = Field(default=None, description="WER for VB variant")
    wer_vc: float | None = Field(default=None, description="WER for VC variant")
    wer_vd: float | None = Field(default=None, description="WER for VD variant")

    cer_va: float | None = Field(default=None, description="CER for VA variant")
    cer_vb: float | None = Field(default=None, description="CER for VB variant")
    cer_vc: float | None = Field(default=None, description="CER for VC variant")
    cer_vd: float | None = Field(default=None, description="CER for VD variant")

    runtime_seconds: float | None = Field(
        default=None, description="Time taken for transcription"
    )
    error: str | None = Field(default=None, description="Error message if failed")


class SourceFiles(BaseModel):
    """Paths to source files in a subdirectory."""

    audio_path: Path = Field(description="Path to audio file (audio_*.*)")
    reference_path: Path = Field(description="Path to reference transcript (*_REF.txt)")
    subdir_name: str = Field(description="Name of the subdirectory")
