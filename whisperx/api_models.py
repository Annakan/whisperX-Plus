from __future__ import annotations

from typing import Annotated, Any, Literal, Optional

import torch
from fastapi import Form
from pydantic import BaseModel, Field, field_validator
from typing import Annotated

from whisperx.utils import LANGUAGES, TO_LANGUAGE_CODE


LogLevel = Literal["debug", "info", "warning", "error", "critical"]


class TranscribeParams(BaseModel):
    # Model parameters
    model: str = Field(default="small", description="Name of the Whisper model to use")
    model_cache_only: bool = Field(
        default=False,
        description="If True, will not attempt to download models, instead using cached models from model_dir",
    )
    model_dir: Optional[str] = Field(default=None, description="Path to save model files")
    device: str = Field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu",
        description="Device to use for PyTorch inference",
    )
    device_index: int = Field(default=0, description="Device index to use for FasterWhisper inference")
    batch_size: int = Field(default=8, description="Preferred batch size for inference")
    compute_type: Literal["float16", "float32", "int8"] = Field(
        default="float16",
        description="Compute type for computation",
    )

    # Output parameters
    output_dir: str = Field(default=".", description="Directory to save the outputs")
    output_format: Literal["all", "srt", "vtt", "txt", "tsv", "json", "aud"] = Field(
        default="all",
        description="Format of the output file(s)",
    )
    verbose: bool = Field(default=True, description="Whether to print progress and debug messages")
    log_level: Optional[LogLevel] = Field(
        default=None,
        description="Logging level (overrides verbose if set)",
    )

    # Task parameters
    task: Literal["transcribe", "translate"] = Field(
        default="transcribe",
        description="Task to perform",
    )
    language: Optional[str] = Field(default=None, description="Language spoken in the audio")

    # alignment params
    align_model: Optional[str] = Field(default=None, description="Name of phoneme-level ASR model to do alignment")
    interpolate_method: Literal["nearest", "linear", "ignore"] = Field(
        default="nearest",
        description="Method to assign timestamps to non-aligned words",
    )
    no_align: bool = Field(default=False, description="Do not perform phoneme alignment")
    return_char_alignments: bool = Field(default=False, description="Return character-level alignments")

    # vad params
    vad_method: Literal["pyannote", "silero"] = Field(default="pyannote", description="VAD method to be used")
    vad_onset: float = Field(default=0.500, description="Onset threshold for VAD")
    vad_offset: float = Field(default=0.363, description="Offset threshold for VAD")
    chunk_size: int = Field(default=30, description="Chunk size for merging VAD segments")

    # diarization params
    diarize: bool = Field(default=False, description="Apply diarization")
    min_speakers: Optional[int] = Field(default=None, description="Minimum number of speakers")
    max_speakers: Optional[int] = Field(default=None, description="Maximum number of speakers")
    diarize_model: str = Field(
        default="pyannote/speaker-diarization-3.1",
        description="Name of the speaker diarization model to use",
    )
    speaker_embeddings: bool = Field(
        default=False,
        description="Include speaker embeddings in JSON output (only works with diarize)",
    )

    # decoding params
    temperature: float = Field(default=0, description="Temperature to use for sampling")
    best_of: Optional[int] = Field(default=5, description="Number of candidates when sampling with non-zero temperature")
    beam_size: Optional[int] = Field(default=5, description="Number of beams in beam search")
    patience: float = Field(default=1.0, description="Patience value in beam decoding")
    length_penalty: float = Field(default=1.0, description="Token length penalty coefficient")

    suppress_tokens: str = Field(default="-1", description="Comma-separated list of token ids to suppress")
    suppress_numerals: bool = Field(default=False, description="Whether to suppress numeric symbols")

    initial_prompt: Optional[str] = Field(default=None, description="Optional text to provide as a prompt")
    hotwords: Optional[str] = Field(default=None, description="Hotwords/hint phrases to the model")
    condition_on_previous_text: bool = Field(
        default=False,
        description="Provide previous output as prompt",
    )
    fp16: bool = Field(default=True, description="Whether to perform inference in fp16")

    temperature_increment_on_fallback: Optional[float] = Field(default=0.2, description="Temperature increment on fallback")
    compression_ratio_threshold: Optional[float] = Field(default=2.4, description="Compression ratio threshold")
    logprob_threshold: Optional[float] = Field(default=-1.0, description="Log prob threshold")
    no_speech_threshold: Optional[float] = Field(default=0.6, description="No speech threshold")

    max_line_width: Optional[int] = Field(default=None, description="Maximum characters in a line")
    max_line_count: Optional[int] = Field(default=None, description="Maximum number of lines in a segment")
    highlight_words: bool = Field(default=False, description="Underline each word as it is spoken")
    segment_resolution: Literal["sentence", "chunk"] = Field(default="sentence", description="Segment resolution")

    threads: int = Field(default=0, description="Number of torch threads for CPU inference")

    hf_token: Optional[str] = Field(default=None, description="Hugging Face Access Token")

    print_progress: bool = Field(default=False, description="Print progress in transcribe() and align()")

    @field_validator("language")
    @classmethod
    def normalize_language(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        lang = value.lower()
        if lang in LANGUAGES:
            return lang
        if lang in TO_LANGUAGE_CODE:
            return TO_LANGUAGE_CODE[lang]
        raise ValueError(f"Unsupported language: {value}")

    @field_validator("speaker_embeddings")
    @classmethod
    def validate_speaker_embeddings(cls, value: bool, info: Any) -> bool:
        diarize = info.data.get("diarize")
        if value and not diarize:
            # Keep behavior consistent with current CLI: it warns; here we still allow but downstream may warn.
            return value
        return value

    @classmethod
    def as_form(
        cls,
        model: Annotated[str, Form()] = "small",
        model_cache_only: Annotated[bool, Form()] = False,
        model_dir: Annotated[Optional[str], Form()] = None,
        device: Annotated[str, Form()] = "cuda" if torch.cuda.is_available() else "cpu",
        device_index: Annotated[int, Form()] = 0,
        batch_size: Annotated[int, Form()] = 8,
        compute_type: Annotated[str, Form()] = "float16",
        output_dir: Annotated[str, Form()] = ".",
        output_format: Annotated[str, Form()] = "all",
        verbose: Annotated[bool, Form()] = True,
        log_level: Annotated[Optional[str], Form()] = None,
        task: Annotated[str, Form()] = "transcribe",
        language: Annotated[Optional[str], Form()] = None,
        align_model: Annotated[Optional[str], Form()] = None,
        interpolate_method: Annotated[str, Form()] = "nearest",
        no_align: Annotated[bool, Form()] = False,
        return_char_alignments: Annotated[bool, Form()] = False,
        vad_method: Annotated[str, Form()] = "pyannote",
        vad_onset: Annotated[float, Form()] = 0.500,
        vad_offset: Annotated[float, Form()] = 0.363,
        chunk_size: Annotated[int, Form()] = 30,
        diarize: Annotated[bool, Form()] = False,
        min_speakers: Annotated[Optional[int], Form()] = None,
        max_speakers: Annotated[Optional[int], Form()] = None,
        diarize_model: Annotated[str, Form()] = "pyannote/speaker-diarization-3.1",
        speaker_embeddings: Annotated[bool, Form()] = False,
        temperature: Annotated[float, Form()] = 0,
        best_of: Annotated[Optional[int], Form()] = 5,
        beam_size: Annotated[Optional[int], Form()] = 5,
        patience: Annotated[float, Form()] = 1.0,
        length_penalty: Annotated[float, Form()] = 1.0,
        suppress_tokens: Annotated[str, Form()] = "-1",
        suppress_numerals: Annotated[bool, Form()] = False,
        initial_prompt: Annotated[Optional[str], Form()] = None,
        hotwords: Annotated[Optional[str], Form()] = None,
        condition_on_previous_text: Annotated[bool, Form()] = False,
        fp16: Annotated[bool, Form()] = True,
        temperature_increment_on_fallback: Annotated[Optional[float], Form()] = 0.2,
        compression_ratio_threshold: Annotated[Optional[float], Form()] = 2.4,
        logprob_threshold: Annotated[Optional[float], Form()] = -1.0,
        no_speech_threshold: Annotated[Optional[float], Form()] = 0.6,
        max_line_width: Annotated[Optional[int], Form()] = None,
        max_line_count: Annotated[Optional[int], Form()] = None,
        highlight_words: Annotated[bool, Form()] = False,
        segment_resolution: Annotated[str, Form()] = "sentence",
        threads: Annotated[int, Form()] = 0,
        hf_token: Annotated[Optional[str], Form()] = None,
        print_progress: Annotated[bool, Form()] = False,
    ) -> "TranscribeParams":
        return cls(
            model=model,
            model_cache_only=model_cache_only,
            model_dir=model_dir,
            device=device,
            device_index=device_index,
            batch_size=batch_size,
            compute_type=compute_type,
            output_dir=output_dir,
            output_format=output_format,
            verbose=verbose,
            log_level=log_level,
            task=task,
            language=language,
            align_model=align_model,
            interpolate_method=interpolate_method,
            no_align=no_align,
            return_char_alignments=return_char_alignments,
            vad_method=vad_method,
            vad_onset=vad_onset,
            vad_offset=vad_offset,
            chunk_size=chunk_size,
            diarize=diarize,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            diarize_model=diarize_model,
            speaker_embeddings=speaker_embeddings,
            temperature=temperature,
            best_of=best_of,
            beam_size=beam_size,
            patience=patience,
            length_penalty=length_penalty,
            suppress_tokens=suppress_tokens,
            suppress_numerals=suppress_numerals,
            initial_prompt=initial_prompt,
            hotwords=hotwords,
            condition_on_previous_text=condition_on_previous_text,
            fp16=fp16,
            temperature_increment_on_fallback=temperature_increment_on_fallback,
            compression_ratio_threshold=compression_ratio_threshold,
            logprob_threshold=logprob_threshold,
            no_speech_threshold=no_speech_threshold,
            max_line_width=max_line_width,
            max_line_count=max_line_count,
            highlight_words=highlight_words,
            segment_resolution=segment_resolution,
            threads=threads,
            hf_token=hf_token,
            print_progress=print_progress,
        )


class TranscribeRequest(TranscribeParams):
    audio: list[str] = Field(default_factory=list, description="Audio file path(s) to transcribe")


class ServeRequest(BaseModel):
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    workers: int = Field(default=1)
    log_level: LogLevel = Field(default="info")
