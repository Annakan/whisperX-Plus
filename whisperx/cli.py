from __future__ import annotations

from typing import Literal, Optional

import typer

from whisperx.api_models import ServeRequest, TranscribeParams
from whisperx.log_utils import setup_logging

app = typer.Typer(add_completion=True)


@app.command("serve")
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind the server to"),
    port: int = typer.Option(8000, "--port", help="Port to bind the server to"),
    workers: int = typer.Option(1, "--workers", help="Number of worker processes"),
    log_level: Literal["debug", "info", "warning", "error", "critical"] = typer.Option(
        "info",
        "--log-level",
        help="Logging level (debug, info, warning, error, critical)",
    ),
):
    setup_logging(level=log_level)
    from whisperx.server import start_server

    request = ServeRequest(host=host, port=port, workers=workers, log_level=log_level)
    start_server(
        host=request.host,
        port=request.port,
        workers=request.workers,
        log_level=request.log_level,
    )


@app.command("transcribe")
def transcribe(
    audio: list[str] = typer.Argument(..., help="Audio file(s) to transcribe"),
    model: str = typer.Option(
        "small",
        "--model",
        envvar="W_MODEL",
        help="Name of the Whisper model to use",
    ),
    model_cache_only: bool = typer.Option(
        False,
        "--model_cache_only",
        help="If True, will not attempt to download models, instead using cached models from --model_dir",
    ),
    model_dir: Optional[str] = typer.Option(None, "--model_dir", help="Path to save model files"),
    device: Optional[str] = typer.Option(
        None,
        "--device",
        help="Device to use for PyTorch inference (defaults to auto)",
    ),
    device_index: int = typer.Option(0, "--device_index", help="Device index for FasterWhisper inference"),
    batch_size: int = typer.Option(8, "--batch_size", help="Preferred batch size for inference"),
    compute_type: Literal["float16", "float32", "int8"] = typer.Option(
        "float16",
        "--compute_type",
        help="Compute type for computation",
    ),
    output_dir: str = typer.Option(".", "--output_dir", "-o", help="Directory to save outputs"),
    output_format: Literal["all", "srt", "vtt", "txt", "tsv", "json", "aud"] = typer.Option(
        "all",
        "--output_format",
        "-f",
        help="Output format",
    ),
    verbose: bool = typer.Option(True, "--verbose", help="Print progress and debug messages"),
    log_level: Optional[Literal["debug", "info", "warning", "error", "critical"]] = typer.Option(
        None,
        "--log-level",
        help="Logging level (overrides verbose if set)",
    ),
    task: Literal["transcribe", "translate"] = typer.Option(
        "transcribe",
        "--task",
        help="Task to perform",
    ),
    language: Optional[str] = typer.Option(
        None,
        "--language",
        envvar="W_LANG",
        help="Language spoken in the audio",
    ),
    align_model: Optional[str] = typer.Option(None, "--align_model", help="Alignment model name"),
    interpolate_method: Literal["nearest", "linear", "ignore"] = typer.Option(
        "nearest",
        "--interpolate_method",
        help="Interpolation method for non-aligned words",
    ),
    no_align: bool = typer.Option(False, "--no_align", help="Disable phoneme alignment"),
    return_char_alignments: bool = typer.Option(
        False,
        "--return_char_alignments",
        help="Return character-level alignments",
    ),
    vad_method: Literal["pyannote", "silero"] = typer.Option(
        "pyannote",
        "--vad_method",
        help="VAD method",
    ),
    vad_onset: float = typer.Option(0.500, "--vad_onset", help="VAD onset threshold"),
    vad_offset: float = typer.Option(0.363, "--vad_offset", help="VAD offset threshold"),
    chunk_size: int = typer.Option(30, "--chunk_size", help="Chunk size for merging VAD segments"),
    preprocess: int = typer.Option(
        0,
        "--preprocess",
        min=0,
        max=4,
        help="Audio preprocessing level: 0=None, 1=Sanitize, 2=+Filter, 3=+ReduceNoise, 4=+Normalize",
    ),
    stationary_nr: bool = typer.Option(True, "--stationary_nr", help="Use stationary noise reduction"),
    target_dBFS: float = typer.Option(-18.0, "--target_dBFS", help="Target dBFS for normalization"),
    lowpass_freq: int = typer.Option(8000, "--lowpass_freq", help="Lowpass filter frequency in Hz"),
    highpass_freq: int = typer.Option(45, "--highpass_freq", help="Highpass filter frequency in Hz"),
    prop_decrease: float = typer.Option(0.3, "--prop_decrease", help="Proportion to reduce noise by (0.0-1.0)"),
    diarize: bool = typer.Option(False, "--diarize", help="Enable diarization"),
    min_speakers: Optional[int] = typer.Option(None, "--min_speakers", help="Minimum speakers"),
    max_speakers: Optional[int] = typer.Option(None, "--max_speakers", help="Maximum speakers"),
    diarize_model: str = typer.Option(
        "pyannote/speaker-diarization-3.1",
        "--diarize_model",
        help="Diarization model name",
    ),
    speaker_embeddings: bool = typer.Option(
        False,
        "--speaker_embeddings",
        help="Include speaker embeddings in JSON output (requires diarize)",
    ),
    temperature: float = typer.Option(0, "--temperature", help="Sampling temperature"),
    best_of: Optional[int] = typer.Option(5, "--best_of", help="Candidates when sampling with non-zero temperature"),
    beam_size: Optional[int] = typer.Option(5, "--beam_size", help="Beam size"),
    patience: float = typer.Option(1.0, "--patience", help="Patience value"),
    length_penalty: float = typer.Option(1.0, "--length_penalty", help="Length penalty"),
    suppress_tokens: str = typer.Option("-1", "--suppress_tokens", help="Token ids to suppress"),
    suppress_numerals: bool = typer.Option(False, "--suppress_numerals", help="Suppress numerals"),
    initial_prompt: Optional[str] = typer.Option(None, "--initial_prompt", help="Initial prompt"),
    hotwords: Optional[str] = typer.Option(None, "--hotwords", help="Hotwords/hints"),
    condition_on_previous_text: bool = typer.Option(
        False,
        "--condition_on_previous_text",
        help="Condition on previous text",
    ),
    fp16: bool = typer.Option(True, "--fp16", help="Use fp16"),
    temperature_increment_on_fallback: Optional[float] = typer.Option(
        0.2,
        "--temperature_increment_on_fallback",
        help="Temperature increment on fallback",
    ),
    compression_ratio_threshold: Optional[float] = typer.Option(
        2.4,
        "--compression_ratio_threshold",
        help="Compression ratio threshold",
    ),
    logprob_threshold: Optional[float] = typer.Option(-1.0, "--logprob_threshold", help="Logprob threshold"),
    no_speech_threshold: Optional[float] = typer.Option(0.6, "--no_speech_threshold", help="No speech threshold"),
    max_line_width: Optional[int] = typer.Option(None, "--max_line_width", help="Max line width"),
    max_line_count: Optional[int] = typer.Option(None, "--max_line_count", help="Max line count"),
    highlight_words: bool = typer.Option(False, "--highlight_words", help="Highlight words"),
    segment_resolution: Literal["sentence", "chunk"] = typer.Option(
        "sentence",
        "--segment_resolution",
        help="Segment resolution",
    ),
    threads: int = typer.Option(0, "--threads", help="Torch threads"),
    hf_token: Optional[str] = typer.Option(None, "--hf_token", help="Hugging Face token"),
    print_progress: bool = typer.Option(False, "--print_progress", help="Print progress"),
):
    if log_level is not None:
        setup_logging(level=log_level)
    elif verbose:
        setup_logging(level="warning")

    from whisperx.transcribe import run_transcription

    params = TranscribeParams(
        model=model,
        model_cache_only=model_cache_only,
        model_dir=model_dir,
        device=device or TranscribeParams().device,
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
        preprocess=preprocess,
        stationary_nr=stationary_nr,
        target_dBFS=target_dBFS,
        lowpass_freq=lowpass_freq,
        highpass_freq=highpass_freq,
        prop_decrease=prop_decrease,
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

    run_transcription(params, audio)


def main() -> None:
    app()
