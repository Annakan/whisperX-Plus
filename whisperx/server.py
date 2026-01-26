"""
FastAPI REST API server for WhisperX transcription.

This module provides an HTTP REST API that exposes WhisperX functionality
through a single endpoint that accepts the same parameters as the CLI.
"""

import argparse
import io
import os
import tempfile
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from whisperx.utils import LANGUAGES, TO_LANGUAGE_CODE, optional_float, optional_int, str2bool


# Create FastAPI app
app = FastAPI(
    title="WhisperX API",
    description="REST API for WhisperX automatic speech recognition with word-level timestamps",
    version="1.0.0",
)


class TranscriptionResponse(BaseModel):
    """Response model for transcription results."""
    segments: list
    language: str
    word_segments: Optional[list] = None


@app.get("/")
async def root():
    """Root endpoint providing API information."""
    return {
        "name": "WhisperX API",
        "version": "1.0.0",
        "endpoints": {
            "/transcribe": "POST - Transcribe audio file with WhisperX"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    audio: UploadFile = File(..., description="Audio file to transcribe"),
    # Model parameters
    model: str = Form(default="small", description="Name of the Whisper model to use"),
    model_cache_only: bool = Form(default=False, description="If True, will not attempt to download models"),
    model_dir: Optional[str] = Form(default=None, description="Path to save model files"),
    device: str = Form(default="cuda" if torch.cuda.is_available() else "cpu", description="Device to use for PyTorch inference"),
    device_index: int = Form(default=0, description="Device index to use for FasterWhisper inference"),
    batch_size: int = Form(default=8, description="Preferred batch size for inference"),
    compute_type: str = Form(default="float16", description="Compute type for computation"),
    # Output parameters
    output_format: str = Form(default="json", description="Format of the output (json, srt, vtt, txt, tsv, aud)"),
    verbose: bool = Form(default=True, description="Whether to print out the progress and debug messages"),
    # Task parameters
    task: str = Form(default="transcribe", description="Task to perform: transcribe or translate"),
    language: Optional[str] = Form(default=None, description="Language spoken in the audio"),
    # Alignment parameters
    align_model: Optional[str] = Form(default=None, description="Name of phoneme-level ASR model to do alignment"),
    interpolate_method: str = Form(default="nearest", description="Method to assign timestamps to non-aligned words"),
    no_align: bool = Form(default=False, description="Do not perform phoneme alignment"),
    return_char_alignments: bool = Form(default=False, description="Return character-level alignments"),
    # VAD parameters
    vad_method: str = Form(default="pyannote", description="VAD method to be used"),
    vad_onset: float = Form(default=0.500, description="Onset threshold for VAD"),
    vad_offset: float = Form(default=0.363, description="Offset threshold for VAD"),
    chunk_size: int = Form(default=30, description="Chunk size for merging VAD segments"),
    # Diarization parameters
    diarize: bool = Form(default=False, description="Apply diarization to assign speaker labels"),
    min_speakers: Optional[int] = Form(default=None, description="Minimum number of speakers"),
    max_speakers: Optional[int] = Form(default=None, description="Maximum number of speakers"),
    diarize_model: str = Form(default="pyannote/speaker-diarization-3.1", description="Name of the speaker diarization model"),
    speaker_embeddings: bool = Form(default=False, description="Include speaker embeddings in JSON output"),
    # Decoding parameters
    temperature: float = Form(default=0, description="Temperature to use for sampling"),
    best_of: Optional[int] = Form(default=5, description="Number of candidates when sampling with non-zero temperature"),
    beam_size: Optional[int] = Form(default=5, description="Number of beams in beam search"),
    patience: float = Form(default=1.0, description="Patience value to use in beam decoding"),
    length_penalty: float = Form(default=1.0, description="Token length penalty coefficient"),
    suppress_tokens: str = Form(default="-1", description="Comma-separated list of token ids to suppress"),
    suppress_numerals: bool = Form(default=False, description="Whether to suppress numeric symbols"),
    initial_prompt: Optional[str] = Form(default=None, description="Optional text to provide as a prompt"),
    condition_on_previous_text: bool = Form(default=False, description="Provide previous output as prompt"),
    fp16: bool = Form(default=True, description="Whether to perform inference in fp16"),
    # Fallback parameters
    temperature_increment_on_fallback: Optional[float] = Form(default=0.2, description="Temperature to increase on fallback"),
    compression_ratio_threshold: Optional[float] = Form(default=2.4, description="Compression ratio threshold"),
    logprob_threshold: Optional[float] = Form(default=-1.0, description="Log probability threshold"),
    no_speech_threshold: Optional[float] = Form(default=0.6, description="No speech threshold"),
    # Output formatting parameters
    max_line_width: Optional[int] = Form(default=None, description="Maximum characters in a line"),
    max_line_count: Optional[int] = Form(default=None, description="Maximum number of lines in a segment"),
    highlight_words: bool = Form(default=False, description="Underline each word as it is spoken"),
    segment_resolution: str = Form(default="sentence", description="Segment resolution: sentence or chunk"),
    # System parameters
    threads: int = Form(default=0, description="Number of threads used by torch for CPU inference"),
    hf_token: Optional[str] = Form(default=None, description="Hugging Face Access Token"),
    print_progress: bool = Form(default=False, description="Print progress in transcribe() and align() methods"),
):
    """
    Transcribe an audio file using WhisperX.
    
    This endpoint accepts an audio file and various parameters that mirror
    the command-line interface, processes the audio, and returns the
    transcription results.
    """
    # Create a temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded file to temporary location
        audio_path = Path(temp_dir) / audio.filename
        try:
            content = await audio.read()
            with open(audio_path, "wb") as f:
                f.write(content)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to save audio file: {str(e)}")
        
        # Build arguments dictionary matching CLI structure
        args = {
            "audio": [str(audio_path)],
            "model": model,
            "model_cache_only": model_cache_only,
            "model_dir": model_dir,
            "device": device,
            "device_index": device_index,
            "batch_size": batch_size,
            "compute_type": compute_type,
            "output_dir": temp_dir,
            "output_format": output_format,
            "verbose": verbose,
            "task": task,
            "language": language,
            "align_model": align_model,
            "interpolate_method": interpolate_method,
            "no_align": no_align,
            "return_char_alignments": return_char_alignments,
            "vad_method": vad_method,
            "vad_onset": vad_onset,
            "vad_offset": vad_offset,
            "chunk_size": chunk_size,
            "diarize": diarize,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
            "diarize_model": diarize_model,
            "speaker_embeddings": speaker_embeddings,
            "temperature": temperature,
            "best_of": best_of,
            "beam_size": beam_size,
            "patience": patience,
            "length_penalty": length_penalty,
            "suppress_tokens": suppress_tokens,
            "suppress_numerals": suppress_numerals,
            "initial_prompt": initial_prompt,
            "condition_on_previous_text": condition_on_previous_text,
            "fp16": fp16,
            "temperature_increment_on_fallback": temperature_increment_on_fallback,
            "compression_ratio_threshold": compression_ratio_threshold,
            "logprob_threshold": logprob_threshold,
            "no_speech_threshold": no_speech_threshold,
            "max_line_width": max_line_width,
            "max_line_count": max_line_count,
            "highlight_words": highlight_words,
            "segment_resolution": segment_resolution,
            "threads": threads,
            "hf_token": hf_token,
            "print_progress": print_progress,
        }
        
        # Create a minimal parser for transcribe_task validation
        parser = argparse.ArgumentParser()
        
        try:
            # Import and call transcribe_task
            from whisperx.transcribe import transcribe_task
            
            # Note: transcribe_task writes files but doesn't return results
            # We need to read the output file it creates
            transcribe_task(args, parser)
            
            # Read the JSON output file
            json_file = Path(temp_dir) / f"{audio_path.stem}.json"
            if json_file.exists():
                import json
                with open(json_file, "r") as f:
                    result = json.load(f)
                return TranscriptionResponse(**result)
            else:
                raise HTTPException(
                    status_code=500,
                    detail="Transcription completed but output file not found"
                )
                
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Transcription failed: {str(e)}"
            )


def start_server(host: str = "0.0.0.0", port: int = 8000, workers: int = 1, log_level: str = "info"):
    """
    Start the FastAPI server using uvicorn.
    
    Args:
        host: Host to bind the server to
        port: Port to bind the server to
        workers: Number of worker processes
        log_level: Logging level (debug, info, warning, error, critical)
    """
    import uvicorn
    
    uvicorn.run(
        "whisperx.server:app",
        host=host,
        port=port,
        workers=workers,
        log_level=log_level,
    )


if __name__ == "__main__":
    # Allow running the server directly
    start_server()
