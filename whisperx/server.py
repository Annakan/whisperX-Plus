"""
FastAPI REST API server for WhisperX transcription.

This module provides an HTTP REST API that exposes WhisperX functionality
through a single endpoint that accepts the same parameters as the CLI.
"""

import tempfile
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel

from whisperx.api_models import TranscribeParams


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
async def serve_transcribe(
    audio: UploadFile = File(..., description="Audio file to transcribe"),
    params: TranscribeParams = Depends(TranscribeParams.as_form),
):
    """
    Transcribe an audio file using WhisperX.
    
    This endpoint accepts an audio file and various parameters that mirror
    the command-line interface, processes the audio, and returns the
    transcription results.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_path = Path(temp_dir) / audio.filename
        try:
            content = await audio.read()
            with open(audio_path, "wb") as f:
                f.write(content)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to save audio file: {str(e)}")

        try:
            from whisperx.transcribe import run_transcription

            params = params.model_copy(update={"output_dir": params.output_dir or temp_dir})
            results = run_transcription(params, [str(audio_path)])
            if not results:
                raise HTTPException(status_code=500, detail="Transcription completed but no result returned")
            return TranscriptionResponse(**results[0])
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


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
