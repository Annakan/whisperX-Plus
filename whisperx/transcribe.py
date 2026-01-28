import argparse
import gc
import os
import tempfile
import warnings
from typing import Any, Optional

import numpy as np
import torch

from whisperx.alignment import align, load_align_model
from whisperx.api_models import TranscribeParams
from whisperx.asr import load_model
from whisperx.audio import load_audio
from whisperx.audio_preprocess import preprocess_audio
from whisperx.diarize import DiarizationPipeline, assign_word_speakers
from whisperx.log_utils import get_logger
from whisperx.schema import AlignedTranscriptionResult, TranscriptionResult
from whisperx.utils import get_writer

logger = get_logger(__name__)


import pydevd_pycharm
pydevd_pycharm.settrace('localhost', port=7000, stdout_to_server=True, stderr_to_server=True)


def run_transcription(params: TranscribeParams, audio_paths: list[str]) -> list[dict[str, Any]]:
    model_name = params.model
    batch_size = params.batch_size
    model_dir = params.model_dir
    model_cache_only = params.model_cache_only
    output_dir = params.output_dir
    output_format = params.output_format
    device = params.device
    device_index = params.device_index
    compute_type = params.compute_type
    verbose = params.verbose

    preprocess_level = params.preprocess
    stationary_nr = params.stationary_nr
    target_dBFS = params.target_dBFS
    lowpass_freq = params.lowpass_freq
    highpass_freq = params.highpass_freq
    prop_decrease = params.prop_decrease

    os.makedirs(output_dir, exist_ok=True)

    align_model_name = params.align_model
    interpolate_method = params.interpolate_method
    no_align = params.no_align
    task = params.task
    if task == "translate":
        no_align = True

    return_char_alignments = params.return_char_alignments

    hf_token = params.hf_token
    vad_method = params.vad_method
    vad_onset = params.vad_onset
    vad_offset = params.vad_offset
    chunk_size = params.chunk_size

    diarize = params.diarize
    min_speakers = params.min_speakers
    max_speakers = params.max_speakers
    diarize_model_name = params.diarize_model
    print_progress = params.print_progress
    return_speaker_embeddings = params.speaker_embeddings

    if return_speaker_embeddings and not diarize:
        warnings.warn("--speaker_embeddings has no effect without --diarize")

    language = params.language
    if model_name.endswith(".en") and language != "en":
        if language is not None:
            warnings.warn(
                f"{model_name} is an English-only model but received '{language}'; using English instead."
            )
        language = "en"
    align_language = language if language is not None else "en"

    temperature = params.temperature
    increment = params.temperature_increment_on_fallback
    if increment is not None:
        temperatures = tuple(np.arange(temperature, 1.0 + 1e-6, increment))
    else:
        temperatures = [temperature]

    faster_whisper_threads = 4
    if params.threads > 0:
        torch.set_num_threads(params.threads)
        faster_whisper_threads = params.threads

    asr_options = {
        "beam_size": params.beam_size,
        "patience": params.patience,
        "length_penalty": params.length_penalty,
        "temperatures": temperatures,
        "compression_ratio_threshold": params.compression_ratio_threshold,
        "log_prob_threshold": params.logprob_threshold,
        "no_speech_threshold": params.no_speech_threshold,
        "condition_on_previous_text": False,
        "initial_prompt": params.initial_prompt,
        "hotwords": params.hotwords,
        "suppress_tokens": [int(x) for x in params.suppress_tokens.split(",")],
        "suppress_numerals": params.suppress_numerals,
    }

    writer = get_writer(output_format, output_dir)
    word_options = ["highlight_words", "max_line_count", "max_line_width"]
    if no_align:
        for option in word_options:
            if getattr(params, option):
                raise ValueError(f"--{option} not possible with --no_align")
    if params.max_line_count and not params.max_line_width:
        warnings.warn("--max_line_count has no effect without --max_line_width")
    writer_args = {
        "highlight_words": params.highlight_words,
        "max_line_count": params.max_line_count,
        "max_line_width": params.max_line_width,
    }

    # Part 1: VAD & ASR Loop
    results: list[tuple[dict[str, Any], str]] = []
    model = load_model(
        model_name,
        device=device,
        device_index=device_index,
        download_root=model_dir,
        compute_type=compute_type,
        language=language,
        asr_options=asr_options,
        vad_method=vad_method,
        vad_options={
            "chunk_size": chunk_size,
            "vad_onset": vad_onset,
            "vad_offset": vad_offset,
        },
        task=task,
        local_files_only=model_cache_only,
        threads=faster_whisper_threads,
    )

    preprocess_temp_dir = tempfile.TemporaryDirectory() if preprocess_level > 0 else None
    preprocessed_paths: dict[str, str] = {}

    audio_cache: Optional[np.ndarray] = None
    for audio_path in audio_paths:
        effective_audio_path = audio_path
        if preprocess_level > 0:
            preprocessed_filename = os.path.basename(audio_path).rsplit(".", 1)[0] + "_preprocessed.wav"
            preprocessed_path = os.path.join(preprocess_temp_dir.name, preprocessed_filename)
            logger.info(f"Preprocessing audio (level {preprocess_level})...")
            preprocess_audio(
                input_path=audio_path,
                output_path=preprocessed_path,
                preprocess_level=preprocess_level,
                highpass_freq=highpass_freq,
                lowpass_freq=lowpass_freq,
                prop_decrease=prop_decrease,
                stationary=stationary_nr,
                target_dBFS=target_dBFS,
            )
            effective_audio_path = preprocessed_path
            preprocessed_paths[audio_path] = preprocessed_path

        audio_cache = load_audio(effective_audio_path)
        logger.info("Performing transcription...")
        result: TranscriptionResult = model.transcribe(
            audio_cache,
            batch_size=batch_size,
            chunk_size=chunk_size,
            print_progress=print_progress,
            verbose=verbose,
        )
        results.append((result, audio_path, effective_audio_path))

    # Unload Whisper and VAD
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Part 2: Align Loop
    if not no_align:
        tmp_results = results
        results = []
        align_model, align_metadata = load_align_model(
            align_language, device, model_name=align_model_name
        )
        for result, audio_path, effective_audio_path in tmp_results:
            if len(tmp_results) > 1:
                input_audio = effective_audio_path
            else:
                input_audio = audio_cache

            if align_model is not None and len(result["segments"]) > 0:
                if result.get("language", "en") != align_metadata["language"]:
                    logger.info(
                        f"New language found ({result['language']})! Previous was ({align_metadata['language']}), loading new alignment model for new language..."
                    )
                    align_model, align_metadata = load_align_model(
                        result["language"], device
                    )
                logger.info("Performing alignment...")
                result = align(
                    result["segments"],
                    align_model,
                    align_metadata,
                    input_audio,
                    device,
                    interpolate_method=interpolate_method,
                    return_char_alignments=return_char_alignments,
                    print_progress=print_progress,
                )

            results.append((result, audio_path, effective_audio_path))

        del align_model
        gc.collect()
        torch.cuda.empty_cache()

    # >> Diarize
    if diarize:
        if hf_token is None:
            logger.warning(
                "No --hf_token provided, needs to be saved in environment variable, otherwise will throw error loading diarization model"
            )
        tmp_results = results
        logger.info("Performing diarization...")
        logger.info(f"Using model: {diarize_model_name}")
        results = []
        diarize_model = DiarizationPipeline(
            model_name=diarize_model_name,
            use_auth_token=hf_token,
            device=device,
        )
        for result, audio_path, effective_audio_path in tmp_results:
            diarize_result = diarize_model(
                effective_audio_path,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                return_embeddings=return_speaker_embeddings,
            )

            if return_speaker_embeddings:
                diarize_segments, speaker_embeddings = diarize_result
            else:
                diarize_segments = diarize_result
                speaker_embeddings = None

            result = assign_word_speakers(diarize_segments, result, speaker_embeddings)
            results.append((result, audio_path, effective_audio_path))

    output_results: list[dict[str, Any]] = []
    for item in results:
        if len(item) == 3:
            result, audio_path, _ = item
        else:
            result, audio_path = item
        result["language"] = align_language
        writer(result, audio_path, writer_args)
        output_results.append(result)

    if preprocess_temp_dir is not None:
        preprocess_temp_dir.cleanup()

    return output_results


def transcribe_task(args: dict, parser: argparse.ArgumentParser):
    """Transcription task to be called from CLI.

    Args:
        args: Dictionary of command-line arguments.
        parser: argparse.ArgumentParser object.
    """
    try:
        audio_paths = args.get("audio")
        request_args = {k: v for k, v in args.items() if k != "audio"}
        params = TranscribeParams(**request_args)
    except Exception as exc:
        parser.error(str(exc))
        return

    try:
        if not audio_paths:
            parser.error("the following arguments are required: audio")
            return
        run_transcription(params, audio_paths)
    except ValueError as exc:
        parser.error(str(exc))
