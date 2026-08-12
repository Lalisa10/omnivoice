#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Ray Serve application for the OmniVoice HTTP API."""

import asyncio
import io
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import librosa
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from ray import serve
from ray.serve.handle import DeploymentHandle
from ray.util.metrics import Counter, Histogram

from omnivoice.models.omnivoice import VoiceClonePrompt
from omnivoice.serving import voice_registry
from omnivoice.serving.model_runtime import get_model
from omnivoice.serving.schemas import TTSRequest, VoiceDto, VoicesResponse

logger = logging.getLogger(__name__)

app = FastAPI(title="OmniVoice serving with Ray Serve")

# response_format -> (soundfile format, media type). Other OpenAI-compatible
# formats need a transcoder and are intentionally rejected for now.
_SUPPORTED_FORMATS = {
    "wav": ("WAV", "audio/wav"),
}


@dataclass
class SynthesisInput:
    """Internal request sent from the HTTP ingress to the GPU deployment."""

    text: str
    language: str | None
    instruct: str | None
    voice_clone_prompt: VoiceClonePrompt | None
    duration: float | None
    speed: float | None
    num_step: int
    guidance_scale: float
    t_shift: float
    denoise: bool
    postprocess_output: bool
    layer_penalty_factor: float
    position_temperature: float
    class_temperature: float
    audio_chunk_duration: float
    audio_chunk_threshold: float
    pad_duration: float
    fade_duration: float
    normalize_text: bool

    def compatibility_key(self) -> tuple[Any, ...]:
        """Return fields that must be scalar in one ``generate()`` call."""
        return (
            self.voice_clone_prompt is not None,
            self.num_step,
            self.guidance_scale,
            self.t_shift,
            self.denoise,
            self.postprocess_output,
            self.layer_penalty_factor,
            self.position_temperature,
            self.class_temperature,
            self.audio_chunk_duration,
            self.audio_chunk_threshold,
            self.pad_duration,
            self.fade_duration,
            self.normalize_text,
        )


@dataclass
class SynthesisResult:
    audio: np.ndarray | None = None
    sample_rate: int | None = None
    error: str | None = None
    retryable: bool = False


def _to_synthesis_input(
    request: TTSRequest, voice_clone_prompt: Any = None
) -> SynthesisInput:
    """Convert the public request schema into a serializable model request."""
    return SynthesisInput(
        text=request.input,
        language=request.language,
        instruct=request.instructions,
        voice_clone_prompt=voice_clone_prompt,
        duration=request.duration,
        speed=request.speed,
        num_step=request.num_step,
        guidance_scale=request.guidance_scale,
        t_shift=request.t_shift,
        denoise=request.denoise,
        postprocess_output=request.postprocess_output,
        layer_penalty_factor=request.layer_penalty_factor,
        position_temperature=request.position_temperature,
        class_temperature=request.class_temperature,
        audio_chunk_duration=request.audio_chunk_duration,
        audio_chunk_threshold=request.audio_chunk_threshold,
        pad_duration=request.pad_duration,
        fade_duration=request.fade_duration,
        normalize_text=request.normalize_text,
    )


def _is_cuda_oom(error: BaseException) -> bool:
    return isinstance(error, torch.OutOfMemoryError) or (
        isinstance(error, RuntimeError)
        and "out of memory" in str(error).lower()
        and "cuda" in str(error).lower()
    )


@serve.deployment(
    name="SpeechModel",
    ray_actor_options={"num_cpus": 1, "num_gpus": 1},
    max_ongoing_requests=8,
    user_config={"max_batch_size": 2, "batch_wait_timeout_s": 0.02},
)
class SpeechModel:
    """Own one model/GPU and batch compatible synthesis requests."""

    def __init__(self):
        # Load eagerly so the deployment is not healthy until its model is
        # actually resident on the GPU.
        self.model = get_model()
        self._model_lock = threading.Lock()
        self._batch_items = Counter(
            "omnivoice_batch_items_total",
            description="Synthesis items processed by the GPU deployment.",
        )
        self._sub_batches = Counter(
            "omnivoice_sub_batches_total",
            description="Compatible model.generate calls executed.",
        )
        self._fallbacks = Counter(
            "omnivoice_batch_fallbacks_total",
            description="Batches retried item by item.",
            tag_keys=("reason",),
        )
        self._batch_size = Histogram(
            "omnivoice_batch_size",
            description="Number of HTTP requests received in a dynamic batch.",
            boundaries=[1, 2, 4, 8, 16],
        )
        self._inference_seconds = Histogram(
            "omnivoice_inference_seconds",
            description="Wall time for one compatible model.generate call.",
            boundaries=[0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120],
        )

    def reconfigure(self, config: dict[str, Any]) -> None:
        max_batch_size = int(config.get("max_batch_size", 2))
        batch_wait_timeout_s = float(config.get("batch_wait_timeout_s", 0.02))
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if batch_wait_timeout_s < 0:
            raise ValueError("batch_wait_timeout_s must be >= 0")
        self.generate.set_max_batch_size(max_batch_size)
        self.generate.set_batch_wait_timeout_s(batch_wait_timeout_s)

    def _generate_compatible(self, items: list[SynthesisInput]) -> list[np.ndarray]:
        first = items[0]
        started = time.perf_counter()
        self._sub_batches.inc()
        try:
            return self.model.generate(
                text=[item.text for item in items],
                language=[item.language for item in items],
                instruct=[item.instruct for item in items],
                voice_clone_prompt=[item.voice_clone_prompt for item in items]
                if first.voice_clone_prompt is not None
                else None,
                duration=[item.duration for item in items],
                speed=[item.speed for item in items],
                num_step=first.num_step,
                guidance_scale=first.guidance_scale,
                t_shift=first.t_shift,
                denoise=first.denoise,
                postprocess_output=first.postprocess_output,
                layer_penalty_factor=first.layer_penalty_factor,
                position_temperature=first.position_temperature,
                class_temperature=first.class_temperature,
                audio_chunk_duration=first.audio_chunk_duration,
                audio_chunk_threshold=first.audio_chunk_threshold,
                pad_duration=first.pad_duration,
                fade_duration=first.fade_duration,
                normalize_text=first.normalize_text,
            )
        finally:
            self._inference_seconds.observe(time.perf_counter() - started)

    def _run_sub_batch(
        self,
        indexed_items: list[tuple[int, SynthesisInput]],
        results: list[SynthesisResult | None],
    ) -> None:
        items = [item for _, item in indexed_items]
        try:
            audios = self._generate_compatible(items)
            if len(audios) != len(items):
                raise RuntimeError(
                    f"generate returned {len(audios)} outputs for {len(items)} inputs"
                )
            for (index, _), audio in zip(indexed_items, audios):
                results[index] = SynthesisResult(
                    audio=audio, sample_rate=self.model.sampling_rate
                )
            return
        except Exception as error:
            if len(items) == 1:
                retryable = _is_cuda_oom(error)
                if retryable and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                results[indexed_items[0][0]] = SynthesisResult(
                    error=str(error), retryable=retryable
                )
                logger.exception("Single-item synthesis failed")
                return

            reason = "oom" if _is_cuda_oom(error) else "item_isolation"
            self._fallbacks.inc(tags={"reason": reason})
            logger.warning(
                "Batch of %d failed (%s); retrying each item", len(items), reason
            )
            if reason == "oom" and torch.cuda.is_available():
                torch.cuda.empty_cache()

        for indexed_item in indexed_items:
            self._run_sub_batch([indexed_item], results)

    def _run_batch_sync(self, items: list[SynthesisInput]) -> list[SynthesisResult]:
        self._batch_items.inc(len(items))
        self._batch_size.observe(len(items))
        grouped: dict[tuple[Any, ...], list[tuple[int, SynthesisInput]]] = defaultdict(
            list
        )
        for index, item in enumerate(items):
            grouped[item.compatibility_key()].append((index, item))

        logger.info(
            "Processing dynamic batch: items=%d compatible_sub_batches=%d",
            len(items),
            len(grouped),
        )
        results: list[SynthesisResult | None] = [None] * len(items)
        with self._model_lock:
            for indexed_items in grouped.values():
                self._run_sub_batch(indexed_items, results)

        if any(result is None for result in results):
            raise RuntimeError("Internal batching error: a request has no result")
        return [result for result in results if result is not None]

    @serve.batch(
        max_batch_size=2,
        batch_wait_timeout_s=0.02,
        max_concurrent_batches=1,
    )
    async def generate(self, items: list[SynthesisInput]) -> list[SynthesisResult]:
        return await asyncio.to_thread(self._run_batch_sync, items)

    async def create_voice_clone_prompt(
        self, ref_audio: tuple[np.ndarray, int], ref_text: str | None
    ) -> VoiceClonePrompt:
        def create_prompt():
            with self._model_lock:
                prompt = self.model.create_voice_clone_prompt(
                    ref_audio=ref_audio, ref_text=ref_text
                )
                # Ray must not serialize a CUDA tensor back to the CPU ingress.
                # generate() moves the tokens to the model device when reused.
                return VoiceClonePrompt(
                    ref_audio_tokens=prompt.ref_audio_tokens.detach().cpu(),
                    ref_text=prompt.ref_text,
                    ref_rms=prompt.ref_rms,
                )

        return await asyncio.to_thread(create_prompt)

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "model_loaded": self.model is not None,
            "sample_rate": self.model.sampling_rate,
        }


def _encode_audio_response(
    audio: np.ndarray,
    native_sample_rate: int,
    requested_sample_rate: int | None,
    response_format: str,
) -> Response:
    sample_rate = native_sample_rate
    if requested_sample_rate is not None and requested_sample_rate != sample_rate:
        audio = librosa.resample(
            audio, orig_sr=sample_rate, target_sr=requested_sample_rate
        )
        sample_rate = requested_sample_rate

    sf_format, media_type = _SUPPORTED_FORMATS[response_format]
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format=sf_format)
    return Response(content=buffer.getvalue(), media_type=media_type)


@serve.deployment(
    name="OmniVoiceIngress",
    ray_actor_options={"num_cpus": 1},
    max_ongoing_requests=64,
)
@serve.ingress(app)
class OmniVoiceIngress:
    """CPU-side HTTP ingress; all GPU work is delegated to ``SpeechModel``."""

    def __init__(self, speech_model: DeploymentHandle):
        self.speech_model = speech_model

    @app.get("/healthz")
    async def health(self):
        try:
            return await self.speech_model.health.remote()
        except Exception as error:
            raise HTTPException(status_code=503, detail="model is not ready") from error

    @app.post("/v1/audio/speech")
    async def text_to_speech(self, request: TTSRequest):
        if request.response_format not in _SUPPORTED_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"response_format={request.response_format!r} not supported yet, "
                    f"only {list(_SUPPORTED_FORMATS)} are implemented"
                ),
            )

        voice_clone_prompt = None
        if request.ref_voice is not None:
            registered = voice_registry.get_voice(request.ref_voice)
            if registered is None:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"voice {request.ref_voice!r} not found, clone it via "
                        "POST /v1/voices first"
                    ),
                )
            voice_clone_prompt = registered.prompt

        result = await self.speech_model.generate.remote(
            _to_synthesis_input(request, voice_clone_prompt)
        )
        if result.error is not None:
            raise HTTPException(
                status_code=503 if result.retryable else 500,
                detail=f"speech synthesis failed: {result.error}",
            )

        return await asyncio.to_thread(
            _encode_audio_response,
            result.audio,
            result.sample_rate,
            request.sample_rate,
            request.response_format,
        )

    @app.get("/v1/voices")
    def get_voices(self):
        return VoicesResponse(
            voices=[
                VoiceDto(name=v.name, ref_text=v.prompt.ref_text, language=v.language)
                for v in voice_registry.list_voices()
            ]
        )

    @app.post("/v1/voices")
    async def clone_voice(
        self,
        name: str = Form(..., description="Unique name to register the voice under."),
        ref_audio: UploadFile = File(..., description="Reference audio file."),
        ref_text: str | None = Form(
            None,
            description="Transcript of ref_audio. Auto-transcribed via ASR if omitted.",
        ),
        ref_language: str | None = Form("Vietnamese"),
    ):
        try:
            voice_registry.validate_name(name)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        if voice_registry.get_voice(name) is not None:
            raise HTTPException(status_code=409, detail=f"voice {name!r} already exists")

        audio_bytes = await ref_audio.read()
        try:
            waveform, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        except Exception as error:
            raise HTTPException(
                status_code=400, detail=f"could not decode ref_audio: {error}"
            ) from error
        if waveform.ndim == 2:
            waveform = waveform.T

        prompt = await self.speech_model.create_voice_clone_prompt.remote(
            (waveform, sample_rate), ref_text
        )
        registered = voice_registry.register_voice(name, prompt, ref_language)
        return VoiceDto(
            name=registered.name,
            ref_text=prompt.ref_text,
            language=registered.language,
        )
application = OmniVoiceIngress.bind(SpeechModel.bind())
