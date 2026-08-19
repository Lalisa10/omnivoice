#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import asyncio
import io
import os
import threading
import time
import unittest
import wave
from unittest.mock import patch

import numpy as np
import torch
from fastapi import HTTPException
from pydantic import ValidationError
from ray import cloudpickle

from omnivoice.models.omnivoice import VoiceClonePrompt
from omnivoice.serving import voice_registry
from omnivoice.serving.api_server import (
    OmniVoiceIngress,
    SpeechModel,
    _encode_audio_response,
    _to_synthesis_input,
)
from omnivoice.serving.schemas import TTSRequest


class _Metric:
    def inc(self, *args, **kwargs):
        pass

    def observe(self, *args, **kwargs):
        pass


class _FakeModel:
    sampling_rate = 24000
    _asr_pipe = None

    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if any(text == "bad" for text in kwargs["text"]):
            raise ValueError("bad input")
        return [
            np.full(index + 1, len(text), dtype=np.float32)
            for index, text in enumerate(kwargs["text"])
        ]


class _BatchOomModel(_FakeModel):
    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if len(kwargs["text"]) > 1:
            raise torch.OutOfMemoryError("CUDA out of memory")
        return [np.ones(1, dtype=np.float32)]


class _AlwaysOomModel(_FakeModel):
    def generate(self, **kwargs):
        self.calls.append(kwargs)
        raise torch.OutOfMemoryError("CUDA out of memory")


class _FakeUploadFile:
    """Minimal UploadFile stand-in that records whether it was read."""

    def __init__(self):
        self.reads = 0

    async def read(self):
        self.reads += 1
        return b""


def _item(text: str, **overrides):
    values = {"input": text, **overrides}
    return _to_synthesis_input(TTSRequest(**values))


def _aged_item(text: str, age_s: float, **overrides):
    item = _item(text, **overrides)
    item.enqueued_at = time.time() - age_s
    return item


def _worker(model=None):
    cls = SpeechModel.func_or_class
    worker = object.__new__(cls)
    worker.model = model or _FakeModel()
    worker._model_lock = threading.Lock()
    worker._queue_timeout_s = 300.0
    worker._batch_items = _Metric()
    worker._sub_batches = _Metric()
    worker._fallbacks = _Metric()
    worker._batch_size = _Metric()
    worker._inference_seconds = _Metric()
    return worker


class ServingBatchingTest(unittest.TestCase):
    def test_serialized_ingress_contains_audio_encoder(self):
        ingress_class = cloudpickle.loads(
            cloudpickle.dumps(OmniVoiceIngress.func_or_class)
        )

        self.assertIn(
            "_encode_audio_response",
            ingress_class.text_to_speech.__globals__,
        )

    def test_delete_voice_endpoint_returns_no_content(self):
        with patch.object(voice_registry, "delete_voice", return_value=True):
            response = OmniVoiceIngress.func_or_class.delete_voice(None, "alice")

        self.assertEqual(response.status_code, 204)

    def test_delete_voice_endpoint_returns_not_found(self):
        with patch.object(voice_registry, "delete_voice", return_value=False):
            with self.assertRaises(HTTPException) as raised:
                OmniVoiceIngress.func_or_class.delete_voice(None, "missing")

        self.assertEqual(raised.exception.status_code, 404)

    def test_refresh_voices_endpoint_returns_no_content(self):
        with patch.object(voice_registry, "refresh_cache") as refresh:
            response = OmniVoiceIngress.func_or_class.refresh_voices(None)

        refresh.assert_called_once_with()
        self.assertEqual(response.status_code, 204)

    def test_compatible_requests_use_one_generate_call(self):
        worker = _worker()

        results = worker._run_batch_sync([_item("one"), _item("two")])

        self.assertEqual(len(worker.model.calls), 1)
        self.assertEqual(worker.model.calls[0]["text"], ["one", "two"])
        self.assertEqual(worker.model.calls[0]["language"], ["Vietnamese", "Vietnamese"])
        self.assertEqual(len(results), 2)
        self.assertIsNone(results[0].error)
        self.assertEqual(results[1].sample_rate, 24000)

    def test_incompatible_generation_configs_are_split_and_order_is_kept(self):
        worker = _worker()

        results = worker._run_batch_sync(
            [_item("first", num_step=16), _item("middle"), _item("last", num_step=16)]
        )

        self.assertEqual(len(worker.model.calls), 2)
        self.assertEqual(worker.model.calls[0]["text"], ["first", "last"])
        self.assertEqual(worker.model.calls[1]["text"], ["middle"])
        self.assertEqual([len(result.audio) for result in results], [1, 1, 2])

    def test_per_item_values_are_forwarded_as_lists(self):
        worker = _worker()

        worker._run_batch_sync(
            [
                _item("one", language="English", speed=1.2),
                _item("two", language="Vietnamese", duration=2.5),
            ]
        )

        call = worker.model.calls[0]
        self.assertEqual(call["language"], ["English", "Vietnamese"])
        self.assertEqual(call["speed"], [1.2, 1.0])
        self.assertEqual(call["duration"], [None, 2.5])

    def test_failed_batch_is_retried_per_item(self):
        worker = _worker()

        results = worker._run_batch_sync([_item("good"), _item("bad")])

        self.assertEqual(
            [call["text"] for call in worker.model.calls],
            [["good", "bad"], ["good"], ["bad"]],
        )
        self.assertIsNone(results[0].error)
        self.assertEqual(results[1].error, "bad input")
        self.assertFalse(results[1].retryable)

    def test_cuda_oom_batch_falls_back_to_single_items(self):
        worker = _worker(_BatchOomModel())

        results = worker._run_batch_sync([_item("one"), _item("two")])

        self.assertEqual(
            [call["text"] for call in worker.model.calls],
            [["one", "two"], ["one"], ["two"]],
        )
        self.assertTrue(all(result.error is None for result in results))

    def test_single_item_cuda_oom_is_retryable(self):
        worker = _worker(_AlwaysOomModel())

        with self.assertLogs("omnivoice.serving.api_server", level="ERROR"):
            result = worker._run_batch_sync([_item("one")])[0]

        self.assertIn("out of memory", result.error)
        self.assertTrue(result.retryable)

    def test_clone_and_non_clone_requests_are_not_mixed(self):
        worker = _worker()
        prompt = VoiceClonePrompt(
            ref_audio_tokens=torch.zeros((8, 2), dtype=torch.long),
            ref_text="reference",
            ref_rms=0.1,
        )
        clone_item = _to_synthesis_input(TTSRequest(input="clone"), prompt)

        worker._run_batch_sync([_item("auto"), clone_item])

        self.assertEqual(len(worker.model.calls), 2)
        self.assertIsNone(worker.model.calls[0]["voice_clone_prompt"])
        self.assertEqual(worker.model.calls[1]["voice_clone_prompt"], [prompt])

    def test_out_of_range_request_fields_are_rejected(self):
        for field, value in (
            ("duration", 301),
            ("num_step", 129),
            ("speed", 0.1),
            ("input", "a" * 5001),
        ):
            with self.subTest(field=field):
                with self.assertRaises(ValidationError):
                    TTSRequest(**{"input": "hello", field: value})

    def test_in_range_request_is_accepted(self):
        request = TTSRequest(input="hello", duration=300, num_step=128, speed=4.0)

        self.assertEqual(request.duration, 300)
        self.assertEqual(request.num_step, 128)
        self.assertEqual(request.speed, 4.0)

    def test_expired_item_is_shed_before_reaching_the_model(self):
        worker = _worker()

        with self.assertLogs("omnivoice.serving.api_server", level="WARNING"):
            results = worker._run_batch_sync([_aged_item("old", 600), _item("fresh")])

        self.assertEqual([call["text"] for call in worker.model.calls], [["fresh"]])
        self.assertIn("timed out", results[0].error)
        self.assertTrue(results[0].retryable)
        self.assertIsNone(results[1].error)

    def test_queue_timeout_none_disables_shedding(self):
        worker = _worker()
        worker._queue_timeout_s = None

        results = worker._run_batch_sync([_aged_item("old", 600)])

        self.assertEqual([call["text"] for call in worker.model.calls], [["old"]])
        self.assertIsNone(results[0].error)

    def test_reconfigure_applies_queue_timeout(self):
        worker = _worker()

        worker.reconfigure({"queue_timeout_s": 42})
        self.assertEqual(worker._queue_timeout_s, 42.0)

        worker.reconfigure({"queue_timeout_s": None})
        self.assertIsNone(worker._queue_timeout_s)

        with self.assertRaises(ValueError):
            worker.reconfigure({"queue_timeout_s": -1})

    def test_voice_clone_prompt_without_ref_text_and_asr_fails_fast(self):
        worker = _worker()

        with self.assertRaises(ValueError) as raised:
            asyncio.run(
                worker.create_voice_clone_prompt(
                    (np.zeros(2400, dtype=np.float32), 24000), None
                )
            )

        self.assertIn("ref_text is required", str(raised.exception))

    def test_clone_voice_without_ref_text_returns_400_when_asr_disabled(self):
        upload = _FakeUploadFile()

        with patch.dict(os.environ, {"OMNIVOICE_LOAD_ASR": "0"}):
            with self.assertRaises(HTTPException) as raised:
                asyncio.run(
                    OmniVoiceIngress.func_or_class.clone_voice(
                        None, name="alice", ref_audio=upload, ref_text=None
                    )
                )

        self.assertEqual(raised.exception.status_code, 400)
        self.assertIn("OMNIVOICE_LOAD_ASR", raised.exception.detail)
        self.assertEqual(upload.reads, 0)

    def test_wav_encoding_and_per_request_resampling(self):
        response = _encode_audio_response(
            np.zeros(2400, dtype=np.float32), 24000, 16000, "wav"
        )

        with wave.open(io.BytesIO(response.body), "rb") as wav:
            self.assertEqual(wav.getframerate(), 16000)
            self.assertAlmostEqual(
                wav.getnframes() / wav.getframerate(), 0.1, places=2
            )


if __name__ == "__main__":
    unittest.main()
