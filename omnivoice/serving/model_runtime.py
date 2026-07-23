"""Lazy singleton loader for the OmniVoice model used by the HTTP layer.

Kept separate from api_server.py so the loading logic can be reused as-is
when this gets wrapped in a Ray Serve deployment (one instance per replica)
instead of a plain in-process singleton.
"""

import os

import torch

from omnivoice.models.omnivoice import OmniVoice
from omnivoice.utils.common import get_best_device

_model: OmniVoice | None = None


def get_model() -> OmniVoice:
    global _model
    if _model is None:
        model_id = os.environ.get("OMNIVOICE_MODEL", "k2-fsa/OmniVoice")
        device = os.environ.get("OMNIVOICE_DEVICE") or get_best_device()
        _model = OmniVoice.from_pretrained(
            model_id, device_map=device, dtype=torch.float16
        )
    return _model
