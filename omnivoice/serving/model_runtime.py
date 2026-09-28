"""Lazy singleton loader for the OmniVoice model used by the HTTP layer.

Kept separate from api_server.py so the loading logic can be reused as-is
by the Ray Serve GPU deployment. Each replica has its own process and therefore
its own singleton, so the model is loaded exactly once per replica.
"""

import os
import logging

import torch

from omnivoice.models.omnivoice import OmniVoice
from omnivoice.utils.common import get_best_device, str2bool

_model: OmniVoice | None = None
logger = logging.getLogger(__name__)


def get_model() -> OmniVoice:
    global _model
    if _model is None:
        model_id = os.environ.get("OMNIVOICE_MODEL", "k2-fsa/OmniVoice")
        device = os.environ.get("OMNIVOICE_DEVICE") or get_best_device()
        # ASR is opt-in and loaded here, at replica startup, so a voice clone
        # without ref_text never triggers a download mid-request.
        load_asr = str2bool(os.environ.get("OMNIVOICE_LOAD_ASR", "false"))
        # May be a local directory, which is how an air-gapped cluster
        # preloads Whisper from an NFS mount instead of the Hugging Face hub.
        asr_model_name = os.environ.get("OMNIVOICE_ASR_MODEL")
        _model = OmniVoice.from_pretrained(
            model_id,
            device_map=device,
            dtype=torch.float16,
            load_asr=load_asr,
            asr_model_name=asr_model_name,
        )
        if str2bool(os.environ.get("OMNIVOICE_ENABLE_FLASHINFER", "false")):
            if not str(device).startswith("cuda"):
                raise RuntimeError(
                    "OMNIVOICE_ENABLE_FLASHINFER requires an NVIDIA CUDA device."
                )
            try:
                from omnivoice.models.omnivoice_flashinfer import apply_flashinfer
            except ImportError as exc:
                raise RuntimeError(
                    "OMNIVOICE_ENABLE_FLASHINFER=1 requires flashinfer-python. "
                    "Build the serving image with INSTALL_FLASHINFER=1 or install "
                    "the FlashInfer packages documented in README.md."
                ) from exc

            enable_cuda_graph = str2bool(
                os.environ.get("OMNIVOICE_FLASHINFER_CUDA_GRAPH", "false")
            )
            apply_flashinfer(_model, enable_cuda_graph=enable_cuda_graph)
            logger.info(
                "FlashInfer acceleration enabled (cuda_graph=%s)", enable_cuda_graph
            )
    return _model
