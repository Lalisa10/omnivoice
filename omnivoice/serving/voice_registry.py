"""Disk-backed registry for cloned voices.

Each registered voice is stored as two files under ``OMNIVOICE_VOICE_DIR``:
``{name}.pt`` (the ``VoiceClonePrompt``, via its own ``save``/``load``) and
``{name}.json`` (small metadata not part of ``VoiceClonePrompt``). This lets
voices survive a process restart and be shared by replicas that mount the
same ``OMNIVOICE_VOICE_DIR`` (for example, an RWX volume).

Prompts are loaded lazily by name. The in-process dictionary is only a cache;
the shared directory remains the source of truth so additions and deletions
made by another replica become visible without restarting this process.
"""

import json
import os
import re
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from omnivoice.models.omnivoice import VoiceClonePrompt

# Keeps names filesystem-safe and rules out path traversal (e.g. "../../etc").
_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")

_lock = threading.Lock()
_voices: dict[str, "RegisteredVoice"] = {}
_cache_generation: Optional[str] = None


@dataclass
class RegisteredVoice:
    name: str
    prompt: VoiceClonePrompt
    language: str


@dataclass
class VoiceMetadata:
    name: str
    ref_text: Optional[str]
    language: str


def _voice_dir() -> Path:
    path = Path(
        os.environ.get("OMNIVOICE_VOICE_DIR", str(Path.home() / ".cache" / "omnivoice" / "voices"))
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _prompt_path(name: str) -> Path:
    return _voice_dir() / f"{name}.pt"


def _meta_path(name: str) -> Path:
    return _voice_dir() / f"{name}.json"


def _cache_generation_path() -> Path:
    return _voice_dir() / ".cache-generation"


def _sync_cache_generation() -> None:
    """Clear the local cache when another replica requested a refresh.

    The caller must hold ``_lock``.
    """
    global _cache_generation
    try:
        generation = _cache_generation_path().read_text()
    except FileNotFoundError:
        generation = None

    if generation != _cache_generation:
        _voices.clear()
        _cache_generation = generation


def _refresh_cache_locked() -> None:
    """Refresh every replica's cache while the caller holds ``_lock``."""
    global _cache_generation
    generation = uuid.uuid4().hex
    _cache_generation_path().write_text(generation)
    _voices.clear()
    _cache_generation = generation


def validate_name(name: str) -> None:
    if not _NAME_RE.fullmatch(name):
        raise ValueError(
            f"invalid voice name {name!r}: only letters, digits, '_' and '-' are allowed"
        )


def _load_voice(name: str) -> Optional[RegisteredVoice]:
    """Load one voice from the shared directory."""
    prompt_file = _prompt_path(name)
    meta_file = _meta_path(name)
    if not prompt_file.exists() or not meta_file.exists():
        return None

    try:
        meta = json.loads(meta_file.read_text())
        prompt = VoiceClonePrompt.load(str(prompt_file))
    except Exception:
        return None

    return RegisteredVoice(
        name=name,
        prompt=prompt,
        language=meta.get("language", "Vietnamese"),
    )


def register_voice(name: str, prompt: VoiceClonePrompt, language: str) -> RegisteredVoice:
    validate_name(name)
    entry = RegisteredVoice(name=name, prompt=prompt, language=language)
    with _lock:
        _sync_cache_generation()
        prompt.save(str(_prompt_path(name)))
        _meta_path(name).write_text(
            json.dumps({"language": language, "ref_text": prompt.ref_text})
        )
        _voices[name] = entry
    return entry


def get_voice(name: str) -> Optional[RegisteredVoice]:
    # Avoid using an untrusted name to construct a path. Public callers treat
    # invalid and unknown names identically as a cache miss.
    if not _NAME_RE.fullmatch(name):
        return None

    with _lock:
        _sync_cache_generation()
        prompt_file = _prompt_path(name)
        meta_file = _meta_path(name)

        # The backing files are the source of truth. This invalidates stale
        # caches in replicas that did not handle DELETE.
        if not prompt_file.exists() or not meta_file.exists():
            _voices.pop(name, None)
            return None

        cached = _voices.get(name)
        if cached is not None:
            return cached

        voice = _load_voice(name)
        if voice is not None:
            _voices[name] = voice
        return voice


def list_voices() -> list[VoiceMetadata]:
    """List shared-volume metadata without loading prompt tensors."""
    result = []
    with _lock:
        for meta_file in sorted(_voice_dir().glob("*.json")):
            name = meta_file.stem
            if not _NAME_RE.fullmatch(name) or not _prompt_path(name).exists():
                continue
            try:
                meta = json.loads(meta_file.read_text())
            except Exception:
                continue
            result.append(
                VoiceMetadata(
                    name=name,
                    ref_text=meta.get("ref_text"),
                    language=meta.get("language", "Vietnamese"),
                )
            )
    return result


def delete_voice(name: str) -> bool:
    """Delete a voice from shared storage and refresh all voice caches."""
    validate_name(name)
    with _lock:
        _sync_cache_generation()
        prompt_file = _prompt_path(name)
        meta_file = _meta_path(name)
        existed = name in _voices or prompt_file.exists() or meta_file.exists()

        # Remove metadata first so list/get stop advertising the voice before
        # the larger prompt file is removed.
        meta_file.unlink(missing_ok=True)
        prompt_file.unlink(missing_ok=True)
        _refresh_cache_locked()
        return existed


def clear_cache() -> None:
    """Clear prompts cached by this process without touching shared storage."""
    global _cache_generation
    with _lock:
        _voices.clear()
        try:
            _cache_generation = _cache_generation_path().read_text()
        except FileNotFoundError:
            _cache_generation = None


def refresh_cache() -> None:
    """Clear all cached prompts and request the same on every replica."""
    with _lock:
        _refresh_cache_locked()
