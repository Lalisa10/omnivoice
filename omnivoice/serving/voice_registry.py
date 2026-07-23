"""Disk-backed registry for cloned voices.

Each registered voice is stored as two files under ``OMNIVOICE_VOICE_DIR``:
``{name}.pt`` (the ``VoiceClonePrompt``, via its own ``save``/``load``) and
``{name}.json`` (small metadata not part of ``VoiceClonePrompt``, e.g.
``language``). This lets voices survive a process restart and be shared by
copying the directory, instead of being lost the moment the server exits.

This is still file-based, not a shared service: with more than one Ray
Serve replica on different machines, they'd each need access to the same
``OMNIVOICE_VOICE_DIR`` (e.g. a shared/network filesystem). Fine for the
current single-machine setup.
"""

import json
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from omnivoice.models.omnivoice import VoiceClonePrompt

# Keeps names filesystem-safe and rules out path traversal (e.g. "../../etc").
_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")

_lock = threading.Lock()
_voices: dict[str, "RegisteredVoice"] = {}
_loaded = False


@dataclass
class RegisteredVoice:
    name: str
    prompt: VoiceClonePrompt
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


def validate_name(name: str) -> None:
    if not _NAME_RE.match(name):
        raise ValueError(
            f"invalid voice name {name!r}: only letters, digits, '_' and '-' are allowed"
        )


def _load_from_disk() -> None:
    global _loaded
    if _loaded:
        return
    for meta_file in _voice_dir().glob("*.json"):
        name = meta_file.stem
        prompt_file = _prompt_path(name)
        if not prompt_file.exists():
            continue
        try:
            meta = json.loads(meta_file.read_text())
            prompt = VoiceClonePrompt.load(str(prompt_file))
        except Exception:
            continue
        _voices[name] = RegisteredVoice(
            name=name, prompt=prompt, language=meta.get("language", "Vietnamese")
        )
    _loaded = True


def register_voice(name: str, prompt: VoiceClonePrompt, language: str) -> RegisteredVoice:
    validate_name(name)
    entry = RegisteredVoice(name=name, prompt=prompt, language=language)
    with _lock:
        _load_from_disk()
        prompt.save(str(_prompt_path(name)))
        _meta_path(name).write_text(json.dumps({"language": language}))
        _voices[name] = entry
    return entry


def get_voice(name: str) -> Optional[RegisteredVoice]:
    with _lock:
        _load_from_disk()
        return _voices.get(name)


def list_voices() -> list[RegisteredVoice]:
    with _lock:
        _load_from_disk()
        return list(_voices.values())
