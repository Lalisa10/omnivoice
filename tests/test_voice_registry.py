#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from omnivoice.models.omnivoice import VoiceClonePrompt
from omnivoice.serving import voice_registry


def _prompt(ref_text: str) -> VoiceClonePrompt:
    return VoiceClonePrompt(
        ref_audio_tokens=torch.zeros((8, 2), dtype=torch.long),
        ref_text=ref_text,
        ref_rms=0.1,
    )


class VoiceRegistryTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.voice_dir = Path(self.temp_dir.name)
        self.env = patch.dict(
            os.environ, {"OMNIVOICE_VOICE_DIR": str(self.voice_dir)}
        )
        self.env.start()
        voice_registry.clear_cache()

    def tearDown(self):
        voice_registry.clear_cache()
        self.env.stop()
        self.temp_dir.cleanup()

    def test_get_voice_lazily_loads_one_prompt_after_cache_clear(self):
        voice_registry.register_voice("alice", _prompt("Alice reference"), "English")
        voice_registry.register_voice("bob", _prompt("Bob reference"), "English")
        voice_registry.clear_cache()

        with patch.object(
            VoiceClonePrompt, "load", wraps=VoiceClonePrompt.load
        ) as load:
            voice = voice_registry.get_voice("alice")
            cached_voice = voice_registry.get_voice("alice")

        self.assertEqual(voice.prompt.ref_text, "Alice reference")
        self.assertIs(cached_voice, voice)
        self.assertEqual(load.call_count, 1)
        self.assertTrue(load.call_args.args[0].endswith("alice.pt"))

    def test_get_voice_discovers_a_voice_added_by_another_replica(self):
        voice_registry.register_voice("alice", _prompt("Alice"), "English")
        voice_registry.clear_cache()
        self.assertIsNotNone(voice_registry.get_voice("alice"))

        _prompt("Bob").save(str(self.voice_dir / "bob.pt"))
        (self.voice_dir / "bob.json").write_text(
            json.dumps({"language": "English", "ref_text": "Bob"})
        )

        self.assertEqual(voice_registry.get_voice("bob").prompt.ref_text, "Bob")

    def test_list_voices_reads_metadata_without_loading_prompts(self):
        voice_registry.register_voice("bob", _prompt("Bob reference"), "English")
        voice_registry.register_voice(
            "alice", _prompt("Alice reference"), "Vietnamese"
        )
        voice_registry.clear_cache()

        with patch.object(
            VoiceClonePrompt, "load", side_effect=AssertionError("prompt was loaded")
        ):
            voices = voice_registry.list_voices()

        self.assertEqual([voice.name for voice in voices], ["alice", "bob"])
        self.assertEqual(voices[0].ref_text, "Alice reference")
        self.assertEqual(voices[1].language, "English")

    def test_list_voices_supports_legacy_metadata(self):
        _prompt("Legacy reference").save(str(self.voice_dir / "legacy.pt"))
        (self.voice_dir / "legacy.json").write_text(
            json.dumps({"language": "English"})
        )

        voices = voice_registry.list_voices()

        self.assertEqual(len(voices), 1)
        self.assertEqual(voices[0].name, "legacy")
        self.assertIsNone(voices[0].ref_text)

    def test_missing_backing_file_invalidates_a_cached_voice(self):
        voice_registry.register_voice("alice", _prompt("Reference"), "English")
        self.assertIsNotNone(voice_registry.get_voice("alice"))

        (self.voice_dir / "alice.json").unlink()

        self.assertIsNone(voice_registry.get_voice("alice"))

    def test_delete_removes_files_and_clears_all_cached_voices(self):
        voice_registry.register_voice("alice", _prompt("Alice"), "English")
        voice_registry.register_voice("bob", _prompt("Bob"), "English")

        self.assertTrue(voice_registry.delete_voice("alice"))

        self.assertFalse((self.voice_dir / "alice.pt").exists())
        self.assertFalse((self.voice_dir / "alice.json").exists())
        self.assertEqual(voice_registry._voices, {})
        self.assertIsNone(voice_registry.get_voice("alice"))
        self.assertEqual(voice_registry.get_voice("bob").prompt.ref_text, "Bob")

    def test_delete_missing_voice_returns_false(self):
        self.assertFalse(voice_registry.delete_voice("missing"))

    def test_names_with_trailing_newlines_are_rejected(self):
        with self.assertRaises(ValueError):
            voice_registry.delete_voice("alice\n")

        self.assertIsNone(voice_registry.get_voice("alice\n"))

    def test_refresh_clears_all_cached_voices_but_keeps_files(self):
        voice_registry.register_voice("alice", _prompt("Alice"), "English")
        voice_registry.register_voice("bob", _prompt("Bob"), "English")

        voice_registry.refresh_cache()

        self.assertEqual(voice_registry._voices, {})
        self.assertTrue((self.voice_dir / "alice.pt").exists())
        self.assertTrue((self.voice_dir / "bob.pt").exists())
        self.assertEqual(voice_registry.get_voice("alice").prompt.ref_text, "Alice")

    def test_shared_refresh_marker_invalidates_another_replica_cache(self):
        voice_registry.register_voice("alice", _prompt("Alice"), "English")
        cached_alice = voice_registry.get_voice("alice")

        (self.voice_dir / ".cache-generation").write_text("another-replica")
        reloaded_alice = voice_registry.get_voice("alice")

        self.assertIsNot(reloaded_alice, cached_alice)
        self.assertEqual(reloaded_alice.prompt.ref_text, "Alice")


if __name__ == "__main__":
    unittest.main()
