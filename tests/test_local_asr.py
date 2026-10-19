import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import local_asr


class FakeWaveform:
    def __init__(self, sample_count):
        self.sample_count = sample_count

    def __len__(self):
        return self.sample_count


class LocalASRTests(unittest.TestCase):
    def test_rejects_empty_audio(self):
        with self.assertRaisesRegex(ValueError, "No audio was received"):
            local_asr.transcribe_audio(b"")

    def test_whisper_pipeline_returns_transcript_language_and_mel_metadata(self):
        model = MagicMock()
        model.transcribe.return_value = (
            [SimpleNamespace(text=" Hello"), SimpleNamespace(text=" world ")],
            SimpleNamespace(language="en", language_probability=0.98, duration=2.0),
        )
        fake_audio = types.ModuleType("faster_whisper.audio")
        fake_audio.decode_audio = MagicMock(return_value=FakeWaveform(32000))
        fake_features = types.ModuleType("audio_features")
        fake_features.log_mel = MagicMock(return_value=SimpleNamespace(shape=(80, 198)))
        package = types.ModuleType("faster_whisper")
        package.__path__ = []

        with patch.dict(sys.modules, {
            "faster_whisper": package,
            "faster_whisper.audio": fake_audio,
            "audio_features": fake_features,
        }), patch.object(local_asr, "_get_model", return_value=model), patch.object(
            local_asr, "_mel_preview", return_value=[[0, 255]]
        ):
            result = local_asr.transcribe_audio(b"fake webm", "audio/webm;codecs=opus")

        self.assertEqual(result["text"], "Hello world")
        self.assertEqual(result["language"], "en")
        self.assertEqual(result["languageProbability"], 0.98)
        self.assertEqual(result["logMel"], {"melBands": 80, "frames": 198})
        self.assertEqual(result["melPreview"], [[0, 255]])
        fake_audio.decode_audio.assert_called_once()
        fake_features.log_mel.assert_called_once()
        model.transcribe.assert_called_once()

    def test_rejects_recording_over_duration_limit_before_loading_model(self):
        fake_audio = types.ModuleType("faster_whisper.audio")
        fake_audio.decode_audio = MagicMock(return_value=FakeWaveform(local_asr.MAX_AUDIO_SECONDS * 16000 + 1))
        fake_features = types.ModuleType("audio_features")
        fake_features.log_mel = MagicMock()
        package = types.ModuleType("faster_whisper")
        package.__path__ = []

        with patch.dict(sys.modules, {
            "faster_whisper": package,
            "faster_whisper.audio": fake_audio,
            "audio_features": fake_features,
        }), patch.object(local_asr, "_get_model") as get_model:
            with self.assertRaisesRegex(ValueError, "too long"):
                local_asr.transcribe_audio(b"long recording", "audio/webm")
        get_model.assert_not_called()
        fake_features.log_mel.assert_not_called()


if __name__ == "__main__":
    unittest.main()
