import unittest

import numpy as np

from audio_features import log_mel, mfcc_stats
from local_asr import _mel_preview


class AudioFeatureTests(unittest.TestCase):
    def test_log_mel_and_mfcc_are_finite_with_expected_dimensions(self):
        sample_rate = 16000
        time = np.arange(sample_rate, dtype=np.float32) / sample_rate
        waveform = 0.2 * np.sin(2 * np.pi * 220 * time)
        mel = log_mel(waveform, sr=sample_rate, n_mels=80)
        stats = mfcc_stats(waveform, sr=sample_rate, n_mfcc=20)
        self.assertEqual(mel.shape[0], 80)
        self.assertGreater(mel.shape[1], 1)
        self.assertEqual(stats.shape, (80,))
        self.assertTrue(np.isfinite(mel).all())
        self.assertTrue(np.isfinite(stats).all())

    def test_heatmap_preview_is_bounded_to_byte_values(self):
        mel = np.arange(80 * 240, dtype=np.float32).reshape(80, 240)
        preview = _mel_preview(mel)
        self.assertEqual(len(preview), 180)
        self.assertEqual(len(preview[0]), 80)
        self.assertGreaterEqual(min(min(column) for column in preview), 0)
        self.assertLessEqual(max(max(column) for column in preview), 255)


if __name__ == "__main__":
    unittest.main()
