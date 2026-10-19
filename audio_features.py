"""Audio loading + feature extraction using only numpy/scipy (no librosa needed)."""
from functools import lru_cache

import numpy as np
from scipy import signal
from scipy.fft import dct, rfft
from scipy.io import wavfile

SR = 16000
N_FFT, WIN, HOP = 512, 400, 160  # 32 ms FFT, 25 ms window, 10 ms hop


def load_audio(path, sr=SR):
    """Load an audio file as mono float32 at `sr`. WAV always works; FLAC/OGG need `soundfile`."""
    try:
        import soundfile as sf
        y, fs = sf.read(path, dtype="float32", always_2d=False)
    except Exception:
        fs, y = wavfile.read(path)
        if y.dtype == np.uint8:
            y = (y.astype(np.float32) - 128.0) / 128.0
        elif y.dtype.kind == "i":
            y = y.astype(np.float32) / float(np.iinfo(y.dtype).max)
        else:
            y = y.astype(np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if fs != sr:
        g = int(np.gcd(int(fs), int(sr)))
        y = signal.resample_poly(y, int(sr) // g, int(fs) // g)
    return np.ascontiguousarray(y, dtype=np.float32)


def trim_silence(y, top_db=35.0, frame=WIN, hop=HOP):
    """Cut leading/trailing silence using a simple energy threshold (a basic VAD)."""
    if len(y) < frame * 2:
        return y
    n = 1 + (len(y) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    e = np.sqrt((y[idx] ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(e / (e.max() + 1e-12) + 1e-12)
    keep = np.where(db > -top_db)[0]
    if len(keep) == 0:
        return y
    return y[keep[0] * hop: keep[-1] * hop + frame]


def crop_or_pad(y, n, train=False, rng=None):
    """Return exactly n samples: random crop (train) / centre crop (eval) / tile if too short."""
    if len(y) == 0:
        return np.zeros(n, np.float32)
    if len(y) >= n:
        start = int(rng.integers(0, len(y) - n + 1)) if (train and rng is not None) else (len(y) - n) // 2
        return y[start:start + n]
    return np.tile(y, int(np.ceil(n / len(y))))[:n]


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


@lru_cache(maxsize=8)
def mel_filterbank(sr=SR, n_fft=N_FFT, n_mels=64, fmin=20.0):
    n_bins = n_fft // 2 + 1
    hz = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(sr / 2), n_mels + 2))
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_mels, n_bins), np.float32)
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        c = max(c, l + 1)
        r = max(r, c + 1)
        for k in range(l, min(c, n_bins)):
            fb[m - 1, k] = (k - l) / (c - l)
        for k in range(c, min(r, n_bins)):
            fb[m - 1, k] = (r - k) / (r - c)
    return fb


def power_spectrogram(y, n_fft=N_FFT, win=WIN, hop=HOP):
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))
    y = np.pad(y, (n_fft // 2, n_fft // 2), mode="reflect")
    n = 1 + (len(y) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    frames = y[idx] * np.hanning(win)
    return (np.abs(rfft(frames, n=n_fft, axis=1)) ** 2).T  # (freq, time)


def log_mel(y, sr=SR, n_mels=64):
    """Log-mel spectrogram, shape (n_mels, frames)."""
    y = np.append(y[0], y[1:] - 0.97 * y[:-1]) if len(y) > 1 else y  # pre-emphasis
    mel = mel_filterbank(sr, N_FFT, n_mels) @ power_spectrogram(y)
    return np.log(mel + 1e-8).astype(np.float32)


def normalize(m):
    return ((m - m.mean()) / (m.std() + 1e-5)).astype(np.float32)


def mfcc(y, sr=SR, n_mfcc=20, n_mels=40):
    return dct(log_mel(y, sr, n_mels), type=2, axis=0, norm="ortho")[:n_mfcc]


def mfcc_stats(y, sr=SR, n_mfcc=20):
    """Utterance-level vector for classical ML: mean+std of MFCC and delta-MFCC (4*n_mfcc dims)."""
    m = mfcc(y, sr, n_mfcc)
    d = np.gradient(m, axis=1)
    return np.concatenate([m.mean(1), m.std(1), d.mean(1), d.std(1)]).astype(np.float32)


# ---- augmentation (training only) ----
def add_noise(y, snr_db, rng):
    p = np.mean(y ** 2) + 1e-12
    noise = rng.standard_normal(len(y)).astype(np.float32) * np.sqrt(p / (10 ** (snr_db / 10)))
    return y + noise


def spec_augment(m, rng, freq_mask=8, time_mask=24):
    """SpecAugment: zero one random frequency band and one random time span."""
    m = m.copy()
    f = int(rng.integers(0, freq_mask + 1))
    f0 = int(rng.integers(0, max(1, m.shape[0] - f)))
    m[f0:f0 + f, :] = 0.0
    t = int(rng.integers(0, time_mask + 1))
    t0 = int(rng.integers(0, max(1, m.shape[1] - t)))
    m[:, t0:t0 + t] = 0.0
    return m
