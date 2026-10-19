"""Local Whisper speech recognition and spoken-language identification."""

import os
import tempfile
import threading
from pathlib import Path


DEFAULT_MODEL = os.environ.get("VOICELENS_WHISPER_MODEL", "base")
MODEL_DEVICE = os.environ.get("VOICELENS_WHISPER_DEVICE", "cpu")
MAX_AUDIO_SECONDS = 180
_model = None
_model_lock = threading.Lock()
_inference_lock = threading.Lock()
_SUPPORTED_AUDIO = {
    "audio/webm": ".webm",
    "audio/ogg": ".ogg",
    "audio/mp4": ".mp4",
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
}


def model_status():
    try:
        from importlib.util import find_spec

        installed = find_spec("faster_whisper") is not None
    except (ImportError, ValueError):
        installed = False
    return {"installed": installed, "loaded": _model is not None, "model": DEFAULT_MODEL, "device": MODEL_DEVICE}


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                try:
                    from faster_whisper import WhisperModel
                except ImportError as exc:
                    raise RuntimeError("Local speech AI is not installed. Run: python -m pip install -r requirements.txt") from exc
                # The model weights download on first use and then remain cached locally.
                try:
                    _model = WhisperModel(DEFAULT_MODEL, device=MODEL_DEVICE, compute_type="int8" if MODEL_DEVICE == "cpu" else "default")
                except Exception as exc:
                    raise RuntimeError(f"Whisper model '{DEFAULT_MODEL}' could not load or download. Check the model name and server network access.") from exc
    return _model


def _mel_preview(mel, max_columns=180):
    """Downsample Log-Mel features into a small, normalized heatmap payload."""
    import numpy as np

    columns = min(max_columns, mel.shape[1])
    indexes = np.linspace(0, mel.shape[1] - 1, columns).astype(int)
    low, high = np.percentile(mel, [5, 95])
    if high <= low:
        high = low + 1.0
    scaled = np.clip((mel[:, indexes] - low) / (high - low), 0, 1)
    return np.rint(scaled[::-1, :].T * 255).astype(np.uint8).tolist()


def transcribe_audio(audio_bytes, content_type="audio/webm"):
    """Run local Whisper ASR/LID and return recognized text plus model metadata."""
    if not audio_bytes:
        raise ValueError("No audio was received. Record a short clip and try again.")
    mime = content_type.split(";", 1)[0].strip().lower()
    suffix = _SUPPORTED_AUDIO.get(mime, ".webm")
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix="voicelens-", suffix=suffix, delete=False) as audio_file:
            audio_file.write(audio_bytes)
            temp_path = Path(audio_file.name)

        # VoiceLens' own signal-processing helper extracts an 80-bin Log-Mel
        # representation for diagnostics; Whisper builds its model-specific
        # encoder input from the same decoded waveform during transcription.
        from faster_whisper.audio import decode_audio
        from audio_features import log_mel

        waveform = decode_audio(str(temp_path), sampling_rate=16000)
        duration = len(waveform) / 16000
        if duration <= 0:
            raise ValueError("No audio was found in this recording.")
        if duration > MAX_AUDIO_SECONDS:
            raise ValueError(f"Recording is too long. Keep clips under {MAX_AUDIO_SECONDS} seconds.")
        mel = log_mel(waveform, sr=16000, n_mels=80)
        model = _get_model()
        with _inference_lock:
            segments, info = model.transcribe(str(temp_path), beam_size=5, vad_filter=True)
            text = " ".join(segment.text.strip() for segment in segments).strip()
        return {
            "text": text,
            "language": info.language,
            "languageProbability": round(float(info.language_probability), 4),
            "duration": round(float(info.duration), 2),
            "logMel": {"melBands": int(mel.shape[0]), "frames": int(mel.shape[1])},
            "melPreview": _mel_preview(mel),
            "model": DEFAULT_MODEL,
            "device": MODEL_DEVICE,
        }
    except (RuntimeError, ValueError):
        raise
    except ImportError as exc:
        raise RuntimeError("Local speech AI dependencies are missing. Run: python -m pip install -r requirements.txt") from exc
    except Exception as exc:
        raise ValueError(f"Could not decode or transcribe this recording: {exc}") from exc
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
