"""VoiceLens web application and local speech inference API."""

import logging
import hmac
import os
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge

from local_asr import MAX_AUDIO_SECONDS, model_status, transcribe_audio


PROJECT_DIR = Path(__file__).resolve().parent
MAX_AUDIO_BYTES = 30 * 1024 * 1024
ALLOWED_AUDIO_TYPES = {"audio/webm", "audio/ogg", "audio/mp4", "audio/mpeg", "audio/wav", "audio/x-wav"}

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = MAX_AUDIO_BYTES
API_TOKEN = os.environ.get("VOICELENS_API_TOKEN", "")


@app.get("/")
@app.get("/VoiceLens.html")
def index():
    return send_from_directory(PROJECT_DIR, "VoiceLens.html")


@app.get("/api/health")
def health():
    return jsonify({
        "ok": True,
        "speechModel": model_status(),
        "authenticationRequired": bool(API_TOKEN),
        "limits": {"maxAudioBytes": MAX_AUDIO_BYTES, "maxAudioSeconds": MAX_AUDIO_SECONDS},
    })


@app.post("/api/transcribe")
def transcribe():
    if API_TOKEN:
        scheme, _, supplied = request.headers.get("Authorization", "").partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(supplied, API_TOKEN):
            return jsonify(error="This deployment requires a local AI access token."), 401
    mime_type = request.mimetype or ""
    if mime_type not in ALLOWED_AUDIO_TYPES:
        return jsonify(error="Unsupported audio format. Use browser-recorded WebM, Ogg, MP4 or WAV audio."), 415
    if not request.data:
        return jsonify(error="No audio was received. Record a short clip and try again."), 400
    try:
        return jsonify(transcribe_audio(request.data, mime_type))
    except RuntimeError as exc:
        return jsonify(error=str(exc)), 503
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    except Exception:
        app.logger.exception("Local transcription failed")
        return jsonify(error="Local speech inference failed. Check the server log and try again."), 500


@app.errorhandler(RequestEntityTooLarge)
def request_too_large(_error):
    return jsonify(error="Recording is too large; keep clips under 30 MB."), 413


logging.getLogger("waitress").setLevel(logging.INFO)
