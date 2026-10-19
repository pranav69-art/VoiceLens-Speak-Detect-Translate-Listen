FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VOICELENS_HOST=0.0.0.0 \
    PORT=8000 \
    HF_HOME=/model-cache

WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY app.py serve.py local_asr.py audio_features.py VoiceLens.html ./
RUN useradd --create-home --uid 10001 voicelens \
    && mkdir -p /model-cache \
    && chown -R voicelens:voicelens /app /model-cache

USER voicelens
EXPOSE 8000
CMD ["python", "serve.py", "--no-browser"]
