# VoiceLens – Speak, Detect, Translate, Listen

A sleek single-page web app with a glowing dark/light glassmorphism design. Open your camera, speak in any language and
VoiceLens turns your voice into **live text**, identifies the **spoken language**, **translates** it into 130+ languages,
**reads it aloud**, and lets you **export** the result as a `.txt` or `.pdf` file.


Let's Go ─► camera + mic permission ─► viewfinder + audio visualizer
                                              │
             Browser ASR or local Whisper ─► transcript + language ID
                                              │
                    pick a language ─► translation ─► Read Aloud
                                              │
                                  Export: TXT / PDF
```

---

## Features

| Area | What you get |
|------|--------------|
| **Landing view** | Deep black hero with floating neon particles and an aurora glow. The **Let’s Go** button has a rotating 7-colour conic-gradient border (cyan, magenta, purple, electric blue, green, yellow, pink), a pulsing blurred halo, a ping ring, a floating motion and hover/press scale effects. |
| **Entrance** | Clicking Let’s Go plays a zoom-and-blur exit animation, then a glass permission popup (*Allow access* / *Mic only*). |
| **Viewfinder** | Shows the mirrored live camera feed in a futuristic box with glowing corner brackets and a scan line. Falls back to audio-only if the camera is blocked or missing, with a retry action. |
| **Audio visualizer** | 56-bar glowing waveform driven by the Web Audio `AnalyserNode`; idles with a gentle wave when not listening. |
| **Live transcription** | Continuous speech recognition with interim (in-progress) text, typed out character by character into the transcript panel. |
| **Local AI transcription** | Record a short clip for Faster Whisper ASR and language identification on the local Python server; preview the extracted Log-Mel spectrogram. |
| **Language detection** | A badge shows the detected language (e.g. *Detected: Spanish (es)*). You can also choose the spoken language manually. |
| **Editable transcript** | The transcript is a normal text area – fix mistakes or type extra text at any time. |
| **Translation** | Searchable dropdown of ~130 target languages; picking one translates immediately. |
| **Read Aloud** | Speaks the translation with the browser’s speech synthesis, using a voice that matches the target language. |
| **Export** | TXT / PDF toggle and a glowing **Download transcript** button. The file contains the date, detected language, transcript and (if present) the translation. |
| **Themes** | Dark (default) and light, switched with the ☾/☀ button; follows the system preference when no choice was made. |
| **Responsive** | Two-column layout on desktop, single column on phones. |

---

## How to use

1. Create and activate a virtual environment: `py -3.13 -m venv .venv`, then `\.venv\Scripts\Activate.ps1` on Windows (or `source .venv/bin/activate` on macOS/Linux).
2. Install dependencies with `python -m pip install -r requirements.txt`.
3. Start the local web server with `python serve.py`. It opens VoiceLens at `http://localhost:8000/`.
4. Click **Let’s Go**, then **Allow access** and approve camera and microphone permissions in the browser.
5. Press **Start listening** for browser recognition and use **Stop listening** to finish, or choose **Start local AI**, speak, then press **Stop and transcribe**. Whisper detects the language and adds the transcript.
6. Edit the transcript if needed.
7. Search or scroll the **Translate** dropdown and click a language. The translation appears below.
8. Press **Read Aloud** to hear it (press again to stop).
9. Pick **TXT** or **PDF** and press **Download transcript**.

Tip: for the most accurate recognition, pick the *spoken language* manually instead of Auto-detect.

---

## Tech stack

- **HTML5 + CSS3** – CSS variables for the dark/light themes, `@property` animated conic gradients, `backdrop-filter`
  glassmorphism, keyframe animations, `color-mix()` glows.
- **JavaScript (ES6+), no framework** – a single inline script.
- **Python local server** – serves the page and exposes local health and transcription routes.
- **Faster Whisper** – encoder-decoder Transformer for local speech recognition and language identification.
- **NumPy + SciPy** – project Log-Mel spectrogram extraction in `audio_features.py`.
- **Web Speech API** – `SpeechRecognition` (speech → text) and `SpeechSynthesis` (text → speech).
- **MediaDevices API** – `navigator.mediaDevices.getUserMedia` for camera and microphone.
- **Web Audio API** – `AudioContext` + `AnalyserNode` for the waveform.
- **Canvas 2D** – particle background and audio visualizer.
- **html2pdf.js 0.10.1** (jsPDF + html2canvas, loaded from cdnjs) – client-side PDF generation.
- **Intl.DisplayNames** – language names are generated from language codes, so the list stays compact.

### AI translation setup

VoiceLens runs as a local web app. To use AI translation, open **AI translation settings** in the Translate panel and enter an
OpenAI-compatible chat completions endpoint, model name and API key. The default endpoint and model are for OpenAI;
other compatible providers can be used by changing both values. The key is kept in the current browser session only.
Translation requests go directly from the browser to the configured provider, so that provider must allow browser
requests (CORS). For a public deployment, use a small server-side proxy rather than putting a shared API key in the page.

TXT downloads are generated by the browser. PDF downloads use html2pdf.js from cdnjs and need an internet connection.

---

## How it works

### Machine learning and deep learning

- **Audio capture:** The browser records a short microphone clip only while **Start local AI** is active, then sends it to the VoiceLens Python server at localhost.
- **Log-Mel features:** `audio_features.py` converts decoded 16 kHz audio into an 80-band Log-Mel spectrogram. Whisper builds its model-specific encoder input from that same waveform.
- **ASR and spoken language identification:** `local_asr.py` runs Faster Whisper, an encoder-decoder Transformer based on Whisper. The encoder represents audio features; the decoder generates transcript tokens. Whisper returns the detected language and confidence with the transcript.
- **Neural translation:** The translation panel sends text to the configured chat-completions model. Use an AI provider that supports multilingual translation.
- **Read aloud:** Browser `SpeechSynthesis` reads the translation using an installed voice. Voice quality and neural synthesis depend on the browser and operating system.

The default `base` Whisper model runs on CPU with int8 quantization. Model weights download on first use and are cached by Faster Whisper. Set `VOICELENS_WHISPER_MODEL` before starting the server to choose another model size. Audio recordings are sent only to the local server for recognition; translated text is sent to the AI provider configured in the browser.

**Speech recognition** – `SpeechRecognition` runs in continuous mode with interim results. Final phrases are queued and typed
into the text area; interim phrases are shown in a cyan line underneath. If the engine stops by itself, the app restarts it
while *Listening* is on.

**Language detection** – the Web Speech API cannot identify a language on its own, so VoiceLens works like this in
*Auto-detect* mode:

1. Recognition starts with your browser’s language.
2. Each finished phrase is checked by **Unicode script** (Devanagari → Hindi, Arabic script → Arabic, Hangul → Korean,
   Cyrillic → Russian, Thai, Greek, Hebrew, Bengali, Tamil, Telugu, Hiragana/Katakana → Japanese, Han → Chinese) and, for
   Latin-script text, by **common-word matching** (English, Spanish, French, German, Portuguese, Italian, Dutch,
   Indonesian, Turkish).
3. If a different language is found, the badge updates and recognition restarts in that language.

This is a lightweight heuristic. Similar Latin-script languages can be confused, so choose the spoken language manually when
accuracy matters.

**Translation** – the transcript and chosen target language are sent to the configured AI chat completions endpoint with
an instruction to return only the translation. Configure the endpoint, model and API key in the page before translating.

**Read Aloud** – picks an installed voice whose language matches the target (exact match first, then the base language).
If none is installed, it warns you and uses the default voice.

**PDF export** – the text is placed in an off-screen styled `div`, rendered by html2pdf.js and saved as an A4 PDF. Rendering
from HTML (instead of plain jsPDF text) means non-Latin scripts such as Hindi, Arabic or Japanese display correctly.

---

## Deployment

The project includes a Docker image and a `Procfile` for platforms that provide Python web services. To build and run it:

```sh
docker build -t voicelens .
docker run --rm -p 8000:8000 -v voicelens-model-cache:/model-cache -e VOICELENS_API_TOKEN=choose-a-long-secret voicelens
```

The first local AI request downloads the Whisper weights into the model cache. Keep `/model-cache` on persistent storage
to avoid downloading the model again after a container restart. For a public deployment, set a strong
`VOICELENS_API_TOKEN`, expose the service through HTTPS, and give users that token in **Hosted local AI access**. Camera
and microphone access require HTTPS outside localhost. A deployed transcription request sends its recorded audio to the
server running VoiceLens; local execution sends it to the user's own machine. Translation text and its configured API key
are sent directly from the browser to the selected AI provider.

Run the project checks with `python -m unittest discover -s tests -v` after installing the requirements.

---

## Browser support

| Feature | Chrome / Edge | Safari | Firefox |
|---------|:------------:|:------:|:-------:|
| Camera + microphone | ✅ | ✅ | ✅ |
| Live speech recognition | ✅ | ✅ (recent versions) | ❌ |
| Local AI recording | ✅ | ✅ | ✅ |
| Read Aloud | ✅ | ✅ | ✅ |
| Animated gradient border (`@property`) | ✅ | ✅ | ✅ (recent versions) |

Read Aloud only works for languages your operating system has a voice for. HTTPS is required for camera and microphone access.

---




---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| No transcription appears | Use Chrome, Edge or Safari, allow microphone access and check that the right input device is selected. |
| Local AI transcription is unavailable | Install `requirements.txt` and restart `serve.py`. The Whisper model downloads the first time **Start local AI** is used. |
| Camera box says “Camera unavailable” | Open the page over HTTPS or localhost, allow camera access in the browser site settings, then click **Retry camera**. Close other apps using the camera. Microphone-only mode remains available. |
| Wrong language detected | Choose the spoken language from the dropdown instead of Auto-detect. |
| Translation fails | Check the AI endpoint, model and API key. The provider must allow browser requests from this page (CORS). |
| Read Aloud is silent or uses the wrong voice | Install a voice for that language in your OS settings. |
| Download does nothing | Accept the save confirmation; if downloads are unavailable the text is copied to your clipboard instead. |
| Right-to-left text looks odd | The text boxes follow the browser’s bidi rules; the exported files keep the correct text. |

---

## Privacy

- The camera feed is only displayed in the viewfinder; it is **not recorded or uploaded**.
- Browser speech recognition may send audio to the browser vendor's service. The **Local AI** option sends recordings to
  the local or deployed VoiceLens server for processing.
- Text you translate and your API key are sent to the AI endpoint you configure. Your key is kept in session storage for the current browser session; use a server-side proxy for a public deployment.

---

## Ideas for the future

- Subtitle (`.srt`) export and per-sentence timestamps
- Camera + audio session recording
- Offline translation
- Saving transcript history





## License

 use it, change it, share it.
