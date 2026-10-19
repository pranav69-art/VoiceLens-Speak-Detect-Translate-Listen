"""Run VoiceLens with Waitress, a production WSGI server."""

import argparse
import os
import threading
import webbrowser

from waitress import serve

from app import app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("VOICELENS_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--threads", type=int, default=int(os.environ.get("VOICELENS_THREADS", "2")))
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser automatically")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be at least 1")

    local_url = f"http://localhost:{args.port}/"
    print(f"VoiceLens is running on {args.host}:{args.port}")
    if not args.no_browser and args.host in {"127.0.0.1", "localhost", "::1"}:
        threading.Timer(0.5, lambda: webbrowser.open(local_url)).start()
    serve(app, host=args.host, port=args.port, threads=args.threads, max_request_body_size=30 * 1024 * 1024)


if __name__ == "__main__":
    main()
