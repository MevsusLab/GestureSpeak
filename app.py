import argparse
import io
import ipaddress
import os
import ssl
import threading
import webbrowser

os.environ.setdefault("GLOG_minloglevel", "2")

import cv2
import numpy as np
from flask import Flask, abort, jsonify, render_template, request, send_file
from PIL import Image, UnidentifiedImageError
from waitress import create_server

from recognition import ASSETS, Recognizer


class Speech:
    def __init__(self):
        self.lock = threading.Lock()
        self.voice = None

    def synthesize(self, text):
        import wave

        from piper import PiperVoice

        audio = io.BytesIO()
        with self.lock:
            if self.voice is None:
                self.voice = PiperVoice.load(ASSETS / "ru_RU-irina-medium.onnx")
            with wave.open(audio, "wb") as output:
                self.voice.synthesize_wav(text, output)
            audio.seek(0)
        return audio


def create_app(recognizer, speech, host):
    app = Flask(__name__)
    app.config.update(MAX_CONTENT_LENGTH=500_000, TRUSTED_HOSTS=[host])

    @app.after_request
    def headers(response):
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "media-src 'self' blob:; img-src 'self' blob:; object-src 'none'; "
            "frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        )
        response.headers["Permissions-Policy"] = "camera=(self), microphone=()"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.before_request
    def same_origin():
        if request.method == "POST":
            origin = request.headers.get("Origin")
            if (origin and origin != request.host_url.rstrip("/")) or request.headers.get("Sec-Fetch-Site") == "cross-site":
                abort(403)

    @app.get("/")
    def index():
        classifier = recognizer.classifier
        return render_template("index.html", letters=classifier.labels[:-1], threshold=classifier.threshold)

    @app.post("/api/frame")
    def frame():
        if request.mimetype != "image/jpeg":
            return jsonify(error="Ожидается кадр JPEG."), 415
        data = request.get_data()
        try:
            with Image.open(io.BytesIO(data), formats=["JPEG"]) as jpeg:
                if max(jpeg.size) > 1024 or min(jpeg.size) < 16:
                    return jsonify(error="Недопустимый размер кадра."), 400
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            return jsonify(error="Не удалось прочитать кадр."), 400
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if image is None:
            return jsonify(error="Не удалось прочитать кадр."), 400
        return jsonify(recognizer.read(image))

    @app.post("/api/speech")
    def speak():
        if request.mimetype != "application/json":
            return jsonify(error="Ожидается текст в JSON."), 415
        payload = request.get_json(silent=True)
        text = payload.get("text") if isinstance(payload, dict) else None
        if not isinstance(text, str) or not text.strip() or len(text) > 600:
            return jsonify(error="Введите от 1 до 600 символов."), 400
        return send_file(speech.synthesize(text.strip()), mimetype="audio/wav")

    return app


def main():
    parser = argparse.ArgumentParser(description="GestureSpeak | РЖЯ в текст и голос")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="open the browser after listening")
    parser.add_argument("--lan-address", help="private IPv4 of this computer for a phone connection")
    parser.add_argument("--cert", help="trusted TLS certificate (PEM) for the LAN address")
    parser.add_argument("--key", help="matching private TLS key (PEM)")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.lan_address:
        try:
            address = ipaddress.IPv4Address(args.lan_address)
            is_lan = address.is_private and not (address.is_loopback or address.is_unspecified)
        except ValueError:
            is_lan = False
        if not is_lan or not args.cert or not args.key:
            parser.error("--lan-address requires a private IPv4 address, --cert and --key")
    elif args.cert or args.key:
        parser.error("--cert and --key require --lan-address")

    for name in ("classifier.npz", "hand_landmarker.task", "ru_RU-irina-medium.onnx", "ru_RU-irina-medium.onnx.json"):
        if not (ASSETS / name).is_file():
            parser.exit(1, f"Missing assets/{name}. Run Start.cmd or python setup_assets.py.\n")

    cv2.setNumThreads(1)
    host = args.lan_address or "127.0.0.1"
    recognizer = Recognizer()
    app = create_app(recognizer, Speech(), host)
    try:
        if args.lan_address:
            from werkzeug.serving import make_server

            tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            tls.load_cert_chain(args.cert, args.key)
            server = make_server(host, args.port, app, threaded=True, ssl_context=tls)
            run = server.serve_forever
        else:
            server = create_server(app, host=host, port=args.port, threads=4, max_request_body_size=500_000)
            run = server.run
        protocol = "https" if args.lan_address else "http"
        url = f"{protocol}://{host}:{args.port}/"
        print(f"GestureSpeak: {url}", flush=True)
        try:
            if args.open:
                threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()
            run()
        except KeyboardInterrupt:
            pass
        finally:
            server.close() if hasattr(server, "close") else server.server_close()
    finally:
        recognizer.close()


if __name__ == "__main__":
    main()
