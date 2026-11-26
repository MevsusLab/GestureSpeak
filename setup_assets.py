import hashlib
import shutil
import tempfile
import urllib.request
from pathlib import Path


ASSETS = Path(__file__).resolve().parent / "assets"
PIPER = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/ru/ru_RU/irina/medium/"
SOURCES = {
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
    ),
    "ru_RU-irina-medium.onnx": (
        PIPER + "ru_RU-irina-medium.onnx",
        "8ff38212d23da300bbe3705c645e6e5b9475f0bfde01558eb17813e22acaaaaa",
    ),
    "ru_RU-irina-medium.onnx.json": (
        PIPER + "ru_RU-irina-medium.onnx.json",
        "c2ec28bb38e2b59e93b959b3e40348c1afebbd272f30fed5d41205d08e98a9d7",
    ),
}


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def main():
    ASSETS.mkdir(exist_ok=True)
    for name, (url, expected) in SOURCES.items():
        target = ASSETS / name
        if target.is_file() and digest(target) == expected:
            continue
        print(f"Downloading {name}...", flush=True)
        with tempfile.TemporaryDirectory(dir=ASSETS) as folder:
            candidate = Path(folder) / name
            with urllib.request.urlopen(url, timeout=60) as response, candidate.open("wb") as file:
                shutil.copyfileobj(response, file, length=256 * 1024)
            if digest(candidate) != expected:
                raise ValueError(f"SHA-256 mismatch for {name}")
            candidate.replace(target)


if __name__ == "__main__":
    main()
