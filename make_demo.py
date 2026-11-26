"""Build the local gesture reference videos and a genuine three-letter demo from Bukva."""

import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

os.environ.setdefault("GLOG_minloglevel", "2")

import cv2
import numpy as np

from features import LABELS
from recognition import Recognizer


ROOT = Path(__file__).resolve().parent
WIDTH, HEIGHT, FPS = 640, 480, 24
DEMO_WORD = "МИР"


def frames_from(path):
    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or FPS
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            scale = min(WIDTH / width, HEIGHT / height)
            size = round(width * scale), round(height * scale)
            pixels = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            left, top = (WIDTH - size[0]) // 2, (HEIGHT - size[1]) // 2
            boxed = np.full((HEIGHT, WIDTH, 3), (24, 24, 16), dtype=np.uint8)
            boxed[top:top + size[1], left:left + size[0]] = pixels
            while len(frames) / FPS < (index + 1) / fps:
                frames.append(boxed)
            index += 1
    finally:
        capture.release()
    return frames


def encode(frames, output, temp):
    staging = temp / "staging.mp4"
    writer = cv2.VideoWriter(str(staging), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError("Could not open the video encoder")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(staging),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", "-movflags", "+faststart",
                    str(output)], check=True)


def demo_segment(frames, letter, recognizer):
    scores = []
    for frame in frames:
        _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 83])
        result = recognizer.read(cv2.imdecode(jpg, cv2.IMREAD_COLOR))
        scores.append(result["score"] if result["letter"] == letter else 0)
    peak = int(np.argmax(scores))
    if scores[peak] < max(0.7, recognizer.classifier.threshold):
        return []
    left = right = peak
    threshold = recognizer.classifier.threshold + 0.05
    while left > 0 and scores[left - 1] >= threshold:
        left -= 1
    while right + 1 < len(scores) and scores[right + 1] >= threshold:
        right += 1
    portion = frames[left:right + 1]
    # Extend a real holding pose where the source clip is shorter than the input dwell time.
    padding = max(0, round(1.65 * FPS) - len(portion))
    middle = peak - left + 1
    return portion[:middle] + [frames[peak]] * padding + portion[middle:]


def main():
    cv2.setNumThreads(1)
    media = ROOT / ".cache" / "test-videos"
    out = ROOT / "static" / "signs"
    out.mkdir(exist_ok=True)
    with np.load(ROOT / ".cache" / "bukva_subset.npz", allow_pickle=False) as data:
        points, letters, video_ids, train = (data[key] for key in ("points", "letters", "video_ids", "train"))
    recognizer = Recognizer()
    try:
        segments = {}
        with TemporaryDirectory() as folder:
            temp = Path(folder)
            for class_index, letter in enumerate(LABELS[:-1]):
                mask = (letters == letter) & ~train
                ranking = []
                for clip in np.unique(video_ids[mask]):
                    path = media / f"{clip}.mp4"
                    if path.is_file():
                        probabilities = [recognizer.classifier.probabilities(p)[class_index]
                                         for p in points[(video_ids == clip) & mask]]
                        ranking.append((float(np.mean(probabilities)), path))
                for _, path in sorted(ranking, reverse=True):
                    frames = frames_from(path)
                    if not frames:
                        continue
                    if letter in DEMO_WORD:
                        segment = demo_segment(frames, letter, recognizer)
                        if not segment:
                            continue
                        segments[letter] = segment
                    encode(frames, out / f"{letter}.mp4", temp)
                    print(f"{letter}: {path.stem}", flush=True)
                    break
                else:
                    raise RuntimeError(f"No usable reference clip for {letter}")
            gap = [np.full((HEIGHT, WIDTH, 3), (24, 24, 16), dtype=np.uint8)] * round(0.7 * FPS)
            demo = gap.copy()
            for letter in DEMO_WORD:
                demo += segments[letter] + gap
            encode(demo, ROOT / "static" / "demo.mp4", temp)
    finally:
        recognizer.close()


if __name__ == "__main__":
    main()
