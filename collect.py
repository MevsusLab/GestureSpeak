"""Extract a reproducible subset of hand landmarks from the Bukva video dataset."""

import argparse
import csv
import io
import os
import random
import struct
import threading
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tempfile import NamedTemporaryFile

os.environ.setdefault("GLOG_minloglevel", "2")

import cv2
import mediapipe as mp
import numpy as np
import requests

from features import LABELS, canonical_hand


ROOT = Path(__file__).resolve().parent
URL = "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/bukva/bukva.zip"
LOCAL = threading.local()


def get_range(start, end):
    if not hasattr(LOCAL, "http"):
        LOCAL.http = requests.Session()
    for attempt in range(3):
        try:
            with LOCAL.http.get(URL, headers={"Range": f"bytes={start}-{end}"},
                                timeout=(7, 45), stream=True) as response:
                response.raise_for_status()
                if response.status_code != 206:
                    raise OSError("The archive server did not honor the Range header")
                data = response.content
                if start and len(data) != end - start + 1:
                    raise OSError("Incomplete range response")
                return data, int(response.headers["Content-Range"].split("/")[-1])
        except (requests.RequestException, OSError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


class RemoteArchive(io.RawIOBase):
    def __init__(self):
        self.tail, self.size = get_range("", 2_000_000)
        self.start = self.size - len(self.tail)
        self.position = 0

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = max(0, offset + (0, self.position, self.size)[whence])
        return self.position

    def read(self, size=-1):
        begin = self.position
        end = self.size if size < 0 else min(begin + size, self.size)
        self.position = end
        if end <= begin:
            return b""
        if begin >= self.start:
            return self.tail[begin - self.start:end - self.start]
        return get_range(begin, end - 1)[0]


def clip_bytes(member):
    # Read the local header and its compressed member in a single range request.
    start = member.header_offset
    packet, _ = get_range(start, start + member.compress_size + 1023)
    signature, name_size, extra_size = struct.unpack_from("<4s22xHH", packet)
    if signature != b"PK\x03\x04":
        raise ValueError(f"Invalid ZIP header for {member.filename}")
    offset = 30 + name_size + extra_size
    data = packet[offset:offset + member.compress_size]
    decoded = zipfile._get_decompressor(member.compress_type).decompress(data) if member.compress_type else data
    if zipfile.crc32(decoded) != member.CRC:
        raise ValueError(f"ZIP CRC mismatch for {member.filename}")
    return decoded


def extract(row, member, demo_folder):
    if not hasattr(LOCAL, "tracker"):
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(ROOT / "assets" / "hand_landmarker.task")),
            min_hand_detection_confidence=0.35, min_hand_presence_confidence=0.4,
        )
        LOCAL.tracker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    with NamedTemporaryFile(suffix=".mp4", delete=False) as file:
        path = Path(file.name)
    try:
        path.write_bytes(clip_bytes(member))
        capture = cv2.VideoCapture(str(path))
        points = []
        try:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            for position in np.unique(np.linspace(int(frame_count * 0.20), int(frame_count * 0.80), 5, dtype=int)):
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(position))
                ok, frame = capture.read()
                if not ok:
                    continue
                height, width = frame.shape[:2]
                scale = min(1.0, 640 / max(height, width))
                frame = cv2.resize(frame, (round(width * scale), round(height * scale)),
                                   interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = LOCAL.tracker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                if result.hand_landmarks:
                    hand = canonical_hand(result.hand_landmarks[0], result.handedness[0][0].category_name,
                                          rgb.shape[1], rgb.shape[0])
                    if hand is not None:
                        points.append(hand)
        finally:
            capture.release()
        if demo_folder and row["text"] in LABELS[:-1] and row["train"] == "False" and len(points) >= 4:
            demo_folder.mkdir(parents=True, exist_ok=True)
            path.replace(demo_folder / member.filename.rsplit("/", 1)[-1])
        return np.array(points, dtype=np.float32).reshape(-1, 21, 3)
    finally:
        path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--train-per-letter", type=int, default=60)
    parser.add_argument("--test-per-letter", type=int, default=19)
    parser.add_argument("--save-test-videos", action="store_true")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    folder = ROOT / ".cache"
    folder.mkdir(exist_ok=True)
    records = folder / "records"
    records.mkdir(exist_ok=True)
    with zipfile.ZipFile(RemoteArchive()) as archive:
        annotations = next(n for n in archive.namelist() if n.endswith("annotations.tsv"))
        with archive.open(annotations) as data:
            content = list(csv.DictReader(io.TextIOWrapper(data, encoding="utf-8-sig"), delimiter="\t"))
        clips = {Path(i.filename).stem: i for i in archive.infolist()
                 if i.filename.startswith("trimmed/") and i.filename.endswith(".mp4")}

    groups = defaultdict(list)
    for row in content:
        clip = clips.get(row["attachment_id"])
        if clip and len(row["text"]) == 1:
            groups[row["text"], row["train"] == "True"].append((row, clip))
    rng = random.Random(1903)
    selection = []
    for (letter, train), values in sorted(groups.items()):
        rng.shuffle(values)
        count = (args.train_per_letter if train else args.test_per_letter) if letter in LABELS else (8 if train else 4)
        selection.extend(values[:count])
    rng.shuffle(selection)
    print(f"{len(selection)} videos across {len(groups) // 2} letters", flush=True)

    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        demo_folder = folder / "test-videos" if args.save_test_videos else None
        jobs = {pool.submit(extract, row, clip, demo_folder): row for row, clip in selection
                if not (records / f'{row["attachment_id"]}.npy').exists()}
        for index, job in enumerate(as_completed(jobs), start=1):
            row = jobs[job]
            try:
                points = job.result()
                np.save(records / f'{row["attachment_id"]}.npy', points, allow_pickle=False)
            except Exception as error:
                failures += 1
                if failures <= 12:
                    print(f'{row["attachment_id"]}: {error}', flush=True)
            if index % 20 == 0:
                print(f"Processed {index}/{len(jobs)} (errors: {failures})", flush=True)

    rows = []
    for row, _ in selection:
        path = records / f'{row["attachment_id"]}.npy'
        if path.exists():
            rows.extend((hand, row["text"], row["user_id"], row["attachment_id"], row["train"] == "True")
                        for hand in np.load(path, allow_pickle=False))
    if not rows:
        raise RuntimeError("No hand landmarks were found")
    x, letters, users, video_ids, train = zip(*rows)
    np.savez_compressed(folder / "bukva_subset.npz", points=np.stack(x), letters=letters,
                        users=users, video_ids=video_ids, train=train)
    print(f"{len(rows)} labeled frames, {len(set(video_ids))} usable clips, {failures} errors", flush=True)


if __name__ == "__main__":
    main()
