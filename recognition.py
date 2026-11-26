from pathlib import Path
from threading import Lock

import cv2
import mediapipe as mp
import numpy as np

from features import LABELS, canonical_hand, features


ASSETS = Path(__file__).resolve().parent / "assets"


class Classifier:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as model:
            if tuple(model["labels"]) != LABELS:
                raise ValueError("The classifier uses an incompatible alphabet")
            self.threshold = float(model["threshold"])
            self.mean = model["mean"]
            self.scale = model["scale"]
            self.layers = [(model[f"w{i}"], model[f"b{i}"]) for i in range(3)]

    def probabilities(self, points):
        values = (features(points) - self.mean) / self.scale
        for weights, bias in self.layers[:-1]:
            values = np.maximum(weights @ values + bias, 0)
        weights, bias = self.layers[-1]
        logits = weights @ values + bias
        scores = np.exp(logits - logits.max())
        return scores / scores.sum()


class Recognizer:
    def __init__(self):
        self.classifier = Classifier(ASSETS / "classifier.npz")
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(ASSETS / "hand_landmarker.task")),
            num_hands=1,
            min_hand_detection_confidence=0.35,
            min_hand_presence_confidence=0.40,
        )
        self.tracker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self.lock = Lock()

    def close(self):
        self.tracker.close()

    def read(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        with self.lock:
            result = self.tracker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not result.hand_landmarks:
            return {"points": [], "letter": "", "score": 0.0}
        landmarks = result.hand_landmarks[0]
        points = canonical_hand(landmarks, result.handedness[0][0].category_name, width, height)
        payload = {"points": [[round(p.x, 5), round(p.y, 5)] for p in landmarks], "letter": "", "score": 0.0}
        if points is None:
            return payload
        scores = self.classifier.probabilities(points)
        index = int(scores.argmax())
        score = float(scores[index])
        if LABELS[index]:
            payload.update(letter=LABELS[index], score=round(score, 4))
        return payload
