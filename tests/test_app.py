import io
import unittest
import wave
from pathlib import Path

import cv2
import numpy as np

from app import Speech, create_app
from features import LABELS
from recognition import Recognizer


ROOT = Path(__file__).resolve().parents[1]


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(1)
        cls.recognizer = Recognizer()
        cls.client = create_app(cls.recognizer, Speech(), "127.0.0.1").test_client()

    @classmethod
    def tearDownClass(cls):
        cls.recognizer.close()

    def post_frame(self, frame):
        ok, jpg = cv2.imencode(".jpg", frame)
        self.assertTrue(ok)
        return self.client.post("/api/frame", base_url="http://127.0.0.1",
                                content_type="image/jpeg", data=jpg.tobytes())

    def test_blank_and_demo_frames_are_processed_by_the_real_model(self):
        blank = self.post_frame(np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertEqual(blank.json, {"points": [], "letter": "", "score": 0.0})
        capture = cv2.VideoCapture(str(ROOT / "static" / "demo.mp4"))
        sequence = []
        index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                index += 1
                if index % 4:
                    continue
                result = self.post_frame(frame)
                self.assertEqual(result.status_code, 200)
                prediction = result.json
                letter = prediction["letter"]
                if letter and prediction["score"] >= self.recognizer.classifier.threshold:
                    self.assertEqual(len(prediction["points"]), 21)
                    if not sequence or sequence[-1] != letter:
                        sequence.append(letter)
        finally:
            capture.release()
        self.assertEqual("".join(sequence), "ПРИВЕТ")

    def test_every_supported_letter_has_a_video_guide(self):
        self.assertEqual(self.recognizer.classifier.labels, LABELS)
        url = "http://127.0.0.1"
        with self.client.get("/", base_url=url) as page:
            html = page.get_data(as_text=True)
            for letter in LABELS[:-1]:
                with self.subTest(letter=letter):
                    self.assertIn(f'data-letter="{letter}"', html)
                    with self.client.get(f"/static/signs/{letter}.mp4", base_url=url) as example:
                        self.assertEqual(example.status_code, 200)
                        self.assertEqual(example.mimetype, "video/mp4")

    def test_rejects_bad_frames_and_cross_origin_posts(self):
        url = "http://127.0.0.1"
        invalid = self.client.post("/api/frame", base_url=url, content_type="image/jpeg", data=b"bad")
        self.assertEqual(invalid.status_code, 400)
        too_large = self.post_frame(np.zeros((16, 1025, 3), dtype=np.uint8))
        self.assertEqual(too_large.status_code, 400)
        blocked = self.client.post("/api/speech", base_url=url, json={"text": "МИР"},
                                   headers={"Origin": "https://example.com"})
        self.assertEqual(blocked.status_code, 403)
        blocked_host = self.client.get("/", base_url="http://external.example")
        self.assertEqual(blocked_host.status_code, 400)

    def test_russian_speech_returns_nonempty_pcm(self):
        response = self.client.post("/api/speech", base_url="http://127.0.0.1", json={"text": "Привет, мир!"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "audio/wav")
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        with wave.open(io.BytesIO(response.data), "rb") as audio:
            self.assertEqual(audio.getframerate(), 22050)
            samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
        self.assertGreater(len(samples), 10_000)
        self.assertGreater(np.max(np.abs(samples.astype("int32"))), 100)


if __name__ == "__main__":
    unittest.main()
