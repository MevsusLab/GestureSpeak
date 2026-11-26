import numpy as np


LABELS = tuple("АДИКМОПРСТ") + ("",)
PAIRS = np.triu_indices(21, k=1)


def canonical_hand(landmarks, handedness, width, height):
    points = np.array([(p.x, p.y, p.z) for p in landmarks], dtype=np.float32)
    points[:, 1] *= height / width
    points -= points[0].copy()
    if handedness == "Left":
        points[:, 0] *= -1
    scale = np.linalg.norm(points[[5, 9, 13, 17], :2], axis=1).mean()
    if not np.isfinite(points).all() or scale < 1e-4:
        return None
    return points / scale


def features(points):
    points = np.asarray(points, dtype=np.float32)
    distances = np.linalg.norm(points[..., PAIRS[0], :] - points[..., PAIRS[1], :], axis=-1)
    return np.concatenate([points.reshape(*points.shape[:-2], 63), distances], axis=-1)
