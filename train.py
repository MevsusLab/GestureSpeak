"""Train the PyTorch classifier and export its float32 layers for CPU inference."""

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch import nn

from features import LABELS, features


ROOT = Path(__file__).resolve().parent
UNKNOWN = len(LABELS) - 1


def perturb(points, rng):
    augmented = points.copy()
    angle = rng.normal(0, 0.13, (len(points), 1))
    x, y = points[..., 0], points[..., 1]
    augmented[..., 0] = x * np.cos(angle) - y * np.sin(angle)
    augmented[..., 1] = x * np.sin(angle) + y * np.cos(angle)
    augmented[:, 1:] += rng.normal(0, 0.025, augmented[:, 1:].shape)
    return augmented


def evaluate_clips(probabilities, classes, video_ids, threshold):
    expected, predicted, accepted = [], [], []
    for clip in np.unique(video_ids):
        mask = video_ids == clip
        distribution = probabilities[mask].mean(axis=0)
        candidate = int(distribution.argmax())
        expected.append(int(classes[mask][0]))
        predicted.append(candidate)
        accepted.append(candidate != UNKNOWN and distribution[candidate] >= threshold)
    return np.array(expected), np.array(predicted), np.array(accepted)


def main():
    torch.manual_seed(1903)
    torch.set_num_threads(4)
    rng = np.random.default_rng(1903)
    with np.load(ROOT / ".cache" / "bukva_subset.npz", allow_pickle=False) as data:
        points, letters, users, video_ids, training = (
            data[key] for key in ("points", "letters", "users", "video_ids", "train")
        )
    training_users = np.unique(users[training])
    assert not np.intersect1d(training_users, users[~training]).size, "Bukva train/test signers overlap"
    rng.shuffle(training_users)
    validation = training & np.isin(users, training_users[:max(1, len(training_users) // 5)])
    fit = training & ~validation
    classes = np.array([LABELS.index(letter) if letter in LABELS else UNKNOWN for letter in letters])
    if len(np.unique(classes[fit])) != len(LABELS):
        raise ValueError("The training split must contain every supported class")

    raw = features(points)
    mean = raw[fit].mean(axis=0)
    scale = np.maximum(raw[fit].std(axis=0), 0.06)
    x = torch.from_numpy((raw - mean) / scale)
    y = torch.from_numpy(classes)
    net = nn.Sequential(nn.Linear(x.shape[1], 192), nn.ReLU(), nn.Dropout(0.20),
                        nn.Linear(192, 96), nn.ReLU(), nn.Dropout(0.15), nn.Linear(96, len(LABELS)))
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.from_numpy((np.bincount(classes[fit]) / fit.sum()) ** -0.5).float(),
        label_smoothing=0.035,
    )
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.0015, weight_decay=0.012)
    best_loss, best_weights = float("inf"), None
    for epoch in range(80):
        net.train()
        train_x = torch.from_numpy((features(perturb(points[fit], rng)) - mean) / scale)
        y_fit = y[fit]
        for batch in torch.randperm(len(train_x)).split(128):
            loss = loss_fn(net(train_x[batch]), y_fit[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        net.eval()
        with torch.inference_mode():
            val_loss = loss_fn(net(x[validation]), y[validation]).item()
        if val_loss < best_loss:
            best_loss, best_weights = val_loss, copy.deepcopy(net.state_dict())
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}: validation loss {val_loss:.4f}", flush=True)

    net.load_state_dict(best_weights)
    with torch.inference_mode():
        probabilities = net(x).softmax(dim=-1).numpy()

    def clips(mask, threshold):
        return evaluate_clips(probabilities[mask], classes[mask], video_ids[mask], threshold)

    def validation_score(threshold):
        true, pred, accepted = clips(validation, threshold)
        supported = true != UNKNOWN
        correct = (accepted & (true == pred))[supported].mean()
        wrong = (accepted & (true != pred))[supported].mean()
        false_accept = accepted[~supported].mean()
        return correct - wrong * 0.75 - false_accept * 0.4

    threshold = max((0.55, 0.60, 0.65, 0.70, 0.75), key=validation_score)
    expected, predicted, accepted = clips(~training, threshold)
    supported = expected != UNKNOWN
    metrics = {
        "threshold": threshold,
        "fit_clips": len(set(video_ids[fit])),
        "validation_clips": len(set(video_ids[validation])),
        "test_clips": len(expected),
        "test_signers": len(set(users[~training])),
        "test_supported_clips": int(supported.sum()),
        "test_supported_top1": float((expected[supported] == predicted[supported]).mean()),
        "test_supported_correct_accepted": int((accepted & (expected == predicted) & supported).sum()),
        "test_supported_wrong_accepted": int((accepted & (expected != predicted) & supported).sum()),
        "test_unknown_clips": int((~supported).sum()),
        "test_unknown_false_accepts": int(accepted[~supported].sum()),
        "per_letter": {
            letter: {
                "clips": int((expected == index).sum()),
                "top1_correct": int(((expected == index) & (predicted == index)).sum()),
                "correct_accepted": int(((expected == index) & (predicted == index) & accepted).sum()),
            }
            for index, letter in enumerate(LABELS[:-1])
        },
    }

    layers = [net[index] for index in (0, 3, 6)]
    weights = {f"{name}{index}": value.detach().numpy()
               for index, layer in enumerate(layers)
               for name, value in (("w", layer.weight), ("b", layer.bias))}
    from recognition import Classifier

    report = json.dumps(metrics, ensure_ascii=False, indent=2)
    assets = ROOT / "assets"
    with TemporaryDirectory(dir=assets) as folder:
        candidate = Path(folder) / "classifier.npz"
        np.savez_compressed(candidate, mean=mean, scale=scale, labels=LABELS, threshold=threshold, **weights)
        exported = Classifier(candidate)
        for point, expected_probs in zip(points[:100], probabilities[:100]):
            np.testing.assert_allclose(exported.probabilities(point), expected_probs, atol=2e-6)
        metrics_file = Path(folder) / "metrics.json"
        metrics_file.write_text(report + "\n", encoding="utf-8")
        candidate.replace(assets / candidate.name)
        metrics_file.replace(assets / metrics_file.name)
    print(report, flush=True)


if __name__ == "__main__":
    main()
