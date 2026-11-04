"""The notebook checkpoint must score and flag the same way Try-it does."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from app.classifier import IMAGENET_MEAN, IMAGENET_STD, Classifier, preprocess
from app.config import CFG
from app.flagging import FlagEngine, TrackHistory

ROOT = Path(__file__).resolve().parent.parent


def notebook_score(model, crops, indexes):
    """Literal Try-it score_crops: BGR -> RGB, /255, 224, ImageNet, softmax, sum cheating columns."""
    import torchvision.transforms.functional as TF
    x = torch.stack([
        TF.normalize(
            TF.resize(torch.from_numpy(np.ascontiguousarray(c[:, :, ::-1])).permute(2, 0, 1).float() / 255,
                      [224, 224], antialias=True),
            IMAGENET_MEAN, IMAGENET_STD)
        for c in crops
    ])
    with torch.no_grad():
        return torch.softmax(model(x), 1)[:, indexes].sum(1).numpy()


def notebook_flags(samples, thr):
    """Try-it video state machine for one track. samples: (t, raw_score)."""
    win, state, opened = [], {"above": None, "open": None, "cool": -1}, []
    for t, s in samples:
        win.append((t, float(s)))
        win = [(tt, v) for tt, v in win if tt >= t - 3]
        sm = sum(v for _, v in win) / len(win)
        if sm >= thr:
            state["above"] = t if state["above"] is None else state["above"]
            if state["open"]:
                state["open"]["end"] = t
            elif t - state["above"] >= 2 and t >= state["cool"]:
                state["open"] = {"start": state["above"], "end": t}
                opened.append(state["open"])
        else:
            if state["open"]:
                state["open"], state["cool"] = None, t + 20
            state["above"] = None
    return [(f["start"], f["end"]) for f in opened]


class _Logits(nn.Module):
    """Whole-module export: fixed logits, cheating is column 0."""

    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor([5.0, 0.0]))

    def forward(self, x):
        return self.bias.view(1, -1).expand(x.shape[0], -1)


class _AlreadySoftmax(nn.Module):
    def forward(self, x):
        return torch.tensor([[0.2, 0.8]], dtype=torch.float32).expand(x.shape[0], -1)


def _constant_effnet(bias):
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(bias))
    nn.init.zeros_(m.classifier[1].weight)
    with torch.no_grad():
        m.classifier[1].bias.copy_(torch.tensor(bias, dtype=torch.float32))
    return m


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._saved = (
            list(CFG["classifier"]["class_names"]),
            list(CFG["classifier"]["cheating_classes"]),
            CFG["flagging"]["threshold"],
            CFG["flagging"].get("use_checkpoint_threshold", True),
        )
        # Stale config the backend used to keep even after best.pt was dropped in.
        CFG["classifier"]["class_names"] = [
            "backwatching", "frontwatching", "normal", "showinggestures", "sidewatching", "suspiciouswatching"]
        CFG["classifier"]["cheating_classes"] = [
            "backwatching", "showinggestures", "sidewatching", "suspiciouswatching"]
        CFG["flagging"]["threshold"] = 0.6
        CFG["flagging"]["use_checkpoint_threshold"] = True
        self.crops = [
            np.random.default_rng(0).integers(0, 255, (180, 90, 3), dtype=np.uint8),
            np.random.default_rng(1).integers(0, 255, (90, 220, 3), dtype=np.uint8),
        ]

    def tearDown(self):
        names, cheat, thr, use = self._saved
        CFG["classifier"]["class_names"] = names
        CFG["classifier"]["cheating_classes"] = cheat
        CFG["flagging"]["threshold"] = thr
        CFG["flagging"]["use_checkpoint_threshold"] = use
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_stub_file_matches_notebook_and_uses_checkpoint_metadata(self):
        path = ROOT / "models" / "best_stub.pt"
        clf = Classifier(path)
        self.assertEqual(clf.class_names, ["normal", "cheating"])
        self.assertEqual(clf.cheat_idx, [1])
        self.assertEqual(clf.activation, "softmax")
        self.assertAlmostEqual(clf.threshold, 0.5)
        self.assertEqual(clf.threshold_source, "checkpoint")
        self.assertFalse(clf.is_stub)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        ref_model = models.efficientnet_b0(weights=None)
        ref_model.classifier[1] = nn.Linear(ref_model.classifier[1].in_features, 2)
        ref_model.load_state_dict(ck["model_state_dict"])
        ref_model.eval()
        got = clf.cheat_score(clf.predict(self.crops))
        exp = notebook_score(ref_model, self.crops, [1])
        np.testing.assert_allclose(got, exp, atol=1e-5)

    def test_cheating_index_zero_is_not_column_one(self):
        model = _constant_effnet([5.0, 0.0])
        path = self.tmp / "best.pt"
        torch.save({
            "arch": "efficientnet_b0",
            "model_state_dict": model.state_dict(),
            "class_names": ["cheating", "normal"],
            "cheating_classes": ["cheating"],
            "threshold": 0.22,
            "input_size": 224,
            "mean": IMAGENET_MEAN, "std": IMAGENET_STD,
        }, path)
        clf = Classifier(path)
        self.assertEqual(clf.cheat_idx, [0])
        self.assertAlmostEqual(clf.threshold, 0.22)
        score = float(clf.cheat_score(clf.predict(self.crops[:1]))[0])
        # softmax([5, 0])[0] ≈ 0.993. Column 1 would be ≈ 0.007 and would not flag.
        self.assertGreater(score, 0.99)

    def test_metrics_json_threshold_when_checkpoint_has_none(self):
        model = _constant_effnet([0.0, 0.0])
        path = self.tmp / "best.pt"
        torch.save({
            "arch": "efficientnet_b0",
            "model_state_dict": model.state_dict(),
            "class_names": ["normal", "cheating"],
            "cheating_classes": ["cheating"],
            "input_size": 224,
        }, path)
        (self.tmp / "metrics.json").write_text(json.dumps({"threshold": 0.33, "class_names": ["normal", "cheating"]}))
        clf = Classifier(path)
        self.assertAlmostEqual(clf.threshold, 0.33)
        self.assertEqual(clf.threshold_source, "metrics.json")

    def test_config_threshold_can_override(self):
        model = _constant_effnet([0.0, 4.0])
        path = self.tmp / "best.pt"
        torch.save({
            "arch": "efficientnet_b0", "model_state_dict": model.state_dict(),
            "class_names": ["normal", "cheating"], "cheating_classes": ["cheating"],
            "threshold": 0.22, "input_size": 224,
        }, path)
        CFG["flagging"]["use_checkpoint_threshold"] = False
        CFG["flagging"]["threshold"] = 0.6
        clf = Classifier(path)
        self.assertAlmostEqual(clf.threshold, 0.6)
        self.assertEqual(clf.threshold_source, "config")

    def test_whole_model_respects_class_order(self):
        path = self.tmp / "whole.pt"
        torch.save({
            "model": _Logits(),
            "class_names": ["cheating", "normal"],
            "cheating_classes": ["cheating"],
            "threshold": 0.4,
            "input_size": 224,
        }, path)
        clf = Classifier(path)
        self.assertEqual(clf.activation, "softmax")
        self.assertEqual(clf.cheat_idx, [0])
        self.assertGreater(float(clf.cheat_score(clf.predict(self.crops[:1]))[0]), 0.99)

    def test_already_softmaxed_export_is_not_softmaxed_again(self):
        path = self.tmp / "probs.pt"
        torch.save({
            "model": _AlreadySoftmax(),
            "class_names": ["normal", "cheating"],
            "cheating_classes": ["cheating"],
            "threshold": 0.5,
            "input_size": 224,
        }, path)
        clf = Classifier(path)
        self.assertEqual(clf.activation, "identity")
        score = float(clf.cheat_score(clf.predict(self.crops[:1]))[0])
        self.assertAlmostEqual(score, 0.8, places=5)

    def test_preprocess_matches_notebook_tensor(self):
        crop = self.crops[1]
        import torchvision.transforms.functional as TF
        ref = TF.normalize(
            TF.resize(torch.from_numpy(np.ascontiguousarray(crop[:, :, ::-1])).permute(2, 0, 1).float() / 255,
                      [224, 224], antialias=True),
            IMAGENET_MEAN, IMAGENET_STD)
        got = preprocess([crop], 224)[0]
        self.assertTrue(torch.allclose(got, ref))

    def test_flag_times_match_notebook_state_machine(self):
        # 0.55 stays under the old 0.6 default and over a tuned 0.4 threshold.
        samples = [(i / 3, 0.55) for i in range(0, 24)]
        engine = FlagEngine(threshold=0.4)
        opened = []
        for t, s in samples:
            _smooth, ev = engine.update(t, 7, s, {"zone": "B2", "top_class": "cheating"})
            opened += ev
        self.assertEqual(notebook_flags(samples, 0.4)[0][0], opened[0][1]["start_s"])
        high = FlagEngine(threshold=0.6)
        events = []
        for t, s in samples:
            _s, ev = high.update(t, 7, s, {})
            events += ev
        self.assertEqual(events, [])
        self.assertEqual(notebook_flags(samples, 0.6), [])

    def test_live_history_is_bounded_and_file_history_is_not(self):
        live = TrackHistory(bounded=True, max_seconds=180, threshold=0.4)
        live.add(1, 0.0, 0.1, 0.1)
        live.add(1, 10.0, 0.2, 0.2)
        live.add(1, 200.0, 0.9, 0.9)
        full = TrackHistory(bounded=False, max_seconds=180, threshold=0.4)
        full.add(3, 0.0, 0.2, 0.2)
        full.add(3, 200.0, 0.8, 0.8)
        full.upsert_flag(9, 3, 2.0, 5.0)
        times = [p["t"] for p in full.snapshot()["tracks"][0]["points"]]
        self.assertEqual(times, [0.0, 200.0])
        self.assertEqual(full.window(3, 0, 10), [[0.0, 0.2]])
        self.assertEqual(full.snapshot()["flags"][0]["id"], 9)


if __name__ == "__main__":
    unittest.main()
