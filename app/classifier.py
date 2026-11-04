"""
Behaviour classifier: predict(crops) -> probs

The notebook (Exam_Cheating_TPU_Training.ipynb) saves best.pt as a dict:
  arch, model_state_dict, class_names, cheating_classes, threshold, input_size, mean, std.

Those fields are the source of truth when the file contains them. config.yaml is only
the fallback (and the aspect-ratio STUB used when models/best.pt is missing).

Every crop is a BGR numpy image of one person (from OpenCV).
Output: numpy array [N, num_classes] of class probabilities, in class_names order.
The cheating score is the sum of the columns named in cheating_classes (softmax for the
notebook's 2-way cross-entropy head; sigmoid only for a single-logit head).
"""
import json
import threading
import warnings

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision import models

from .config import CFG, resolve

C = CFG["classifier"]
F = CFG["flagging"]
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def pick_device():
    want = str(CFG["detector"].get("device", "auto"))
    if want == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu" if want == "cpu" else f"cuda:{want}" if want.isdigit() else want


def build_arch(arch: str, n_classes: int) -> nn.Module:
    """Same architectures as the notebook's build_model (weights come from the checkpoint)."""
    if arch == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
    else:
        raise ValueError(f"Unknown arch '{arch}' - add it to build_arch()")
    return m


def preprocess(crops, size, mean=None, std=None):
    """BGR crops -> normalised tensor [N,3,size,size].

    Same steps as the notebook Try-it `score_crops`: BGR to RGB, /255,
    squash to size x size, ImageNet normalise. No crop padding here
    (padding is applied when the box is cut out).
    """
    mean = IMAGENET_MEAN if mean is None else mean
    std = IMAGENET_STD if std is None else std
    batch = []
    for crop in crops:
        rgb = np.ascontiguousarray(crop[:, :, ::-1])
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        t = TF.resize(t, [size, size], antialias=True)
        batch.append(TF.normalize(t, mean, std))
    return torch.stack(batch)


def _output_count(state):
    mats = [v for k, v in state.items() if getattr(v, "ndim", 0) == 2 and str(k).endswith("weight")]
    if not mats:
        raise ValueError("state_dict has no linear weight; cannot tell how many classes")
    return int(mats[-1].shape[0])


def _sidecar_threshold(path):
    """metrics.json / threshold.json saved next to best.pt by the notebook."""
    folder = path.parent
    for name in ("metrics.json", "threshold.json"):
        p = folder / name
        if not p.is_file():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            print(f"[classifier] could not read {p}: {e}")
            continue
        if isinstance(data, dict) and data.get("threshold") is not None:
            return float(data["threshold"]), name
    return None


class Classifier:
    def __init__(self, model_path=None):
        self.device = pick_device()
        self.path = resolve(model_path or C["model_path"])
        self.is_stub = not self.path.exists()
        self.untrained = False
        self.arch = C.get("arch", "efficientnet_b0")
        self.class_names = [str(n) for n in C["class_names"]]
        self.cheating_classes = [str(n) for n in C["cheating_classes"]]
        self.size = int(C.get("input_size", 224))
        self.mean = list(IMAGENET_MEAN)
        self.std = list(IMAGENET_STD)
        self.activation = "stub"
        self.n_outputs = len(self.class_names)
        self.threshold = float(F["threshold"])
        self.threshold_source = "config"
        self._bind_cheat_index()

        if self.is_stub:
            self.rng = np.random.default_rng(0)
            self._log()
            return

        model, meta, known_act = self._load(self.path)
        self._apply_meta(meta)
        self.model = model.to(self.device).eval()
        if known_act is None:
            self.n_outputs, self.activation = self._inspect_output(self.model)
        else:
            self.activation = known_act
        self._reconcile_head()
        self._choose_threshold(meta)
        self._log()

    # ---------------------------------------------------------------- load
    def _load(self, path):
        """Return (module, metadata_dict, activation or None if it must be probed)."""
        ckpt = None
        try:
            ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        except Exception:
            ckpt = None
        if ckpt is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                return torch.jit.load(str(path), map_location="cpu"), {}, None

        if isinstance(ckpt, nn.Module):
            return ckpt, {}, None

        meta = ckpt if isinstance(ckpt, dict) else {}
        state = meta
        for key in ("model_state_dict", "state_dict", "model"):
            if isinstance(meta, dict) and key in meta:
                state = meta[key]
                break
        if isinstance(state, nn.Module):
            return state, meta, None

        arch = meta.get("arch", C.get("arch", "efficientnet_b0")) if isinstance(meta, dict) else C["arch"]
        n_out = _output_count(state)
        try:
            model = build_arch(arch, n_out)
        except ValueError:
            # config/checkpoint arch disagrees with a known builder; try the other one
            other = "resnet50" if arch == "efficientnet_b0" else "efficientnet_b0"
            model = build_arch(other, n_out)
            arch = other
        try:
            model.load_state_dict(state)
        except RuntimeError as e:
            raise RuntimeError(f"Could not load {path} as {arch} with {n_out} outputs: {e}") from e
        act = "sigmoid" if n_out == 1 else "softmax"
        meta = dict(meta)
        meta["arch"] = arch
        meta["_n_outputs"] = n_out
        return model, meta, act

    def _apply_meta(self, meta):
        if not isinstance(meta, dict):
            return
        if meta.get("arch"):
            self.arch = str(meta["arch"])
        if meta.get("input_size"):
            self.size = int(meta["input_size"])
        if meta.get("mean") and meta.get("std"):
            self.mean = [float(x) for x in meta["mean"]]
            self.std = [float(x) for x in meta["std"]]
        names = meta.get("class_names")
        if names:
            self.class_names = [str(n) for n in names]
            cheat = meta.get("cheating_classes") or ["cheating"]
            self.cheating_classes = [str(c) for c in cheat]
            self._bind_cheat_index()
        self.untrained = bool(meta.get("stub", False))
        if "_n_outputs" in meta:
            self.n_outputs = int(meta["_n_outputs"])

    def _reconcile_head(self):
        """class_names must line up with the head, otherwise the cheating column is wrong."""
        n = self.n_outputs
        if len(self.class_names) == n and all(c in self.class_names for c in self.cheating_classes):
            self._bind_cheat_index()
            return
        cfg_names = [str(x) for x in C["class_names"]]
        cfg_cheat = [str(x) for x in C["cheating_classes"]]
        print(f"[classifier] WARNING: class_names {self.class_names} do not match {n} outputs")
        if len(cfg_names) == n:
            self.class_names = cfg_names
            self.cheating_classes = cfg_cheat
        elif n == 2:
            # notebook binary head: index 0 = normal, index 1 = cheating
            self.class_names = ["normal", "cheating"]
            self.cheating_classes = ["cheating"]
        else:
            self.class_names = [f"class_{i}" for i in range(n)]
            self.cheating_classes = ["class_1"] if n > 1 else ["class_0"]
        self._bind_cheat_index()

    def _bind_cheat_index(self):
        missing = [c for c in self.cheating_classes if c not in self.class_names]
        if missing:
            print(f"[classifier] WARNING: cheating classes {missing} are not in {self.class_names}")
            self.cheating_classes = [c for c in self.cheating_classes if c in self.class_names]
        if not self.cheating_classes:
            if "cheating" in self.class_names:
                self.cheating_classes = ["cheating"]
            else:
                self.cheating_classes = [self.class_names[-1]]
                print(f"[classifier] WARNING: no cheating class named; using '{self.class_names[-1]}'")
        self.cheat_idx = [self.class_names.index(c) for c in self.cheating_classes]

    def _choose_threshold(self, meta):
        """Notebook: TRY_THRESHOLD or the threshold saved in best.pt, else 0.5."""
        if not F.get("use_checkpoint_threshold", True):
            self.threshold = float(F["threshold"])
            self.threshold_source = "config"
            return
        if isinstance(meta, dict) and meta.get("threshold") is not None:
            self.threshold = float(meta["threshold"])
            self.threshold_source = "checkpoint"
            return
        side = _sidecar_threshold(self.path)
        if side is not None:
            self.threshold, self.threshold_source = side
            return
        self.threshold = float(F["threshold"])
        self.threshold_source = "config"

    @torch.no_grad()
    def _inspect_output(self, model):
        """Whole-model / TorchScript exports sometimes already apply softmax."""
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        y = model(torch.zeros(1, 3, self.size, self.size, device=device))
        if isinstance(y, (tuple, list)):
            y = y[0]
        y = y.float()
        n = int(y.shape[-1]) if y.ndim >= 2 else 1
        if n == 1:
            return n, "sigmoid"
        looks_prob = bool(
            torch.all(y >= -1e-4) and torch.all(y <= 1 + 1e-4)
            and torch.allclose(y.sum(dim=-1), torch.ones(y.shape[0]), atol=1e-2)
        )
        return n, ("identity" if looks_prob else "softmax")

    def _log(self):
        fw = CFG["pipeline"].get("max_frame_width") or 0
        names = ", ".join(f"{i}:{n}" for i, n in enumerate(self.class_names))
        cheat = ", ".join(f"{i}:{self.class_names[i]}" for i in self.cheat_idx)
        where = "not found -> STUB (aspect-ratio scores, not a real model)" if self.is_stub else str(self.path)
        print("[classifier] " + "-" * 64)
        print(f"[classifier] model: {where}")
        if self.untrained:
            print("[classifier] WARNING: checkpoint is marked stub (untrained head). Scores are not meaningful.")
        print(f"[classifier] arch: {self.arch}   outputs: {self.n_outputs}   "
              f"activation: {self.activation}   device: {self.device}")
        print(f"[classifier] classes: [{names}]")
        print(f"[classifier] cheating index: [{cheat}]")
        print(f"[classifier] threshold: {self.threshold:.4f}   source: {self.threshold_source}")
        print(f"[classifier] input_size: {self.size}   mean: {self.mean}   std: {self.std}")
        print(f"[classifier] sampling_fps: {CFG['pipeline']['process_fps']}   "
              f"smooth_window: {F['window_seconds']}s   "
              f"min_duration: {F['min_duration_seconds']}s   "
              f"cooldown: {F['cooldown_seconds']}s")
        print(f"[classifier] detector_conf: {CFG['detector']['conf']}   "
              f"crop_pad: {C.get('crop_pad', 0)}   "
              f"max_frame_width: {'off' if not fw else fw}")
        print("[classifier] " + "-" * 64)

    # ---------------------------------------------------------------- predict
    def _to_probs(self, logits):
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        logits = logits.float()
        if self.activation == "sigmoid" or logits.shape[1] == 1:
            p = torch.sigmoid(logits[:, :1]).squeeze(1)
            return self._cheat_mass_to_probs(p)
        if self.activation == "identity":
            probs = logits
        else:
            probs = torch.softmax(logits, 1)
        if probs.shape[1] == len(self.class_names):
            return probs
        if probs.shape[1] == 2 and len(self.class_names) != 2:
            # Stale multi-class config, binary notebook head. Column 1 is P(cheating)
            # unless the checkpoint named a cheating class inside those two columns.
            col = self.cheat_idx[0] if self.cheat_idx and self.cheat_idx[0] < 2 else 1
            return self._cheat_mass_to_probs(probs[:, col])
        return probs

    def _cheat_mass_to_probs(self, cheat):
        """Put a single P(cheating) onto the cheating columns (sums back to `cheat`)."""
        n = len(self.class_names)
        probs = cheat.new_zeros(cheat.shape[0], n)
        other = [i for i in range(n) if i not in self.cheat_idx]
        if self.cheat_idx:
            probs[:, self.cheat_idx] = cheat.unsqueeze(1) / len(self.cheat_idx)
        if other:
            probs[:, other] = (1 - cheat).unsqueeze(1) / len(other)
        return probs

    @torch.no_grad()
    def predict(self, crops):
        """crops: list of BGR images -> probs [N, len(class_names)]"""
        n_cls = len(self.class_names)
        if len(crops) == 0:
            return np.zeros((0, n_cls), dtype=np.float32)
        if self.is_stub:
            return self._stub(crops)
        out = []
        bs = int(C.get("batch_size", 16))
        for i in range(0, len(crops), bs):
            x = preprocess(crops[i:i + bs], self.size, self.mean, self.std).to(self.device)
            out.append(self._to_probs(self.model(x)).cpu().numpy())
        return np.concatenate(out)

    def cheat_score(self, probs):
        """Sum of the cheating-class probabilities. Empty input -> empty vector."""
        if len(probs) == 0:
            return np.zeros(0, dtype=np.float32)
        idx = [i for i in self.cheat_idx if i < probs.shape[1]]
        if not idx:
            return np.zeros(len(probs), dtype=np.float32)
        return probs[:, idx].sum(1)

    def top_class(self, prob_row):
        i = int(np.argmax(prob_row))
        if i < len(self.class_names):
            return self.class_names[i]
        return str(i)

    def info(self):
        return {
            "model_path": str(C["model_path"]),
            "resolved_path": str(self.path),
            "stub": bool(self.is_stub or self.untrained),
            "aspect_stub": bool(self.is_stub),
            "untrained_head": bool(self.untrained),
            "arch": self.arch,
            "class_names": list(self.class_names),
            "cheating_classes": list(self.cheating_classes),
            "cheating_index": list(self.cheat_idx),
            "activation": self.activation,
            "n_outputs": self.n_outputs,
            "threshold": self.threshold,
            "threshold_source": self.threshold_source,
            "input_size": self.size,
            "device": self.device,
        }

    # ---------------------------------------------------------------- stub
    def _stub(self, crops):
        """STUB: wider boxes (leaning / arm out / turning) -> higher score. NOT a real model."""
        s = C["stub"]
        aspect = np.array([c.shape[1] / max(c.shape[0], 1) for c in crops])
        score = 1 / (1 + np.exp(-s["gain"] * (aspect - s["ref_aspect"] - 0.25)))
        score = np.clip(score + self.rng.normal(0, s["noise"], len(crops)), 0, 1).astype(np.float32)
        return self._cheat_mass_to_probs(torch.from_numpy(score)).numpy()


_classifier = None
_clf_lock = threading.Lock()


def get_classifier():
    """Shared instance, loaded once. The first call logs what was loaded."""
    global _classifier
    with _clf_lock:
        if _classifier is None:
            _classifier = Classifier()
    return _classifier


def get_classifier_info():
    return get_classifier().info()
