"""
Behaviour classifier:  predict(crops) -> probs

* If classifier.model_path (default models/best.pt) exists, Nigel's model is loaded.
* Otherwise a STUB is used so the rest of the system can be built and demoed.

Every crop is a BGR numpy image of one person (from OpenCV).
Output: numpy array [N, num_classes] of class probabilities, in config class order.
"""
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision import models

from .config import CFG, resolve

C = CFG["classifier"]
CLASS_NAMES = C["class_names"]
CHEAT_IDX = [CLASS_NAMES.index(c) for c in C["cheating_classes"]]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]   # ImageNet normalisation


def pick_device():
    want = str(CFG["detector"].get("device", "auto"))
    if want == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu" if want == "cpu" else f"cuda:{want}" if want.isdigit() else want


def build_arch(arch: str, n_classes: int) -> nn.Module:
    """Same architectures as train.py (only needed when best.pt is a state_dict)."""
    if arch == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
    else:
        raise ValueError(f"Unknown arch '{arch}' - add it to build_arch()")
    return m


def preprocess(crops, size):
    """BGR crops -> normalised tensor [N,3,size,size] (matches train.py)."""
    batch = []
    for crop in crops:
        rgb = np.ascontiguousarray(crop[:, :, ::-1])
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        t = TF.resize(t, [size, size], antialias=True)
        batch.append(TF.normalize(t, MEAN, STD))
    return torch.stack(batch)


class Classifier:
    def __init__(self):
        self.device = pick_device()
        self.size = C["input_size"]
        path = resolve(C["model_path"])
        self.is_stub = not path.exists()
        if self.is_stub:
            print(f"[classifier] {path} not found -> using STUB model (scores are NOT real)")
            self.rng = np.random.default_rng(0)
            return
        self.model = self._load(path).to(self.device).eval()
        print(f"[classifier] loaded {path} on {self.device}")

    def _load(self, path):
        # 1) TorchScript export
        try:
            return torch.jit.load(str(path), map_location="cpu")
        except Exception:
            pass
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        # 2) whole pickled model
        if isinstance(ckpt, nn.Module):
            return ckpt
        # 3) checkpoint dict / plain state_dict
        state = ckpt
        if isinstance(ckpt, dict):
            for key in ("model_state_dict", "state_dict", "model"):
                if key in ckpt:
                    state = ckpt[key]
                    break
            if isinstance(state, nn.Module):
                return state
        arch = ckpt.get("arch", C["arch"]) if isinstance(ckpt, dict) else C["arch"]
        # number of outputs = rows of the last weight matrix
        last_w = [v for k, v in state.items() if k.endswith("weight") and v.ndim == 2][-1]
        model = build_arch(arch, last_w.shape[0])
        model.load_state_dict(state)
        return model

    @torch.no_grad()
    def predict(self, crops):
        """crops: list of BGR images -> probs [N, len(CLASS_NAMES)]"""
        if len(crops) == 0:
            return np.zeros((0, len(CLASS_NAMES)), dtype=np.float32)
        if self.is_stub:
            return self._stub(crops)
        out = []
        bs = C["batch_size"]
        for i in range(0, len(crops), bs):
            x = preprocess(crops[i:i + bs], self.size).to(self.device)
            logits = self.model(x)
            if logits.shape[1] == 1:                       # binary model with one logit
                p = torch.sigmoid(logits)
                logits_p = torch.cat([1 - p, p], 1)
            else:
                logits_p = torch.softmax(logits, 1)
            out.append(logits_p.cpu().numpy())
        probs = np.concatenate(out)
        if probs.shape[1] == 2 and len(CLASS_NAMES) != 2:
            # binary model: column 1 = "cheating" -> spread over cheating classes
            probs = self._binary_to_classes(probs[:, 1])
        return probs

    # ---------------- STUB ----------------
    def _stub(self, crops):
        """STUB: wider boxes (leaning / arm out / turning) -> higher score. NOT a real model."""
        s = C["stub"]
        aspect = np.array([c.shape[1] / max(c.shape[0], 1) for c in crops])
        score = 1 / (1 + np.exp(-s["gain"] * (aspect - s["ref_aspect"] - 0.25)))
        score = np.clip(score + self.rng.normal(0, s["noise"], len(crops)), 0, 1)
        return self._binary_to_classes(score)

    def _binary_to_classes(self, cheat):
        n = len(CLASS_NAMES)
        probs = np.zeros((len(cheat), n), dtype=np.float32)
        other = [i for i in range(n) if i not in CHEAT_IDX]
        probs[:, CHEAT_IDX] = cheat[:, None] / max(len(CHEAT_IDX), 1)
        probs[:, other] = (1 - cheat[:, None]) / max(len(other), 1)
        return probs


def cheat_score(probs):
    """Binary score = total probability of the 'cheating' classes."""
    return probs[:, CHEAT_IDX].sum(1)


def top_class(prob_row):
    return CLASS_NAMES[int(np.argmax(prob_row))]


def get_classifier_info():
    path = resolve(C["model_path"])
    return {"model_path": str(C["model_path"]), "stub": not path.exists(),
            "cheating_classes": C["cheating_classes"]}
