"""Build stub checkpoints: ImageNet EfficientNet-B0 backbone + untrained head (scores are NOT meaningful)."""
import torch, torch.nn as nn
from torchvision import models
torch.manual_seed(0)
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
def make(n, class_names, cheating, out):
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, n)
    torch.save({"arch": "efficientnet_b0", "model_state_dict": {k: v.cpu() for k, v in m.state_dict().items()},
                "class_names": class_names, "cheating_classes": cheating, "head": "binary" if n == 2 else "multiclass",
                "input_size": 224, "mean": MEAN, "std": STD, "threshold": 0.5, "stub": True,
                "note": "STUB: ImageNet backbone, untrained head. Scores are meaningless."}, out)
    print("saved", out)
make(2, ["normal", "cheating"], ["cheating"], "best_stub.pt")
six = ["backwatching", "frontwatching", "normal", "showinggestures", "sidewatching", "suspiciouswatching"]
make(6, six, ["backwatching", "showinggestures", "sidewatching", "suspiciouswatching"], "best_stub_6class.pt")
