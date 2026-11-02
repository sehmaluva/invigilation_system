import nbformat as nbf
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Exam cheating classifier: TPU v5e-1 training
Trains EfficientNet-B0 (or ResNet50) on student crops and exports `best.pt` for the invigilation backend.
**Runtime → Change runtime type → v5e-1 TPU.** It falls back to a GPU or CPU if no TPU is available.
""")

md("Install the extra packages. Torch and the TPU backend come preinstalled.")
code(r'''
import os, sys, subprocess
SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"   # box test only: tiny synthetic data, CPU
if not SMOKE_TEST:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "imagehash", "kaggle", "lap"], check=True)
''')

md("""
Settings. **Edit this cell only.**
`SOURCES` is the list of datasets combined into one training set. Each entry:
- `type`: `images` (one sub-folder per class) or `video` (videos in class folders, or label from the file name)
- `path`: folder under `DRIVE_ROOT` (or an absolute path); `kaggle`: optional slug to download instead
- `label_map`: optional `{folder or keyword: 0 normal / 1 cheating}`; without it, the `NORMAL_CLASSES` rule is used
- `multi`: what to do when YOLO finds several people: `largest` or `all` (one crop each)
- `train`: `False` keeps a source for testing only
""")
code(r'''
CFG = dict(
    SOURCE="drive",               # drive | kaggle (uses each source's "kaggle" slug; upload kaggle.json)
    DRIVE_ROOT="/content/drive/MyDrive/SmartInvigilation",
    SOURCES=[
        dict(name="examdatasetnew", type="images", path="examdatasetnew/datasetnew"),
        dict(name="ExamCheatingDataset", type="images", path="archive(1)/ExamCheatingDataset/train",
             kaggle="ardutraagiginting/exam-cheating-dataset"),
        dict(name="hashemite", type="video", path="hashemite", enabled=False, fps=1.0,
             kaggle="fairouzfarouq/student-cheating-behavior-dataset",
             label_map={"not cheating": 0, "normal": 0, "exchang": 1, "looking": 1, "sheet": 1, "phone": 1, "mobile": 1}),
        # dict(name="my_images", type="images", path="my_images", label_map={"normal": 0, "copying": 1}),
        # dict(name="my_videos", type="video", path="my_videos", fps=2.0, label_map={"normal": 0, "cheat": 1}),
    ],
    OUT_DIR="/content/drive/MyDrive/SmartInvigilation_runs/tpu_effnet",
    WORK_DIR="/content/work",     # fast local disk for frames + crop cache

    # labels
    NORMAL_CLASSES=["normal", "normal act", "not cheating", "notcheating"],
    FRONTWATCHING_IS_CHEATING=False,   # try both ways
    CROSS_PAIRS=[("examdatasetnew", "ExamCheatingDataset")],   # train on one, test on the other (both ways)

    # model
    ARCH="efficientnet_b0",       # efficientnet_b0 | resnet50
    HEAD="binary",                # binary (2 outputs) | multiclass (collapsed to binary for metrics)
    IMG=224, CACHE_SIZE=256,
    BATCH=64, EPOCHS_HEAD=3, EPOCHS_FT=25, LR_HEAD=1e-3, LR_FT=1e-4, WD=1e-4,
    PATIENCE=5, LOSS="weighted_ce",  # weighted_ce | sampler
    BF16=True,
    N_FOLDS=5, FOLDS_TO_RUN=5,
    TARGET_PRECISION=0.80,

    # cleaning + cropping
    DROP_FLIP_COPY_FILES=True, DEDUP_HAMMING=4,
    YOLO_WEIGHTS="yolov8n.pt", YOLO_CONF=0.3, CROP_IF_PERSON_AREA_BELOW=0.6, CROP_PAD=0.10, MIN_CROP_PX=40,
    SEED=42,
)
if SMOKE_TEST:  # tiny CPU run on synthetic images + videos
    CFG.update(SOURCE="synthetic", DRIVE_ROOT="/tmp/smoke_data", OUT_DIR="/tmp/smoke_out", WORK_DIR="/tmp/smoke_work",
               SOURCES=[dict(name="examdatasetnew", type="images", path="examdatasetnew"),
                        dict(name="ExamCheatingDataset", type="images", path="ExamCheatingDataset"),
                        dict(name="my_images", type="images", path="my_images", label_map={"ok": 0, "copying": 1}),
                        dict(name="my_videos", type="video", path="my_videos", fps=2.0, label_map={"normal": 0, "phone": 1})],
               BATCH=8, EPOCHS_HEAD=1, EPOCHS_FT=2, N_FOLDS=3, FOLDS_TO_RUN=2, PATIENCE=1, IMG=64, CACHE_SIZE=72,
               HEAD=os.environ.get("SMOKE_HEAD", "binary"), YOLO_WEIGHTS=os.environ.get("SMOKE_YOLO", "yolov8n.pt"))
SOURCES = [s for s in CFG["SOURCES"] if s.get("enabled", True)]
''')

md("Pick the device: TPU (`torch_tpu`, then `torch_xla`), else CUDA, else CPU.")
code(r'''
import contextlib, torch
BACKEND, xm = "cpu", None
DEVICE = torch.device("cpu")
try:                                  # 2026 Colab TPU images ship TorchTPU
    import torch_tpu  # noqa: F401
    DEVICE, BACKEND = torch.device("tpu"), "torch_tpu"
except Exception:
    try:                              # older images: PyTorch/XLA
        import torch_xla, torch_xla.core.xla_model as xm
        DEVICE = torch_xla.device() if hasattr(torch_xla, "device") else xm.xla_device()
        BACKEND = "torch_xla"
    except Exception:
        if torch.cuda.is_available():
            DEVICE, BACKEND = torch.device("cuda"), "cuda"
if SMOKE_TEST:
    DEVICE, BACKEND = torch.device("cpu"), "cpu"

def autocast():
    """bf16 autocast where it works (TPU, newer GPUs)."""
    if not CFG["BF16"] or BACKEND == "cpu":
        return contextlib.nullcontext()
    if BACKEND == "cuda" and not torch.cuda.is_bf16_supported():
        return contextlib.nullcontext()
    kind = {"torch_tpu": "tpu", "torch_xla": "xla", "cuda": "cuda"}[BACKEND]
    return torch.autocast(device_type=kind, dtype=torch.bfloat16)

def sync():
    """Run the queued graph (XLA) - one call per training step."""
    if BACKEND == "torch_xla":
        (torch_xla.sync if hasattr(torch_xla, "sync") else xm.mark_step)()

def opt_step(opt):
    if BACKEND == "torch_xla":
        xm.optimizer_step(opt)          # (all-reduce is a no-op on 1 chip)
    else:
        opt.step()
    sync()

try:   # probe bf16 once; switch off if this backend rejects it
    with autocast():
        (torch.ones(2, 2, device=DEVICE) @ torch.ones(2, 2, device=DEVICE)).cpu()
except Exception as e:
    print("bf16 autocast disabled:", e); CFG["BF16"] = False
print("backend:", BACKEND, "| device:", DEVICE, "| torch", torch.__version__)
''')

md("Mount Drive (or download from Kaggle with your own `kaggle.json`; no tokens are stored in the notebook).")
code(r'''
from pathlib import Path
if CFG["SOURCE"] == "drive" or CFG["OUT_DIR"].startswith("/content/drive"):
    from google.colab import drive
    drive.mount("/content/drive")

def kaggle_download(slug, dest):
    from google.colab import files
    kj = Path.home() / ".kaggle/kaggle.json"
    if not kj.exists():
        print("Upload your kaggle.json (Kaggle → Settings → API → Create New Token)")
        up = files.upload(); kj.parent.mkdir(exist_ok=True)
        kj.write_bytes(up["kaggle.json"]); kj.chmod(0o600)
    subprocess.run(["kaggle", "datasets", "download", "-d", slug, "-p", dest, "--unzip"], check=True)

def src_root(s):
    p = Path(s["path"]) if Path(s["path"]).is_absolute() else Path(CFG["DRIVE_ROOT"]) / s["path"]
    if (CFG["SOURCE"] == "kaggle" or not p.exists()) and s.get("kaggle") and not SMOKE_TEST:
        dest = Path("/content/data") / s["name"]
        if not dest.exists(): kaggle_download(s["kaggle"], str(dest))
        print(f"{s['name']}: downloaded to {dest}. If labels look wrong, set its path to the class-folder level:")
        for q in sorted(dest.glob("*/*"))[:15]: print("   ", q)
        p = dest
    return p
for d in (CFG["OUT_DIR"], CFG["WORK_DIR"]):
    Path(d).mkdir(parents=True, exist_ok=True)
''')

md("SMOKE_TEST only: tiny fake images and videos (skipped on Colab).")
code(r'''
if SMOKE_TEST:
    import numpy as np, cv2
    from PIL import Image
    from ultralytics.utils import ASSETS
    rng = np.random.default_rng(0)
    people = [np.array(Image.open(ASSETS / f).convert("RGB")) for f in ("zidane.jpg", "bus.jpg")]
    layout = {"examdatasetnew": ["backwatching", "frontwatching", "normal", "showinggestures", "sidewatching", "suspiciouswatching"],
              "ExamCheatingDataset": ["normal act", "looking friend", "giving code"], "my_images": ["ok", "copying"]}
    for ds, classes in layout.items():
        for ci, c in enumerate(classes):
            d = Path(CFG["DRIVE_ROOT"]) / ds / c; d.mkdir(parents=True, exist_ok=True)
            for i in range(14):
                a = (rng.random((96, 80, 3)) * 60).astype("uint8"); a[:, :, ci % 3] += 120 + 10 * ci
                Image.fromarray(a).save(d / f"{i}.jpg")
            Image.fromarray(a).save(d / f"{i}_f.jpg"); Image.fromarray(a[:, ::-1]).save(d / f"mirror_{i}.jpg")
            for k in range(3):   # multi-person photos -> YOLO crops
                b = people[k % 2].astype(int) + rng.integers(-20, 20, people[k % 2].shape)
                Image.fromarray(np.clip(b, 0, 255).astype("uint8")).save(d / f"group_{k}.jpg")
    for c in ("normal", "phone"):
        d = Path(CFG["DRIVE_ROOT"]) / "my_videos" / c; d.mkdir(parents=True, exist_ok=True)
        for v in range(3):
            fr0 = cv2.resize(people[v % 2][:, :, ::-1], (320, 240))
            w = cv2.VideoWriter(str(d / f"cam{v}.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 4, (320, 240))
            for t in range(8): w.write(np.roll(fr0, 2 * t, axis=1))
            w.release()
''')

md("Collect every source into one table: images (class from folder) and videos (frames every `1/fps` s).")
code(r'''
import re, numpy as np, pandas as pd, cv2
IMG_EXT, VID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}, {".mp4", ".avi", ".mov", ".mkv", ".webm"}
norm = lambda s: re.sub(r"[_\-]+", " ", str(s).strip().lower())

def is_cheating(cls):
    c = norm(cls)
    if c in [norm(x) for x in CFG["NORMAL_CLASSES"]]: return 0
    if c == "frontwatching": return int(CFG["FRONTWATCHING_IS_CHEATING"])
    return 1

def label_of(s, cls, rel):
    lm = {norm(k): v for k, v in (s.get("label_map") or {}).items()}
    if not lm: return is_cheating(cls)
    if norm(cls) in lm: return lm[norm(cls)]
    return next((v for k, v in lm.items() if k in norm(rel)), None)   # keyword in path/file name

rows, skipped = [], []
for s in SOURCES:
    root = src_root(s)
    if not root.exists(): print("MISSING:", s["name"], root); continue
    kind, multi = s.get("type", "images"), s.get("multi", "all" if s.get("type") == "video" else "largest")
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root); cls = norm(rel.parts[0]) if len(rel.parts) > 1 else norm(p.stem)
        if kind == "images" and p.suffix.lower() in IMG_EXT:
            y = label_of(s, cls, rel)
            if y is None: skipped.append(str(rel)); continue
            rows.append(dict(path=str(p), dataset=s["name"], kind="images", cls=cls, y=y, subject=None,
                             multi=multi, train=s.get("train", True)))
        elif kind == "video" and p.suffix.lower() in VID_EXT:
            y = label_of(s, cls, rel)
            if y is None: skipped.append(str(rel)); continue
            fdir = Path(CFG["WORK_DIR"]) / "frames" / s["name"]; fdir.mkdir(parents=True, exist_ok=True)
            cap = cv2.VideoCapture(str(p)); fps = cap.get(cv2.CAP_PROP_FPS) or 25
            step, i, vid = max(int(round(fps / s.get("fps", 1.0))), 1), 0, norm(str(rel.with_suffix(""))).replace("/", "_")
            while True:
                ok, fr = cap.read()
                if not ok: break
                if i % step == 0:
                    out = fdir / f"{vid}_{i:06d}.jpg"; cv2.imwrite(str(out), fr)
                    rows.append(dict(path=str(out), dataset=s["name"], kind="video", cls=cls, y=y,
                                     subject=f"{s['name']}/{vid}", multi=multi, train=s.get("train", True)))
                i += 1
            cap.release()
df = pd.DataFrame(rows)
if skipped: print(f"no label for {len(skipped)} files (add them to label_map), e.g.", skipped[:5])
print(df.groupby(["dataset", "kind", "cls"]).agg(n=("y", "size"), y=("y", "first")).to_string())
''')

md("Drop `_f` (flipped) and `Copy` files (image sources).")
code(r'''
if CFG["DROP_FLIP_COPY_FILES"]:
    stem = df.path.map(lambda p: Path(p).stem.lower())
    bad = (df.kind == "images") & (stem.str.contains(r"copy") | stem.str.contains(r"_f(?:\s*\(\d+\))?$"))
    print("dropped flip/copy files:", int(bad.sum())); df = df[~bad].reset_index(drop=True)
''')

md("Perceptual-hash dedup of images (incl. mirrored) → duplicate groups. Video frames are grouped by video.")
code(r'''
import imagehash
from PIL import Image, ImageOps
def phashes(p):
    im = Image.open(p).convert("L")
    return int(str(imagehash.phash(im)), 16), int(str(imagehash.phash(ImageOps.mirror(im))), 16)
im_idx = np.nonzero((df.kind == "images").values)[0]
H = np.array([phashes(df.path[i]) for i in im_idx], dtype=np.uint64).reshape(-1, 2)
h, hm = H[:, 0], H[:, 1]
popcnt = (lambda a: np.bitwise_count(a)) if hasattr(np, "bitwise_count") else (lambda a: np.unpackbits(a.view(np.uint8).reshape(*a.shape, 8), -1).sum(-1))
parent = list(range(len(im_idx)))
def find(i):
    while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
    return i
for i in range(len(im_idx)):
    d = np.minimum(popcnt(h[i] ^ h[i + 1:]), popcnt(h[i] ^ hm[i + 1:]))
    for j in np.nonzero(d <= CFG["DEDUP_HAMMING"])[0] + i + 1:
        parent[find(int(j))] = find(i)
df.loc[im_idx, "subject"] = [f"img/g{find(k)}" for k in range(len(im_idx))]   # duplicates share a group (across sources too)
print("rows:", len(df), "| images:", len(im_idx), "| image dedup groups:", len(set(df.subject[im_idx])),
      "| videos:", df[df.kind == "video"].subject.nunique())
''')

md("""
YOLOv8n person crops (CPU, cached to local disk, squash-resized like the backend).
0 people → whole image; 1 person → used as-is (cropped only if small); several → `largest` or `all` per source. Video frames keep only detected people.
""")
code(r'''
from ultralytics import YOLO
yolo = YOLO(CFG["YOLO_WEIGHTS"])
cache = Path(CFG["WORK_DIR"]) / f"crops{CFG['CACHE_SIZE']}"; cache.mkdir(parents=True, exist_ok=True)
S, out_rows, n_multi = CFG["CACHE_SIZE"], [], 0
for r in df.itertuples():
    im = Image.open(r.path).convert("RGB"); W, Hh = im.size
    res = yolo.predict(im, classes=[0], conf=CFG["YOLO_CONF"], device="cpu", verbose=False)[0]
    b = res.boxes.xyxy.cpu().numpy() if len(res.boxes) else np.zeros((0, 4))
    b = b[np.minimum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]) >= CFG["MIN_CROP_PX"]]
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    if len(b) > 1:                                   # several people
        n_multi += 1
        b = b if r.multi == "all" else b[[area.argmax()]]
    elif len(b) == 1 and r.kind == "images" and area[0] / (W * Hh) >= CFG["CROP_IF_PERSON_AREA_BELOW"]:
        b = np.zeros((0, 4))                         # already a single-person crop
    crops = []
    for x1, y1, x2, y2 in b:
        px, py = CFG["CROP_PAD"] * (x2 - x1), CFG["CROP_PAD"] * (y2 - y1)
        crops.append(im.crop((max(0, x1 - px), max(0, y1 - py), min(W, x2 + px), min(Hh, y2 + py))))
    if not crops and r.kind == "images": crops = [im]
    for k, c in enumerate(crops):
        out = cache / f"{r.Index:06d}_{k}.jpg"
        c.resize((S, S), Image.BILINEAR).save(out, quality=95)
        out_rows.append({**r._asdict(), "crop": str(out)})
data = pd.DataFrame(out_rows).drop(columns=["Index"])
print("multi-person inputs:", n_multi, "| crops:", len(data)); print(data.groupby(["dataset", "y"]).size().unstack(fill_value=0))
''')

md("Class lists for the head, then grouped stratified folds over the combined training sources.")
code(r'''
from sklearn.model_selection import StratifiedGroupKFold
if CFG["HEAD"] == "binary":
    CLASS_NAMES, CHEAT_CLASSES = ["normal", "cheating"], ["cheating"]
    data["t"] = data.y
else:   # multiclass: one output per original class, collapsed to binary for metrics
    CLASS_NAMES = sorted(data[data.train].cls.unique())
    CHEAT_CLASSES = sorted(set(data[data.train & (data.y == 1)].cls))
    data = data[data.cls.isin(CLASS_NAMES)].reset_index(drop=True)
    data["t"] = data.cls.map(CLASS_NAMES.index)
CHEAT_IDX = [CLASS_NAMES.index(c) for c in CHEAT_CLASSES]
train_all = data[data.train].reset_index(drop=True)
sgkf = StratifiedGroupKFold(n_splits=CFG["N_FOLDS"], shuffle=True, random_state=CFG["SEED"])
train_all["fold"] = -1
for f, (_, va) in enumerate(sgkf.split(train_all, train_all.t, groups=train_all.subject)):
    train_all.loc[va, "fold"] = f
print(CLASS_NAMES, "| cheating:", CHEAT_CLASSES)
print(pd.crosstab(train_all.fold, [train_all.dataset, train_all.y]))
''')

md("Augmentations: random crop, flip, rotation, colour, blur, downscale-upscale, JPEG.")
code(r'''
import io, random
import torchvision.transforms as T
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
class DownUp:      # looks like a distant student
    def __call__(self, im):
        if random.random() > 0.3: return im
        s = random.randint(CFG["IMG"] // 4, CFG["IMG"] // 2)
        return im.resize((s, s), Image.BILINEAR).resize(im.size, Image.BILINEAR)
class Jpeg:        # camera compression
    def __call__(self, im):
        if random.random() > 0.3: return im
        b = io.BytesIO(); im.save(b, "JPEG", quality=random.randint(30, 90)); b.seek(0)
        return Image.open(b).convert("RGB")
train_tf = T.Compose([T.RandomResizedCrop(CFG["IMG"], scale=(0.6, 1.0)), T.RandomHorizontalFlip(),
                      T.RandomRotation(10), T.ColorJitter(0.3, 0.3, 0.2, 0.03),
                      T.RandomApply([T.GaussianBlur(5, (0.1, 2.0))], p=0.2), DownUp(), Jpeg(),
                      T.ToTensor(), T.Normalize(MEAN, STD)])
eval_tf = T.Compose([T.Resize((CFG["IMG"], CFG["IMG"])), T.ToTensor(), T.Normalize(MEAN, STD)])
''')

md("Dataset and loaders. Fixed batch shapes (`drop_last`, padded eval) so XLA compiles once.")
code(r'''
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
class Crops(Dataset):
    def __init__(self, d, tf): self.p, self.t, self.tf = d.crop.tolist(), d.t.tolist(), tf
    def __len__(self): return len(self.p)
    def __getitem__(self, i): return self.tf(Image.open(self.p[i]).convert("RGB")), self.t[i]

NW = min(8, os.cpu_count() or 2)
def train_loader(d):
    ds = Crops(d, train_tf)
    if CFG["LOSS"] == "sampler":
        w = 1.0 / d.t.map(d.t.value_counts()).values
        return DataLoader(ds, CFG["BATCH"], sampler=WeightedRandomSampler(w, len(d), True), drop_last=True, num_workers=NW)
    return DataLoader(ds, CFG["BATCH"], shuffle=True, drop_last=True, num_workers=NW)

@torch.no_grad()
def predict(model, d):
    """Returns P(cheating) per image; pads the last batch to a fixed size."""
    model.eval(); out = []
    for x, _ in DataLoader(Crops(d, eval_tf), CFG["BATCH"], num_workers=NW):
        n = len(x)
        if n < CFG["BATCH"]: x = torch.cat([x, x.new_zeros(CFG["BATCH"] - n, *x.shape[1:])])
        with autocast():
            p = torch.softmax(model(x.to(DEVICE)).float(), 1)[:, CHEAT_IDX].sum(1)
        sync(); out.append(p.cpu()[:n])
    return torch.cat(out).numpy()
''')

md("Model: ImageNet backbone + new head (same layout as the backend's `build_arch`).")
code(r'''
import torch.nn as nn
from torchvision import models
def build_model(n):
    if CFG["ARCH"] == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2); m.fc = nn.Linear(m.fc.in_features, n)
        head = m.fc
    else:
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n); head = m.classifier
    return m, head
''')

md("Metrics: precision, recall, macro-F1, AP, confusion matrix; threshold for the target precision.")
code(r'''
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, confusion_matrix, precision_recall_curve
def metrics(y, p, thr=0.5):
    yh = (p >= thr).astype(int)
    return dict(threshold=float(thr), precision=float(precision_score(y, yh, zero_division=0)),
                recall=float(recall_score(y, yh, zero_division=0)), macro_f1=float(f1_score(y, yh, average="macro", zero_division=0)),
                ap=float(average_precision_score(y, p)) if len(set(y)) > 1 else float("nan"),
                confusion=confusion_matrix(y, yh, labels=[0, 1]).tolist(), n=int(len(y)))
def threshold_for_precision(y, p, target):
    pr, rc, th = precision_recall_curve(y, p)
    ok = np.nonzero(pr[:-1] >= target)[0]
    return float(th[ok[np.argmax(rc[ok])]]) if len(ok) else 0.5
''')

md("Two-stage training: frozen backbone (head only), then full fine-tune; AdamW, cosine, early stopping on val macro-F1.")
code(r'''
import copy, time
def train(tr, va=None, epochs_ft=None):
    torch.manual_seed(CFG["SEED"]); random.seed(CFG["SEED"]); np.random.seed(CFG["SEED"])
    model, head = build_model(len(CLASS_NAMES)); model.to(DEVICE)
    cw = None
    if CFG["LOSS"] == "weighted_ce":
        cnt = np.bincount(tr.t, minlength=len(CLASS_NAMES)).astype(float)
        cw = torch.tensor(cnt.sum() / np.maximum(cnt, 1) / len(cnt), dtype=torch.float32, device=DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)
    dl = train_loader(tr)
    best, best_state, best_ep, bad, hist = -1, None, 0, 0, []
    stages = [("head", CFG["EPOCHS_HEAD"], CFG["LR_HEAD"]), ("ft", epochs_ft or CFG["EPOCHS_FT"], CFG["LR_FT"])]
    for stage, n_ep, lr in stages:
        for p in model.parameters(): p.requires_grad = (stage == "ft")
        for p in head.parameters(): p.requires_grad = True
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=CFG["WD"])
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(n_ep * len(dl), 1))
        for ep in range(n_ep):
            model.train(); t0 = time.time(); tot = torch.zeros((), device=DEVICE)
            for x, t in dl:
                x, t = x.to(DEVICE), t.to(DEVICE)
                with autocast():
                    loss = loss_fn(model(x).float(), t)
                opt.zero_grad(); loss.backward(); opt_step(opt); sch.step()
                tot += loss.detach()
            rec = dict(stage=stage, epoch=ep, loss=float(tot.cpu()) / max(len(dl), 1), sec=round(time.time() - t0, 1))
            if va is not None and stage == "ft":
                rec["val_macro_f1"] = metrics(va.y.values, predict(model, va))["macro_f1"]
                if rec["val_macro_f1"] > best:
                    best, best_ep, bad = rec["val_macro_f1"], ep + 1, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    bad += 1
            print(rec); hist.append(rec)
            if va is not None and stage == "ft" and bad >= CFG["PATIENCE"]: break
    if best_state is not None: model.load_state_dict(best_state)
    return model, best_ep or (epochs_ft or CFG["EPOCHS_FT"]), hist
''')

md("Grouped cross-validation → out-of-fold predictions, threshold, metrics.")
code(r'''
oof = np.full(len(train_all), np.nan); fold_res, best_eps = [], []
for f in range(min(CFG["FOLDS_TO_RUN"], CFG["N_FOLDS"])):
    tr, va = train_all[train_all.fold != f], train_all[train_all.fold == f]
    print(f"--- fold {f}: train {len(tr)}  val {len(va)}")
    model, ep, hist = train(tr, va)
    oof[va.index] = predict(model, va); best_eps.append(ep)
    fold_res.append(dict(fold=f, best_epoch=ep, **metrics(va.y.values, oof[va.index])))
m = ~np.isnan(oof)
THRESH = threshold_for_precision(train_all.y.values[m], oof[m], CFG["TARGET_PRECISION"])
cv = dict(at_0_5=metrics(train_all.y.values[m], oof[m]), at_target=metrics(train_all.y.values[m], oof[m], THRESH))
print("OOF @0.5:", cv["at_0_5"]); print(f"OOF @{THRESH:.3f} (target precision {CFG['TARGET_PRECISION']}):", cv["at_target"])
train_all.assign(p_cheat=oof).to_csv(f"{CFG['OUT_DIR']}/oof_predictions.csv", index=False)
''')

md("Cross-dataset test: train on A, test on B (and back).")
code(r'''
cross = {}
present = set(data.dataset)
for a, b in CFG["CROSS_PAIRS"]:
    if a not in present or b not in present: print("skip cross pair", a, b); continue
    for A, B in [(a, b), (b, a)]:
        tr, te = data[data.dataset == A], data[data.dataset == B]
        model, _, _ = train(tr.reset_index(drop=True), epochs_ft=int(np.median(best_eps)))
        p = predict(model, te.reset_index(drop=True))
        cross[f"{A}->{B}"] = dict(at_0_5=metrics(te.y.values, p), at_cv_threshold=metrics(te.y.values, p, THRESH))
        print(A, "->", B, cross[f"{A}->{B}"]["at_0_5"])
''')

md("Final model on all training data (median best epoch count), then save `best.pt` + `metrics.json`.")
code(r'''
import json
final, _, _ = train(train_all, epochs_ft=int(np.median(best_eps)))
ckpt = {"arch": CFG["ARCH"], "model_state_dict": {k: v.detach().cpu() for k, v in final.state_dict().items()},
        "class_names": CLASS_NAMES, "cheating_classes": CHEAT_CLASSES, "head": CFG["HEAD"],
        "input_size": 224, "mean": MEAN, "std": STD, "threshold": THRESH, "cfg": CFG}
torch.save(ckpt, f"{CFG['OUT_DIR']}/best.pt")   # state_dict already on CPU (safe for XLA/TPU tensors)
json.dump(dict(backend=BACKEND, cv_folds=fold_res, cv=cv, cross_dataset=cross, threshold=THRESH, class_names=CLASS_NAMES,
               cheating_classes=CHEAT_CLASSES, config=CFG), open(f"{CFG['OUT_DIR']}/metrics.json", "w"), indent=2, default=str)
print("saved", CFG["OUT_DIR"])
''')

md("Export: config snippet for the backend's `config.yaml` (copy into the `classifier:` section).")
code(r'''
snippet = f"""classifier:
  model_path: models/best.pt
  arch: {CFG['ARCH']}
  input_size: 224
  class_names: {json.dumps(CLASS_NAMES)}
  cheating_classes: {json.dumps(CHEAT_CLASSES)}
# flagging.threshold: start near {THRESH:.2f} (precision ~{CFG['TARGET_PRECISION']} on held-out folds), then tune on video
"""
open(f"{CFG['OUT_DIR']}/config_snippet.yaml", "w").write(snippet); print(snippet)
''')

md("Check: load `best.pt` exactly like the backend does and score a few crops.")
code(r'''
ck = torch.load(f"{CFG['OUT_DIR']}/best.pt", map_location="cpu", weights_only=False)
m2, _ = build_model(ck["model_state_dict"][[k for k, v in ck["model_state_dict"].items() if k.endswith("weight") and v.ndim == 2][-1]].shape[0])
m2.load_state_dict(ck["model_state_dict"]); m2.eval()
x = torch.stack([eval_tf(Image.open(p).convert("RGB")) for p in train_all.crop[:4]])
print(torch.softmax(m2(x), 1).detach().numpy().round(3), ck["class_names"])
''')

md("""
## Try it: test a picture or a video
Works on its own after a restart (run the first two cells and this section). Uses `best.pt`, or `best_stub.pt` if it isn't trained yet.
""")

md("Try-it settings. Leave `TRY_PATH` empty to get an upload button.")
code(r'''
TRY_PATH = ""  #@param {type:"string"}
TRY_MODEL = "/content/drive/MyDrive/SmartInvigilation_runs/tpu_effnet/best.pt"  #@param {type:"string"}
TRY_STUB = "/content/best_stub.pt"  #@param {type:"string"}
TRY_THRESHOLD = 0.0  #@param {type:"number"}
TRY_FPS = 3.0  #@param {type:"number"}
from pathlib import Path
WINDOW_S, MIN_DURATION_S, COOLDOWN_S = 3.0, 2.0, 20.0   # same smoothing rules as the backend
if "CFG" in globals(): TRY_MODEL = f"{CFG['OUT_DIR']}/best.pt"
if SMOKE_TEST:
    TRY_PATH, TRY_MODEL, TRY_STUB = os.environ.get("TRY_PATH", ""), os.environ.get("TRY_MODEL", TRY_MODEL), os.environ.get("TRY_STUB", TRY_STUB)
    TRY_THRESHOLD = float(os.environ.get("TRY_THRESHOLD", 0))
TRY_OUT = Path("/tmp/smoke_out/try" if SMOKE_TEST else "/content/try_outputs"); TRY_OUT.mkdir(parents=True, exist_ok=True)
''')

md("Load the model like the backend does (best.pt → stub → ImageNet backbone with untrained head).")
code(r'''
import os, numpy as np, torch, torch.nn as nn, cv2
import torchvision.transforms.functional as TF
from torchvision import models
from pathlib import Path
from ultralytics import YOLO
def make_arch(arch, n, pretrained=False):
    if arch == "resnet50":
        m = models.resnet50(weights="IMAGENET1K_V2" if pretrained else None); m.fc = nn.Linear(m.fc.in_features, n)
    else:
        m = models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None); m.classifier[1] = nn.Linear(m.classifier[1].in_features, n)
    return m
def load_try_model():
    for p in (TRY_MODEL, TRY_STUB):
        if p and Path(p).exists():
            ck = torch.load(p, map_location="cpu", weights_only=False); sd = ck["model_state_dict"]
            n = [v for k, v in sd.items() if k.endswith("weight") and v.ndim == 2][-1].shape[0]
            m = make_arch(ck.get("arch", "efficientnet_b0"), n); m.load_state_dict(sd)
            print("model:", p, "| stub:", ck.get("stub", False)); return m, ck
    print("No best.pt or best_stub.pt found → ImageNet backbone + untrained head (scores are meaningless)")
    return make_arch("efficientnet_b0", 2, True), {"class_names": ["normal", "cheating"], "cheating_classes": ["cheating"], "threshold": 0.5, "stub": True}
TRY_DEV = "cuda" if torch.cuda.is_available() else "cpu"     # inference on CPU/GPU avoids XLA recompiles
tmodel, tck = load_try_model(); tmodel = tmodel.to(TRY_DEV).eval()
T_IDX = [tck["class_names"].index(c) for c in tck["cheating_classes"]]
THR = TRY_THRESHOLD or float(tck.get("threshold", 0.5))
tyolo = YOLO(CFG["YOLO_WEIGHTS"] if "CFG" in globals() else "yolov8n.pt")
AREA_FULL = CFG["CROP_IF_PERSON_AREA_BELOW"] if "CFG" in globals() else 0.6

@torch.no_grad()
def score_crops(crops_bgr):
    """BGR crops → P(cheating), same preprocessing as the backend."""
    if not crops_bgr: return np.zeros(0)
    x = torch.stack([TF.normalize(TF.resize(torch.from_numpy(np.ascontiguousarray(c[:, :, ::-1])).permute(2, 0, 1).float() / 255,
                                            [224, 224], antialias=True), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) for c in crops_bgr])
    return torch.softmax(tmodel(x.to(TRY_DEV)), 1)[:, T_IDX].sum(1).cpu().numpy()
print("threshold:", round(THR, 3))
''')

md("Pick the file: the path box above, or upload.")
code(r'''
if not TRY_PATH:
    from google.colab import files
    up = files.upload(); TRY_PATH = str(Path("/content") / next(iter(up)))
    Path(TRY_PATH).write_bytes(up[Path(TRY_PATH).name])
print("testing:", TRY_PATH)
''')

md("Picture: person boxes, score per person, FLAG above the threshold.")
code(r'''
import matplotlib.pyplot as plt
def draw(img, box, text, flag):
    x1, y1, x2, y2 = map(int, box); col = (0, 0, 255) if flag else (0, 200, 0); fs = max(0.5, img.shape[1] / 1100)
    cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 2); y = max(th + 6, y1)
    cv2.rectangle(img, (x1, y - th - 6), (x1 + tw + 4, y), col, -1)
    cv2.putText(img, text, (x1 + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2)
def test_image(path):
    img = cv2.imread(path); H, W = img.shape[:2]
    b = tyolo.predict(img, classes=[0], conf=0.3, verbose=False)[0].boxes.xyxy.cpu().numpy()
    if len(b) == 0 or (len(b) == 1 and (b[0, 2] - b[0, 0]) * (b[0, 3] - b[0, 1]) / (W * H) >= AREA_FULL):
        b = np.array([[0, 0, W - 1, H - 1]], float)      # one student already fills the picture
    s = score_crops([img[int(y1):int(y2), int(x1):int(x2)] for x1, y1, x2, y2 in b])
    out = img.copy()
    for i, (box, sc) in enumerate(zip(b, s)):
        draw(out, box, f"P{i + 1} {sc:.2f}" + (" FLAG" if sc >= THR else ""), sc >= THR)
        print(f"person {i + 1}: cheating score {sc:.2f}" + ("  -> FLAG for review" if sc >= THR else ""))
    plt.figure(figsize=(10, 7)); plt.imshow(out[:, :, ::-1]); plt.axis("off"); plt.show()
    cv2.imwrite(str(TRY_OUT / (Path(path).stem + "_scored.jpg")), out); print("saved:", TRY_OUT / (Path(path).stem + "_scored.jpg"))
''')

md("Video: ~3 fps, YOLO + ByteTrack, rolling-mean smoothing per student, annotated mp4, score plot, flagged intervals.")
code(r'''
import subprocess, shutil, collections
from IPython.display import Video, display
def test_video(path):
    cap = cv2.VideoCapture(path); fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(int(round(fps / TRY_FPS)), 1); raw = str(TRY_OUT / (Path(path).stem + "_scored_raw.mp4"))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), fps / step, (W, H))
    hist, win, state, flags, i = collections.defaultdict(list), collections.defaultdict(collections.deque), {}, [], 0
    tyolo.predictor = None                                    # fresh tracker per video
    while True:
        ok, fr = cap.read()
        if not ok: break
        if i % step == 0:
            t = i / fps
            r = tyolo.track(fr, classes=[0], conf=0.3, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
            if r.boxes.id is not None:
                bx, ids = r.boxes.xyxy.cpu().numpy(), r.boxes.id.int().cpu().tolist()
                sc = score_crops([fr[int(y1):int(y2), int(x1):int(x2)] for x1, y1, x2, y2 in bx])
                for box, tid, s in zip(bx, ids, sc):
                    w = win[tid]; w.append((t, float(s)))
                    while w[0][0] < t - WINDOW_S: w.popleft()
                    sm = sum(v for _, v in w) / len(w); hist[tid].append((t, float(s), sm))
                    st = state.setdefault(tid, dict(above=None, open=None, cool=-1))
                    if sm >= THR:
                        st["above"] = t if st["above"] is None else st["above"]
                        if st["open"]: st["open"]["end"], st["open"]["peak"] = t, max(st["open"]["peak"], sm)
                        elif t - st["above"] >= MIN_DURATION_S and t >= st["cool"]:
                            st["open"] = dict(track=tid, start=st["above"], end=t, peak=sm); flags.append(st["open"])
                    else:
                        if st["open"]: st["open"], st["cool"] = None, t + COOLDOWN_S
                        st["above"] = None
                    draw(fr, box, f"#{tid} {sm:.2f}" + (" FLAG" if st["open"] else ""), bool(st["open"]))
            vw.write(fr)
        i += 1
    cap.release(); vw.release()
    out = raw.replace("_raw", "")
    if shutil.which("ffmpeg"):                                # browser-playable H.264
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-c:v", "libx264", "-pix_fmt", "yuv420p", out], check=True)
    else:
        out = raw
    print("annotated video:", out)
    display(Video(out, embed=True, width=720))
    plt.figure(figsize=(10, 4))
    for tid, h in hist.items():
        plt.plot([a for a, _, _ in h], [c for _, _, c in h], label=f"#{tid}")
    plt.axhline(THR, ls="--", c="r", label="threshold"); plt.xlabel("time (s)"); plt.ylabel("smoothed cheating score")
    plt.legend(ncol=6, fontsize=8); plt.show()
    print(f"{len(flags)} flagged interval(s):" if flags else "No flags.")
    for f in flags: print(f"  student #{f['track']}: {f['start']:.1f}s - {f['end']:.1f}s  (peak {f['peak']:.2f})")
    return flags
''')

md("Run the test on the chosen file.")
code(r'''
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".3gp"}
for p in [q.strip() for q in TRY_PATH.split(",") if q.strip()]:
    test_video(p) if Path(p).suffix.lower() in VIDEO_EXT else test_image(p)
''')

md("""
**Notes**
- To add your own photos or videos, add one line to `SOURCES` (type, path, label_map) and re-run.
- To test a file, use the **Try it** section (picture or video).
- Copy `best.pt` into the backend's `models/` folder and paste the snippet into `config.yaml`.
- Report precision, recall, AP and false alarms, not accuracy. Cross-dataset scores will be lower; that is a finding, not a failure.
- **Declare AI assistance** in your submission (module rule: AI-assisted work must be declared).
- Cite the datasets: CUI-EXAM, IEEE Access 2021, doi:10.1109/ACCESS.2021.3068223; ExamCheatingDataset (Kaggle, ardutraagiginting); Hashemite video dataset, MDPI *Data* 2022, 7(9):122 (if used).
""")

nb = nbf.v4.new_notebook(); nb.cells = cells
nb.metadata = {"accelerator": "TPU", "colab": {"provenance": [], "gpuType": "V5E1"},
               "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}}
nbf.write(nb, "/workspace/colab_tpu/Exam_Cheating_TPU_Training.ipynb")
print("cells:", len(cells))
