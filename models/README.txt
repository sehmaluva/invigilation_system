Exam_Cheating_TPU_Training.ipynb  - Colab notebook (Runtime > Change runtime type > v5e-1 TPU > Save, then Run all).
   Edit only the settings cell. SOURCES = list of image and/or video datasets combined into one training set:
   dict(name=..., type="images"|"video", path=..., label_map={folder_or_keyword: 0|1}, multi="largest"|"all", fps=1.0, train=True, kaggle=slug)
best_stub.pt          - EfficientNet-B0 (ImageNet backbone) + UNTRAINED 2-output head. Integration only; scores are meaningless.
best_stub_6class.pt   - same with a 6-class head in the backend's default class order (scores ~0.65 -> would flag everyone; prefer best_stub.pt).
make_stub.py          - rebuilds the stubs.  src/build_nb.py - regenerates the notebook.
Use: copy a .pt to invigilation_system/models/best.pt (the loader reads arch from the file).

TRY IT (end of notebook)
- Type a path (or comma-separated paths) into TRY_PATH, or leave it empty to get an upload button.
- Pictures: YOLO person boxes, a cheating score per person, a red FLAG label at or above the threshold. Saved to /content/try_outputs/<name>_scored.jpg.
- Videos (.mp4/.avi/.mov/.mkv/.webm/.3gp): sampled at TRY_FPS (3), YOLO+ByteTrack, a 3 s rolling mean per student, flags after 2 s above the threshold (20 s cooldown), like the backend. Gives an annotated H.264 mp4 shown inline, a score plot and a list of flagged intervals.
- Model order: best.pt from OUT_DIR, then TRY_STUB (/content/best_stub.pt, upload it to Colab), then an ImageNet backbone with an untrained head.
- Threshold: TRY_THRESHOLD, or 0 to use the threshold saved in the checkpoint.
