# 🧍 Smart Human Detection System

A real-time **human (person) detection dashboard** built with Python, OpenCV,
Ultralytics YOLO and Streamlit. It watches your webcam, draws a bounding box
around every person it sees, counts them live, and lets you capture photos —
all from a modern dark dashboard that runs entirely on your laptop (CPU only,
no GPU required).

---

## ✨ Features

| # | Feature | Notes |
|---|---------|-------|
| 1 | Professional dark dashboard | Custom HTML/CSS styling, metric cards, status pills |
| 2 | Real-time webcam detection | Background worker thread keeps the UI responsive |
| 3 | Pretrained YOLO weights | `yolov8n.pt` downloaded from the official Ultralytics GitHub release |
| 4 | Bounding boxes + labels | Teal boxes with per-person confidence ("Person 87%") |
| 5 | Live people counter | "People now" + peak + frames-with-people statistics |
| 6 | Start / Stop controls | Stop releases the webcam within ~3 seconds (no blocking loops) |
| 7 | Adjustable confidence | Sidebar slider 0.05 – 0.95, applies live without restarting |
| 8 | Photo capture | Saves annotated JPEGs to `captures/` with one click |
| 9 | Live camera status pill | `● LIVE` / `■ STOPPED` / `✕ ERROR` in the header and controls |
| 10 | Detection statistics | FPS, inference time, session time, occupancy %, totals |
| 11 | Friendly error messages | Plain-language hints for camera/model/file problems |
| 12 | Guaranteed cleanup | `threading.Event` + `join()` + `VideoCapture.release()` on stop |
| 13 | Responsive layout | Wide layout, three-column controls, expandable detail panels |
| 14 | Webcam fallback | Auto-probes camera indexes 0–3; image/video file sources instead |

---

## 🖥️ Requirements

* Windows 11 (works on Windows 10 / macOS / Linux too)
* Python **3.10 – 3.12** (a 3.12.10 virtual environment is used here)
* A laptop webcam (can be skipped — see *Fallback modes* below)
* ~400 MB disk for dependencies + 6 MB model weights

---

## 📦 Installation

```powershell
# 1. Open a terminal in the project folder, then:
py -3.12 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

That's it — the YOLO weights (`yolov8n.pt`, 6.2 MB) download automatically
from the **official Ultralytics release** the first time you start the app.

### Dependencies (`requirements.txt`)

* `streamlit` — dashboard & web UI
* `opencv-python` — webcam capture & drawing
* `ultralytics` — YOLO inference pipeline
* `torch` / `torchvision` (CPU build) — deep-learning backend
* `numpy` — array handling

---

## ▶️ Running the application

Easiest (double-click):

```powershell
.\run_app.bat
```

Manually:

```powershell
.\.venv\Scripts\activate
streamlit run app.py
```

Then open **http://localhost:8501** (a browser tab opens automatically).

**How to use:** pick a confidence value → press **▶️ Start camera** →
people get boxes and a live count → **📸 Capture photo** saves a frame →
**⏹ Stop camera** releases the webcam for other apps.

---

## 🗂️ Project structure

```
Human-Detection/
├── app.py               # Streamlit dashboard (UI, controls, statistics)
├── camera_worker.py     # Background capture+detection thread (never blocks the UI)
├── detector.py          # YOLO wrapper: model loading, person-only detection, drawing
├── requirements.txt     # Pinned-compatible dependencies (CPU torch build)
├── README.md            # This file
├── run_app.bat          # One-click launcher (creates venv + installs if needed)
├── yolov8n.pt           # Pretrained weights (auto-downloaded, 6.2 MB)
├── captures/            # Photos you save from the app
├── tests/
│   ├── test_detector.py # 9 tests: model loading, detection, confidence, errors
│   ├── test_worker.py   # 9 tests: lifecycle, stop speed, capture, webcam (if present)
│   └── data/            # Sample image/video used by the tests
└── .venv/               # Isolated Python environment
```

---

## ✅ Testing

```powershell
.\.venv\Scripts\activate
python -m unittest discover -s tests -v
```

All **18 tests** pass on this machine, including a **physical-webcam test**
(this laptop's camera: 1280×720, index 0). Tests are offline-friendly: they
run against a sample photo and a generated video when no camera is present
(the webcam test skips itself instead of failing).

---

## 🔁 Fallback modes (no webcam required)

In the sidebar choose **Image file** or **Video file** instead of *Webcam*:
the same detection pipeline processes the uploaded file, so you can develop
and demo on machines without a camera. Videos loop automatically; still
images repeat as a live source.

---

## 🛠️ Troubleshooting

| Symptom | Fix |
|---|---|
| "No webcam was found" | Close Zoom/Teams/browsers using the camera; check **Settings → Privacy & security → Camera** (allow desktop apps) |
| Wrong camera | Change **Camera index** in the sidebar (0 = built-in) |
| Model load error | First run needs internet to fetch `yolov8n.pt`; check your connection and rerun |
| Black / very slow video | CPU inference ≈ 0.1 s/frame; close heavy apps, use `yolov8n` (default) |
| Port already in use | `streamlit run app.py --server.port 8502` |
| Stop button feels slow | At most ~3 s (join timeout) while a frame finishes; the pill turns `■ STOPPED` when the camera is released |

---

## ⚠️ Limitations

* **CPU-only inference:** ~7–10 FPS warm (measured 95–126 ms/frame on
  `bus.jpg`, 494–737 ms/frame end-to-end at 1280×720 on this laptop). Not
  GPU-accelerated; add `torch.cuda` wheels for real-time on high-res video.
* Detects **persons only** (COCO class `person`); no pose/skeleton tracking,
  no re-identification across time — the counter counts people *per frame*.
* One browser session owns the camera; a second tab will fail to open it.
* Streamlit refresh cadence ≈ 0.05 s + inference time; button clicks are
  processed between frames, so controls can lag by up to ~1 frame.
* Photos are saved as annotated JPEGs only (raw frames are not kept).
* No persistence: statistics reset when the browser page is reloaded.

---

## 🔐 Notes

* No credentials or personal information are hardcoded.
* Model weights come only from `github.com/ultralytics/assets` releases.
* Images you capture are saved **locally** in `captures/` and never uploaded.

---

## 📄 License

Project code is yours to use. YOLO weights and models follow the
[Ultralytics AGPL-3.0 license](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).