# 🏏 Cricket AI Coach

A fully modular Python pipeline that analyses cricket batting technique from
video using **YOLOv8 pose estimation** and generates natural-language coaching
feedback via **Gemma 3n 270M** (local, on-device).

---

## Project Structure

```
cricket_ai/
├── main.py                    ← entry point
├── config.py                  ← all constants (paths, thresholds, weights)
├── requirements.txt
│
├── core/
│   ├── geometry.py            ← angle / speed / smoothing math
│   ├── models.py              ← YOLOv8 pose + object detector wrappers
│   └── batsman_selector.py    ← picks the batsman from multi-person detections
│
├── analysis/
│   ├── biomechanics.py        ← per-frame & peak measurement extraction
│   ├── scorer.py              ← converts measurements → 0-10 scores
│   ├── renderer.py            ← skeleton, trail, HUD drawing on frames
│   └── pipeline.py            ← orchestrates video → annotated output
│
├── feedback/
│   └── gemma_coach.py         ← Gemma 3n 270M coaching (rule-based fallback)
│
├── storage/
│   └── database.py            ← SQLite persistence (save / retrieve sessions)
│
└── ui/
    └── cli.py                 ← rich terminal menu
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download YOLOv8 models
They auto-download on first run via `ultralytics`:
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8m-pose.pt'); YOLO('yolov8n.pt')"
```

### 3. Download Gemma 3n 270M (GGUF)
```bash
pip install huggingface-hub
mkdir -p models
huggingface-cli download google/gemma-3n-E2B-it-GGUF \
    gemma-3n-270m-instruct.gguf \
    --local-dir ./models/
# Then rename / update config.py → GEMMA_MODEL_PATH if needed
```

> **No GPU?** The 270M model runs fine on CPU. Expect ~3-8 seconds per
> inference call on a modern laptop.

---

## Running

```bash
cd cricket_ai
python main.py
```

Menu options:
| # | Action |
|---|--------|
| 1 | Analyse a video — full pipeline |
| 2 | View all sessions for a player |
| 3 | Detailed session breakdown |
| 4 | List all players in the database |
| 5 | Delete a session |
| 0 | Exit |

---

## What gets measured

| Measurement | Ideal range | Notes |
|---|---|---|
| Knee angle | 110–140° | Front-foot drive flexion |
| Hip angle | 90–160° | Rotation at contact |
| Elbow angle | 120–160° | Arm extension / timing |
| Shoulder angle | — | Elevation of leading shoulder |
| Bat swing angle | < 25° | Deviation from vertical |
| Wrist speed | higher = more power | Proxy for bat speed |
| Stance width | 100–250 px | Stability indicator |
| Hip-shoulder offset | — | Early rotation detection |

---

## Extending

- **Custom bat detection** — swap `COCO_BAT_CLASS` in `config.py` with your
  own fine-tuned YOLOv8 class ID for a cricket bat.
- **MediaPipe backend** — replace `PoseDetector` in `core/models.py` with a
  MediaPipe `PoseLandmarker` implementation; the rest of the pipeline is
  backend-agnostic.
- **Flutter / mobile** — export `PeakBiomechanics` as JSON and call
  `GemmaCoach` via a local FastAPI endpoint, or run Gemma via MediaPipe LLM
  SDK on-device.
