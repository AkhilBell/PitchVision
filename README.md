# ⚽ PitchVision — Real-Time Soccer Video Tracking & Tactical UI

PitchVision is a desktop computer vision application that ingests broadcast or tactical-cam soccer video and renders a synchronized, side-by-side dashboard featuring:

- **Broadcast-style player overlays** — team-colored ellipses, ID badges, ball marker
- **Live 2D tactical radar** — top-down pitch minimap with player dots and formation lines
- **Dynamic formation detection** — automatically identifies formations like 4-3-3, 4-4-2, etc.
- **Real-time individual heatmaps** — select any tracked player to see their positional density

---

## Prerequisites

- **Python 3.10+** (tested on 3.12)
- **macOS, Linux, or Windows**
- A soccer/football match video file (`.mp4`, `.avi`, `.mkv`, or `.mov`)

---

## Quick Start

### 1. Clone & enter the project

```bash
cd ~/Desktop/pitchvision/PitchVision
```

### 2. Create and activate the virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This installs: `ultralytics`, `supervision`, `opencv-python`, `PySide6`, `scikit-learn`, `scipy`, `numpy`, `matplotlib`, `mplsoccer`.

### 4. Download YOLO weights

The detector uses a YOLOv8 model. The first time you run the app, Ultralytics will **auto-download** `yolov8x.pt` (~130 MB) if it's not found locally. To pre-download:

```bash
source venv/bin/activate
python3 -c "from ultralytics import YOLO; YOLO('yolov8x.pt')"
```

The weights file will be cached by Ultralytics in its default directory. Alternatively, you can place a weights file manually in `models/weights/` and update `MODEL_PATH` in `src/config.py`.

### 5. Launch the application

```bash
source venv/bin/activate
python3 -m src.main
```

---

## How to Use

### Loading a Video

1. Click **"Load Video"** in the toolbar
2. Select any `.mp4`, `.avi`, `.mkv`, or `.mov` soccer match clip
3. The pipeline starts automatically — you'll see:
   - The **video feed** on the left with player overlays appearing as YOLO detects players
   - The **2D radar** (top-right) showing player positions on a pitch minimap
   - The **heatmap panel** (bottom-right) building as player positions accumulate

### Playback Controls

| Control | Action |
|---------|--------|
| **Load Video** | Open a new video file |
| **Play/Pause** | Toggle pipeline processing |
| **Export MP4** | *(Coming soon)* Save annotated video |

### Overlay Toggles (Status Bar)

| Checkbox | What it controls |
|----------|------------------|
| **Show Foot Markers** | Team-colored ellipses at players' feet + ID badges |
| **Show Formation Lines** | Formation cluster lines on the radar + formation labels |
| **Ball Trail** | Ball position marker on the video |

### Using the Heatmap

- The **"Select Player"** dropdown populates automatically as players are tracked
- Select any player ID to view their real-time positional heatmap
- The heatmap updates live frame-by-frame with Gaussian-smoothed density
- Hot (bright) areas show where the player spends the most time

---

## UI Layout

```
+-----------------------------------------------------------------------------------+
|  PitchVision Tactical Studio               [Load Video] [Play/Pause] [Export MP4] |
+-------------------------------------------------+---------------------------------+
|                                                 |  LIVE 2D TACTICAL RADAR         |
|                                                 |  +---------------------------+  |
|                                                 |  |  o---o---o     x---x---x  |  |
|          MAIN VIDEO FEED (WITH OVERLAYS)        |  |    o---o    *    x---x    |  |
|                                                 |  |  o---o---o     x---x---x  |  |
|     - Team-colored ellipses under feet          |  +---------------------------+  |
|     - Player ID badges (#7, #10, etc.)          |  Team A: 4-3-3  | Team B: 4-4-2 |
|     - Floating triangle over the ball           +---------------------------------+
|                                                 |  LIVE PLAYER HEATMAP            |
|                                                 |  Select Player: [ Player #7 v ] |
|                                                 |  +---------------------------+  |
|                                                 |  |      ..::####::..         |  |
|                                                 |  |     .::########::.        |  |
|                                                 |  |      ..::####::..         |  |
|                                                 |  +---------------------------+  |
+-------------------------------------------------+---------------------------------+
|  [x] Show Foot Markers   [x] Show Formation Lines   [x] Ball Trail   FPS: 29.8   |
+-----------------------------------------------------------------------------------+
```

---

## How the Pipeline Works

When you load a video, a background thread processes each frame through these stages:

1. **YOLO Detection** — Detects all people (class 0) and sports balls (class 32) in each frame
2. **ByteTrack Tracking** — Assigns persistent IDs to players across frames so the same player keeps the same number
3. **Ball Tracking** — Picks the highest-confidence ball detection; fills short gaps with linear interpolation
4. **Team Classification** — Extracts jersey crops, clusters dominant colors with K-Means to separate Team A / Team B / Referees (auto-fits after 30 frames)
5. **Homography Transform** — Maps pixel positions to real-world pitch coordinates (105×68 m)
6. **Formation Detection** — Clusters outfield players by position along the pitch to identify tactical shapes (4-3-3, 4-4-2, etc.)
7. **Heatmap Accumulation** — Increments a per-player 2D grid for spatial density visualization

Each processed frame is emitted as a `FramePacket` to the UI thread, which updates all three panels simultaneously.

---

## Project Structure

```
PitchVision/
├── README.md                      ← You are here
├── requirements.txt               ← Python dependencies
├── data/
│   ├── samples/                   ← Drop test video clips here
│   └── exports/                   ← Saved annotated videos / heatmaps
├── models/
│   └── weights/                   ← YOLO weights (auto-downloaded)
└── src/
    ├── main.py                    ← Entry point
    ├── config.py                  ← All tunable constants
    ├── core/
    │   ├── packet.py              ← PlayerState, BallState, FramePacket dataclasses
    │   ├── video_io.py            ← Video reader generator + MP4 exporter
    │   └── pipeline_worker.py     ← QThread orchestrating the full pipeline
    ├── vision/
    │   ├── detector.py            ← YOLO inference + ByteTrack tracking
    │   ├── ball_tracker.py        ← Ball interpolation for missing frames
    │   ├── team_classifier.py     ← Jersey color K-Means clustering
    │   └── homography.py          ← Pixel ↔ pitch coordinate transforms
    ├── analytics/
    │   ├── heatmap_engine.py      ← 2D accumulator + Gaussian smoothing
    │   └── formation_engine.py    ← Spatial clustering + formation templates
    └── ui/
        ├── main_window.py         ← PySide6 main window with toolbar/status
        ├── video_canvas.py        ← Overlay renderer (ellipses, badges, ball)
        ├── radar_widget.py        ← 2D pitch minimap with formation lines
        └── heatmap_widget.py      ← Matplotlib per-player heatmap display
```

---

## Configuration

All tunable parameters live in [`src/config.py`](src/config.py):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_PATH` | `models/weights/yolov8x.pt` | Path to YOLO weights |
| `CONFIDENCE_THRESHOLD` | `0.3` | Min detection confidence |
| `IOU_THRESHOLD` | `0.5` | NMS IoU threshold |
| `HEATMAP_SIGMA` | `2.5` | Gaussian blur for heatmaps |
| `PITCH_LENGTH` | `105.0` | Pitch length in metres |
| `PITCH_WIDTH` | `68.0` | Pitch width in metres |
| `MAX_FPS` | `30` | Playback FPS cap |

---

## Tips for Best Results

- **Use broadcast footage** with a wide-angle view of the pitch for best detection coverage
- **15–30 second clips** are ideal for initial testing
- **Tactical cam (top-down)** video produces much better homography and formation accuracy than broadcast angle
- The team classifier needs **at least 3 visible players** to fit — it auto-fits at frame 30 and re-fits every 300 frames
- Formation detection needs **≥ 6 players with valid pitch positions** (3+ per team)

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Make sure you activated the venv: `source venv/bin/activate` |
| Window doesn't appear | Ensure PySide6 is installed: `pip install PySide6` |
| No detections showing | Check that YOLO weights downloaded — run the pre-download command above |
| Slow FPS | Use `yolov8s.pt` (small) instead of `yolov8x.pt` (extra-large) in `config.py` |
| Heatmap is blank | Select a player from the dropdown — heatmap only shows after positions accumulate |

---

## Tech Stack

| Library | Role |
|---------|------|
| **Ultralytics** | YOLOv8 object detection |
| **Supervision** | ByteTrack multi-object tracking |
| **OpenCV** | Frame I/O, drawing, homography |
| **Scikit-Learn** | K-Means for jerseys & formations |
| **SciPy** | Gaussian heatmap smoothing |
| **PySide6** | Desktop UI framework (Qt) |
| **Matplotlib** | Heatmap rendering |
| **mplsoccer** | Pitch visualization utilities |
