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
git clone https://github.com/AkhilBell/PitchVision.git
cd PitchVision
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

This installs: `ultralytics`, `supervision`, `trackers`, `opencv-python`, `PySide6`, `scikit-learn`, `scipy`, `numpy`, `matplotlib`, `mplsoccer`.

### 4. Download the model weights

PitchVision uses three soccer-trained YOLOv8 models published with the open-source [roboflow/sports](https://github.com/roboflow/sports) project. They are about 137–140 MB each and are not stored in this repository. Download them once into `models/weights/`, from the project root:

```bash
curl -L "https://drive.usercontent.google.com/download?id=17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q&export=download&confirm=t" \
     -o models/weights/football-player-detection.pt
curl -L "https://drive.usercontent.google.com/download?id=1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V&export=download&confirm=t" \
     -o models/weights/football-ball-detection.pt
curl -L "https://drive.usercontent.google.com/download?id=1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf&export=download&confirm=t" \
     -o models/weights/football-pitch-detection.pt
```

| File | What it finds |
|------|---------------|
| `football-player-detection.pt` | Players, goalkeepers and referees on the pitch (ignores crowd and bench) |
| `football-ball-detection.pt` | The ball |
| `football-pitch-detection.pt` | 32 pitch landmarks used to calibrate the camera each frame |

The status bar shows which models are in use once a video is loaded. If a file is missing the app still runs, with weaker results:

- No player model: falls back to the generic `yolov8n.pt` (downloaded automatically), which marks every person in the frame and cannot tell referees or goalkeepers apart.
- No ball model: the ball is taken from the player model, which finds it far less often.
- No pitch model: radar, heatmap and formation stay empty (no default stretch onto the pitch).

To use different weights, update `MODEL_PATH`, `BALL_MODEL_PATH` and `PITCH_MODEL_PATH` in `src/config.py`. Check the licence terms of the weights (Ultralytics models are AGPL-3.0) before redistributing them.

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
| **Pitch Lines** | Debug: project calibrated pitch markings back onto the video |

### Using the Heatmap

- The **"Select Player"** dropdown lists durable tracks as `Team A #12`, `GK #3 (Team B)`, etc.
- Brief false IDs (seen for fewer than ~15 frames) are kept out of the list
- Select any entry to view that player's real-time positional heatmap
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

1. **YOLO Detection** — A soccer-trained model detects players, goalkeepers and referees; each person carries a `role`
2. **ByteTrack Tracking** — Assigns persistent IDs to players across frames so the same player keeps the same number
3. **Ball Tracking** — A dedicated ball model searches a small full-resolution window around the ball's last position (the whole frame when the ball is lost); detections that jump implausibly far are ignored and short gaps are filled with linear interpolation
4. **Team Classification** — Outfield jersey colours (tight crop, grass filtered) are clustered once into Team A / B; each track's team and role are majority-voted so labels stay stable. Referees come from the detector; goalkeepers take the nearest team's side
5. **Pitch Calibration** — A pose model finds pitch landmarks; with ≥ 4 confident points a homography maps pixels onto the 105×68 m pitch. Close-ups and cuts leave positions empty ("No pitch view"); hard cuts also reset the tracker
6. **Formation Detection** — Clusters outfield players into lines, oriented from each team's own goal. Players who have just left the frame still count for a few seconds. The label is blank unless enough of the team is accounted for
7. **Heatmap Accumulation** — On calibrated frames only. Colour is the share of the **whole clip** spent in that area (not each player's own peak), so it is only a full picture when the clip ends. The dropdown also has a team-wide heatmap

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
│   └── weights/                   ← Model weights (downloaded in step 4, not in git)
└── src/
    ├── main.py                    ← Entry point
    ├── config.py                  ← All tunable constants
    ├── core/
    │   ├── packet.py              ← PlayerState, BallState, FramePacket dataclasses
    │   ├── video_io.py            ← Video reader generator + MP4 exporter
    │   └── pipeline_worker.py     ← QThread orchestrating the full pipeline
    ├── vision/
│   ├── detector.py            ← YOLO inference, roles, ball search + ByteTrack tracking
│   ├── ball_tracker.py        ← Ball selection, jump rejection, gap interpolation
│   ├── pitch_keypoints.py     ← Pitch landmark detection for per-frame homography
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
| `MODEL_PATH` | `models/weights/football-player-detection.pt` | Player/goalkeeper/referee weights (relative paths are taken from the project root) |
| `BALL_MODEL_PATH` | `models/weights/football-ball-detection.pt` | Ball weights; `None` uses the main model for the ball |
| `FALLBACK_MODEL_PATH` | `models/weights/yolov8n.pt` | Generic model used when `MODEL_PATH` is missing |
| `DETECTION_IMGSZ` | `640` | Image size the player model runs on; larger is slower |
| `BALL_SEARCH_WINDOW` | `(320, 192)` | Size in pixels of the window searched around the ball's last position |
| `BALL_FULL_SEARCH_INTERVAL` | `5` | While the ball is lost, search the whole frame every N frames |
| `BALL_MAX_JUMP_FRACTION` | `0.05` | Largest plausible ball movement per frame, as a fraction of the frame diagonal |
| `PITCH_MODEL_PATH` | `models/weights/football-pitch-detection.pt` | Pitch landmark weights; `None` disables calibration |
| `PITCH_CONFIDENCE_THRESHOLD` | `0.5` | Min per-keypoint confidence to keep a landmark |
| `PITCH_MIN_KEYPOINTS` | `4` | Landmarks needed before a homography is estimated |
| `SCENE_CUT_HIST_CORR` | `0.6` | Histogram correlation below which a hard cut is assumed |
| `POSITION_SMOOTHING_ALPHA` | `0.4` | EMA weight on new pitch positions (1.0 = no smoothing) |
| `TEAM_FIT_MIN_PLAYERS` | `8` | Outfield players needed in one frame before jersey colours are fitted |
| `TEAM_VOTE_SETTLE` / `ROLE_VOTE_SETTLE` | `5` | Observations before a track's team / role is locked |
| `HEATMAP_MIN_TRACK_FRAMES` | `15` | Sightings before a track appears in the heatmap dropdown |
| `CONFIDENCE_THRESHOLD` | `0.3` | Min detection confidence |
| `IOU_THRESHOLD` | `0.5` | NMS IoU threshold |
| `DEVICE` | `None` | Inference device (`"mps"`, `"cuda"`, `"cpu"`); `None` picks the fastest available |
| `HEATMAP_REDRAW_INTERVAL_S` | `0.5` | Minimum seconds between live heatmap redraws |
| `HEATMAP_SIGMA` | `2.5` | Gaussian blur for heatmaps |
| `PITCH_LENGTH` | `105.0` | Pitch length in metres |
| `PITCH_WIDTH` | `68.0` | Pitch width in metres |
| `PLAYBACK_FPS` | `24` | Target speed for showing the video |
| `ANALYSIS_STRIDE` | `12` | Run the heavy models every Nth frame (`1` = every frame, much slower) |
| `FORMATION_MIN_OUTFIELD` | `7` | Outfield players needed before a formation is named |
| `OFFSCREEN_HOLD_S` | `2.5` | Seconds a player who left the frame still counts in the formation |
| `HEATMAP_VMAX_FRACTION` | `0.12` | Clip share that maps to the hottest heatmap colour |
| `MAX_FPS` | `30` | Playback FPS cap |

---

## Tips for Best Results

- **Use broadcast footage** with a wide-angle view of the pitch for best detection coverage
- **15–30 second clips** are ideal for initial testing
- **Wide broadcast shots** are required for pitch calibration; close-ups and replays show "No pitch view" and freeze the radar/heatmap until a wide shot returns
- Turn on **Pitch Lines** in the status bar to check that projected markings sit on the real pitch lines while the camera pans
- The team classifier needs **at least 8 outfield players** in one frame to fit — it fits once, then locks each track's team after a few colour votes
- Formation detection needs **at least 3 players per team** with valid pitch positions

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Make sure you activated the venv: `source venv/bin/activate` |
| Window doesn't appear | Ensure PySide6 is installed: `pip install PySide6` |
| No detections showing | Check the status bar for the models in use; download the weights with the commands in step 4 |
| Crowd or bench staff get markers | The status bar says "generic fallback": the soccer player model is missing, see step 4 |
| Radar stays empty / "No pitch view" | Download `football-pitch-detection.pt` (step 4); close-ups also pause calibration until a wide shot returns |
| Slow FPS | Set `BALL_MODEL_PATH = None` in `config.py` to skip the separate ball search (faster, but the ball is found far less often) |
| Heatmap is blank | Select a player from the dropdown — heatmap only builds on calibrated (wide) frames |

---

## Tech Stack

| Library | Role |
|---------|------|
| **Ultralytics** | YOLOv8 object detection |
| **Supervision** | Detection data structures |
| **Trackers** | ByteTrack multi-object tracking |
| **OpenCV** | Frame I/O, drawing, homography |
| **Scikit-Learn** | K-Means for jerseys & formations |
| **SciPy** | Gaussian heatmap smoothing |
| **PySide6** | Desktop UI framework (Qt) |
| **Matplotlib** | Heatmap rendering |
| **mplsoccer** | Pitch visualization utilities |
