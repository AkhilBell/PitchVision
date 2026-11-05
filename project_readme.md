# PitchVision: Real-Time Soccer Video Tracking & Tactical UI

PitchVision is an end-to-end computer vision and tactical analytics desktop application written in Python. It ingests broadcast or tactical-cam soccer video and renders a synchronized, side-by-side live dashboard featuring broadcast-style player overlays, a 2D live tactical radar, dynamic team formation detection, and real-time individual player heatmaps.

---

## Key Features

* **Broadcast-Style Video Overlays:** Draws 3D perspective ellipses beneath players' feet, color-coded by team, along with persistent player tracker IDs and a floating ball marker.
* **Automated Team Assignment:** Uses unsupervised clustering ($K$-Means in Lab/HSV color space) on jersey crops to automatically separate Team A, Team B, and referees without manual labeling.
* **2D Pitch Homography Transformation:** Maps camera pixel coordinates $(u, v)$ to standard FIFA pitch coordinates $(X, Y)$ in meters ($105 \times 68\text{ m}$) using perspective transformation matrices.
* **Dynamic Formation Detection:** Continuously clusters outfield players along the pitch length to classify live tactical shapes (e.g., `4-4-2`, `4-3-3`, `3-5-2`) and renders connected formation lines on the live radar.
* **Live Individual Heatmaps:** Click or select any tracked player ID in the UI to watch their spatial density heatmap build frame-by-frame in real time.
* **Side-by-Side Interactive UI:** Built on a multi-threaded Producer-Consumer architecture so heavy YOLO inference never freezes the video playback or interactive controls.

---

## User Interface Layout

```text
+-----------------------------------------------------------------------------------+
|  PitchVision Tactical Studio               [Load Video] [Play/Pause] [Export MP4] |
+-------------------------------------------------+---------------------------------+
|                                                 |  LIVE 2D TACTICAL RADAR         |
|                                                 |  +---------------------------+  |
|                                                 |  |  o---o---o     x---x---x  |  |
|                                                 |  |   \ / \ /       \ / \ /   |  |
|          MAIN VIDEO FEED (WITH OVERLAYS)        |  |    o---o    *    x---x    |  |
|                                                 |  |   / \ / \       / \ / \   |  |
|     - Team-colored ellipses under feet          |  |  o---o---o     x---x---x  |  |
|     - Player ID badges (#7, #10, etc.)          |  +---------------------------+  |
|     - Floating triangle over the ball           |  Team A: 4-3-3  | Team B: 4-4-2 |
|     - Optional tactical lines on grass          +---------------------------------+
|                                                 |  LIVE PLAYER HEATMAP            |
|                                                 |  Select Player: [ Player #7 v ] |
|                                                 |  +---------------------------+  |
|                                                 |  |      ..::####::..         |  |
|                                                 |  |     .::########::.        |  |
|                                                 |  |      ..::####::..         |  |
|                                                 |  +---------------------------+  |
+-------------------------------------------------+---------------------------------+
|  [x] Show Foot Markers   [x] Show Formation Lines   [x] Ball Trail   FPS: 29.8    |
+-----------------------------------------------------------------------------------+
```

---

## Repository Structure

```text
pitchvision/
├── README.md
├── requirements.txt
├── docs/
│   ├── ARCHITECTURE.md          # Deep-dive system design, math, and data flow
│   └── ROADMAP.md               # Phased implementation checklist & Python skills
├── data/
│   ├── samples/                 # Sample 15-30s match clips for testing
│   └── exports/                 # Saved annotated videos and heatmap PNGs
├── models/
│   └── weights/                 # YOLO weights (e.g., yolov8x.pt or fine-tuned pitch model)
└── src/
    ├── __init__.py
    ├── main.py                  # Entry point launching the UI and worker threads
    ├── config.py                # Pitch dimensions, color palettes, and model hyperparameters
    ├── core/
    │   ├── packet.py            # Dataclasses for FramePacket, PlayerState, PitchState
    │   ├── video_io.py          # Generator-based video reader and MP4 exporter
    │   └── pipeline_worker.py   # Background QThread running CV pipeline into a Queue
    ├── vision/
    │   ├── detector.py          # YOLO inference wrapper + ByteTrack ID management
    │   ├── ball_tracker.py      # Ball detection filtering + Kalman/linear interpolation
    │   ├── team_classifier.py   # Jersey crop extraction + K-Means team assignment
    │   └── homography.py        # Pixel (u,v) <-> Pitch (X,Y) perspective transformation
    ├── analytics/
    │   ├── heatmap_engine.py    # Fast 2D NumPy accumulator grid + Gaussian smoothing
    │   └── formation_engine.py  # Outfield spatial clustering + tactical line generation
    └── ui/
        ├── main_window.py       # PySide6 main application window and layout
        ├── video_canvas.py      # OpenCV broadcast overlay renderer (ellipses, badges)
        ├── radar_widget.py      # Live 2D pitch minimap and formation graph renderer
        └── heatmap_widget.py    # Real-time player heatmap canvas and selector
```

---

## Tech Stack & Dependencies

Create a `requirements.txt` file in the root of your repository with the following packages:

```txt
opencv-python>=4.9.0
ultralytics>=8.2.0
supervision>=0.20.0
numpy>=1.26.0
scipy>=1.13.0
scikit-learn>=1.4.0
PySide6>=6.7.0
matplotlib>=3.8.0
mplsoccer>=1.2.4
```

### Why These Libraries?
* **Ultralytics & Supervision:** Handles object detection and `ByteTrack` multi-object persistence with minimal boilerplate.
* **OpenCV & SciPy:** Powers frame drawing, perspective homography matrices, and ultra-fast 2D Gaussian filtering (`scipy.ndimage.gaussian_filter`) for live heatmaps.
* **Scikit-Learn:** Runs $K$-Means clustering for both jersey color identification and formation line discovery.
* **PySide6 (Qt for Python):** Enables a responsive 30 FPS desktop UI where video playback, live radar, and dropdown selectors update simultaneously without browser lag.

---

## Quickstart (Once Implemented)

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/pitchvision.git
cd pitchvision

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
python -m src.main
```