# Eye-Tracker

**A Modular, Real-Time Eye and Gaze Tracking Toolkit**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8.svg)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-00A98F.svg)](https://mediapipe.dev/)

---

## Overview

Eye-Tracker is a collection of Python modules for real-time pupil detection,
3D gaze estimation, head pose tracking, and screen-space cursor control. The
project is designed to be modular: each component targets a specific hardware
setup and tracking approach, from simple 2D ellipse-fitting on a single
camera feed to full binocular 3D gaze estimation with virtual monitor plane
projection using only a standard webcam and MediaPipe.

The toolkit spans five categories of functionality:

- **2D pupil detection** via adaptive multi-threshold binarization and
  ellipse fitting with goodness-of-fit scoring
- **3D gaze estimation** using ray-based eye models, orthogonal ray
  intersection, and ray-sphere projection
- **Webcam-based 3D tracking** combining MediaPipe face mesh, PCA-based
  head orientation, per-eye sphere models, and gaze-to-screen mapping
- **Head pose and hand-tracking cursor control** with blink-to-click input
- **OpenGL and Unity integration** for real-time 3D eyeball visualization
  and VR/AR gaze vector consumption

---

## Project Structure

```
Eye-Tracker/
|-- PupilDetector.py              # Full-featured 2D pupil detector (video/webcam)
|-- PupilDetectorLite.py          # Lightweight single-threshold pupil detector
|-- PupilDetectorRaspberryPi.py   # Optimized pupil detector for Raspberry Pi
|
|-- 3DTracker/
|   |-- EyeTracker3D.py           # 3D eye tracker with ray intersection model
|   |-- gl_sphere.py                     # OpenGL wireframe eyeball visualization (PyQt5)
|   |-- GazeFollower.cs                  # Unity C# script for gaze vector consumption
|   |-- gaze_vector.txt                  # Shared gaze vector output (origin + direction)
|
|-- FrontCameraTracker/
|   |-- EyeTrackerFrontCamera.py  # Eye-to-scene camera gaze calibration
|   |-- gaze_vector.txt                 # Gaze vector output for this module
|
|-- HeadTracker/
|   |-- MonitorTracking.py               # Head/finger cursor control with blink-to-click
|   |-- CursorCircle.py                  # Transparent cursor overlay widget (PyQt5)
|   |-- face_landmarker.task             # MediaPipe face landmark model
|   |-- tracking_log.csv                 # Tracking session log
|
|-- Webcam3DTracker/
|   |-- MonitorTracking.py               # 3D gaze to screen mapping with debug orbit view
|   |-- face_landmarker.task             # MediaPipe face landmark model
|   |-- screen_position.txt              # Real-time screen position output
|   |-- requirements.txt                 # Python dependencies for this module
|
|-- LICENSE                              # MIT
|-- README.md
```

---

## Modules

### 2D Pupil Detector (`PupilDetector.py`)

The core pupil detection engine. Processes each frame through a multi-stage
pipeline:

1. **Darkest-area search** -- sparse sampling across the image to locate the
   approximate pupil region
2. **Multi-threshold binarization** -- three threshold levels (strict,
   medium, relaxed) applied in parallel to handle varying illumination
3. **Contour filtering** -- area and aspect-ratio constraints reject
   non-pupil blobs
4. **Ellipse fitting** -- `cv2.fitEllipse` with a composite goodness-of-fit
   metric (fill ratio, contour pixel overlap, skewness)
5. **Contour optimization** -- angle-based filtering removes outlier contour
   points that deviate from the expected curvature

Accepts both video file and live webcam input. A debug mode (toggle with `D`)
displays all intermediate processing stages.

### Pupil Detector Lite (`PupilDetectorLite.py`)

A streamlined variant that uses a single threshold level. Suited for
scenarios with consistent illumination or limited compute budget.

### Raspberry Pi Detector (`PupilDetectorRaspberryPi.py`)

Optimized for embedded deployment on Raspberry Pi:

- Larger sparse-sampling step sizes to reduce per-frame computation
- Built-in FPS counter overlay
- Downscaled display output (320 x 240)
- Direct USB camera input with no GUI file picker

### 3D Eye Tracker (`3DTracker/`)

Extends 2D detection into 3D gaze estimation:

| Component | Description |
|-----------|-------------|
| `EyeTracker3D.py` | Computes orthogonal rays from fitted pupil ellipses, finds their average intersection to estimate the 3D eye center, and projects the gaze direction via ray-sphere intersection |
| `gl_sphere.py` | PyQt5 + PyOpenGL wireframe eyeball with a rotating green iris ring; renders to an OpenCV-compatible image buffer for overlay |
| `GazeFollower.cs` | Unity MonoBehaviour that reads `gaze_vector.txt` each frame and drives a GameObject's position and rotation |

The gaze vector is written to `gaze_vector.txt` as six comma-separated
floats: three for the 3D sphere center (origin) and three for the normalized
gaze direction.

### Front Camera Tracker (`FrontCameraTracker/`)

Dual-camera gaze projection system. Tracks the pupil with a dedicated eye
camera and projects the computed 3D gaze vector onto a separate front-facing
scene camera. Supports runtime calibration (`C` key) to align the gaze
coordinate system with the external camera, and sphere center locking (`L`
key) to freeze the eye model position.

### Head Tracker (`HeadTracker/`)

A cursor control system combining head pose and hand tracking:

- **MediaPipe Face Mesh** computes real-time head orientation (yaw and pitch)
  from 468 facial landmarks
- **Hand tracking** uses the index finger tip (landmark 8) as the primary
  cursor driver, mapped to full screen coordinates
- **Head precision offset** applies calibrated head rotation as a fine-grained
  adjustment to finger-based positioning
- **Blink-to-click** detects deliberate blinks via Eye Aspect Ratio (EAR)
  thresholding with configurable hold duration
- **Cursor overlay** (`CursorCircle.py`) renders a transparent green ring at
  the cursor position using PyQt5

### Webcam 3D Tracker (`Webcam3DTracker/`)

The most advanced module. Provides full 3D gaze tracking from a single
webcam:

- PCA-based head orientation from facial landmark point clouds with
  eigenvector stabilization to prevent frame-to-frame sign flips
- Independent left and right eye sphere models locked to the head coordinate
  frame via nose-region PCA
- Combined binocular gaze computed as the average of per-eye gaze directions
- Virtual 3D monitor plane placed in world space for gaze-to-screen
  coordinate mapping with real-world scale estimation
- Orbit debug camera with full keyboard controls for 3D scene inspection
- Direct mouse cursor positioning from gaze (toggle with `F7`)
- Screen position output to `screen_position.txt` for external consumption

---

## Keyboard Controls

| Key | Action | Module(s) |
|-----|--------|-----------|
| `Q` | Quit | All |
| `D` | Toggle debug visualization | Pupil Detector, 3D Tracker |
| `Space` | Pause / Resume playback | Pupil Detector |
| `C` | Calibrate center / head orientation | HeadTracker, Webcam3DTracker, FrontCameraTracker |
| `L` | Lock sphere center | 3D Tracker, FrontCameraTracker |
| `F7` | Toggle mouse control | HeadTracker, Webcam3DTracker |
| `I` / `K` | Orbit camera pitch up / down | Webcam3DTracker |
| `J` / `L` | Orbit camera yaw left / right | Webcam3DTracker |
| `[` / `]` | Zoom out / in (debug camera) | Webcam3DTracker |
| `R` | Reset debug camera view | Webcam3DTracker |
| `X` | Add gaze marker on monitor plane | Webcam3DTracker |

---

## Getting Started

### Prerequisites

- Python 3.10+
- A webcam or USB eye camera
- Windows or Linux (Raspberry Pi supported for the embedded detector)
- (Optional) Unity for `GazeFollower.cs` integration

### 1. Clone and install dependencies

```bash
git clone https://github.com/mahaddev-x/Eye-Tracker.git
cd Eye-Tracker
```

For the full Webcam3DTracker module:

```bash
pip install opencv-python numpy mediapipe scipy pyautogui keyboard
```

For the basic pupil detectors only:

```bash
pip install opencv-python numpy matplotlib
```

For modules with OpenGL visualization (3DTracker):

```bash
pip install opencv-python numpy PyQt5 PyOpenGL PyOpenGL_accelerate
```

### 2. Run the basic pupil detector

```bash
python PupilDetector.py
```

A file dialog will prompt for a video file, or you can modify the script to
use a live webcam feed (set `input_method` to `2`).

### 3. Run the webcam 3D gaze tracker

```bash
cd Webcam3DTracker
python MonitorTracking.py
```

Press `C` to calibrate head center, then `F7` to enable mouse control.

### 4. Run the head and hand cursor controller

```bash
cd HeadTracker
python MonitorTracking.py
```

Press `C` to calibrate, then point with your index finger to move the
cursor. Deliberate blinks trigger a click.

---

## Output Formats

### `gaze_vector.txt` (3DTracker, FrontCameraTracker)

```
origin_x,origin_y,origin_z,direction_x,direction_y,direction_z
```

Six comma-separated floats per line: 3D sphere center (origin) and
normalized gaze direction vector. Overwritten each frame.

### `screen_position.txt` (Webcam3DTracker)

```
screen_x,screen_y
```

Two comma-separated integers: estimated gaze position on the monitor in
pixel coordinates. Overwritten each frame.

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| OpenCV | Image processing, video I/O, contour analysis, ellipse fitting |
| NumPy | Numerical computation, linear algebra, array operations |
| MediaPipe | Face mesh (468 landmarks) and hand landmark detection |
| SciPy | Rotation representations and Euler angle decomposition |
| PyAutoGUI | Programmatic mouse and click control |
| PyQt5 | OpenGL widget hosting and transparent cursor overlay |
| PyOpenGL | 3D wireframe eyeball rendering |
| Matplotlib | Score distribution visualization |
| keyboard | Global hotkey detection for runtime controls |

---

## Contributing

Contributions are welcome. To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

```
MIT License
Copyright (c) 2026 Mahad Asif
```

---

## Author

**Mahad Asif** -- [@mahaddev-x](https://github.com/mahaddev-x)
