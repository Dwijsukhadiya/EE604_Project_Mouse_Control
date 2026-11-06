# Virtual Mouse Using Hand Gesture Recognition

A webcam-based virtual mouse that uses computer vision and hand landmarks to control the desktop pointer. The project is implemented in Python with OpenCV and MediaPipe Hands, with PyAutoGUI translating recognized gestures into mouse and keyboard actions.

This project was developed for the **EE604 Computer Vision course**. It is intended as a course project and experimentation tool; it does not require a separate hardware controller beyond a webcam.

## Features

- Tracks one hand and uses its landmarks to recognize gestures.
- Moves the mouse pointer from the hand's position in the camera frame.
- Supports left click, click-and-drag, right click, scrolling, and keyboard-based zoom.
- Captures timestamped screenshots using a gesture.
- Brightens dark camera frames with adaptive gamma correction before hand detection.
- Smooths pointer motion and stabilizes gesture recognition to reduce jitter and accidental actions.
- Displays a live camera preview with the recognized gesture and estimated FPS.
- Logs recognized actions and their measured latency to a CSV file.

## Requirements

- Python 3.9 or newer (use a Python version supported by the installed MediaPipe release).
- A webcam connected to the computer.
- A desktop session where the application is permitted to control the mouse and keyboard.

Python dependencies:

- OpenCV (`opencv-python`)
- MediaPipe
- NumPy
- PyAutoGUI

## Installation

Open PowerShell in the project directory and create a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install opencv-python mediapipe numpy pyautogui
```

If PowerShell prevents activating the virtual environment, you can invoke its Python directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install opencv-python mediapipe numpy pyautogui
```

## Run

From the project directory, start the application with:

```powershell
python .\EE604_courseproject_code_mousecontrol.py
```

The application opens the default webcam (`CAM_INDEX = 0`) and displays a window titled **Virtual Mouse**. Position your hand in view of the camera and use the gestures below. Press **Q** or **Esc** while the preview window is active to exit.

On startup, the program creates a `screenshots` directory if needed and opens `gesture_log.csv` in append mode. These files are created in the current working directory.

## Gesture controls

| Gesture | Action |
|---|---|
| Index finger up, middle finger down | Move the cursor |
| Pinch thumb and index finger briefly | Left click |
| Hold thumb-index pinch for more than 0.45 seconds | Start dragging; release the pinch to stop |
| Pinch thumb and middle finger | Right click |
| Pinch thumb and ring finger | Save a screenshot in `screenshots` |
| Index and middle fingers up, ring and pinky down | Scroll; move the hand up or down |
| Index, middle, and ring fingers up, pinky down | Zoom in or out; move the hand up or down |
| Fist or another unrecognized pose | Idle; no action |

Click, right-click, and screenshot actions are subject to a short cooldown to help prevent repeated activations. Gestures must also remain stable for several frames before they are accepted.

### Pointer mapping

The camera image is mirrored to behave like a mirror. The central 70% of the image is the active region by default; pointer positions are mapped from that region to the screen. The index MCP landmark (the base knuckle of the index finger) is used as the pointer anchor so that pinching is less likely to shift the cursor.

## Configuration

The constants near the top of `EE604_courseproject_code_mousecontrol.py` control the webcam, pointer behavior, gesture thresholds, and output paths. Common settings include:

| Setting | Default | Purpose |
|---|---:|---|
| `CAM_INDEX` | `0` | Camera device index |
| `CAM_W`, `CAM_H` | `640`, `480` | Requested camera frame size |
| `FRAME_MARGIN` | `0.15` | Inactive margin at each edge of the camera frame |
| `SMOOTHING` | `0.30` | Pointer movement smoothing; lower values are smoother but respond more slowly |
| `PINCH_ON`, `PINCH_OFF` | `0.30`, `0.42` | Normalized pinch activation and release thresholds |
| `STABLE_FRAMES` | `3` | Consecutive frames required to accept a gesture |
| `ACTION_COOLDOWN` | `0.40` seconds | Minimum interval between discrete actions |
| `DRAG_HOLD` | `0.45` seconds | Hold duration before a left pinch becomes a drag |
| `SCREENSHOT_DIR` | `screenshots` | Directory for gesture-captured screenshots |
| `LOG_FILE` | `gesture_log.csv` | CSV file to which actions are appended |

Pinch thresholds use a distance normalized by hand size, which helps make recognition less dependent on how close the hand is to the camera. The separate on/off thresholds provide hysteresis to reduce flickering at the threshold.

## Output files

### `gesture_log.csv`

Each recorded action is appended as one CSV row with these columns:

| Column | Description |
|---|---|
| `timestamp` | Local time when the action was recorded |
| `event` | Action label, such as `left_click`, `drag_start`, `scroll_up`, or `zoom_in` |
| `latency_ms` | Time in milliseconds from the first frame of the accepted gesture to the recorded action |
| `fps` | Smoothed estimate of the camera-processing frame rate at that time |

The log is for course evaluation and basic observation; it is not a comprehensive performance benchmark. Existing rows are preserved when the application is run again.

### `screenshots/`

The thumb-ring pinch saves a desktop screenshot as a PNG file with a timestamped name, for example `shot_20260101_120000.png`.

## Implementation overview

1. OpenCV reads frames from the configured webcam and mirrors them horizontally.
2. If the frame's mean grayscale brightness is below the configured threshold, an adaptive gamma lookup table brightens it.
3. MediaPipe Hands detects one hand and returns its 21 landmarks.
4. The application measures fingertip distances relative to hand size, checks which fingers are extended, and classifies the frame into one gesture.
5. Pinch detection uses separate activation and release thresholds; a debouncer requires a gesture to persist for multiple frames.
6. Recognized gestures are translated into PyAutoGUI pointer, mouse-button, scroll, screenshot, or keyboard operations.
7. The preview overlays the active region, current gesture, and FPS, while action events are appended to the CSV log.

## Troubleshooting and use notes

- **Camera does not open or the preview immediately exits:** Confirm the webcam is connected and not in use by another application. If necessary, change `CAM_INDEX` to another device index.
- **Hand detection is unreliable:** Use even, adequate lighting, keep the hand fully visible, and avoid busy backgrounds. Dark frames are brightened automatically, but severe underexposure can still reduce landmark quality.
- **Pointer motion is too sensitive or laggy:** Adjust `SMOOTHING` and `FRAME_MARGIN`. A larger smoothing weight responds faster but may look less stable.
- **Gestures trigger accidentally or do not register:** Adjust the pinch thresholds and `STABLE_FRAMES` carefully. The pinch ratios are relative to the detected hand size.
- **Zoom does not affect the current application:** Zoom is sent as `Ctrl` + `=` or `Ctrl` + `-` on Windows/Linux, and `Command` + the corresponding key on macOS. The active application must support those shortcuts.
- **Operating-system permission prompts appear:** Grant camera access and any required desktop-control or accessibility permissions to Python or the terminal running it.

The program controls the real desktop. Keep the camera preview visible, use deliberate gestures, and exit with **Q** or **Esc** when finished. PyAutoGUI's fail-safe is enabled as an additional safeguard.

## Limitations

- The application currently processes at most one hand and uses simple landmark geometry and fixed thresholds rather than a trained custom gesture classifier.
- Hand pose, camera placement, lighting, and individual differences can affect recognition.
- The default camera index, frame size, and gestures may need adjustment for a particular computer or use environment.
- Screenshot and gesture-log output paths are relative to the working directory from which the script is launched.
