# Hand Jarvis

Control **Windows** with your webcam and hand gestures — a lightweight, local “Iron Man” style desktop controller. No cloud, no training required.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Platform](https://img.shields.io/badge/platform-Windows-0078D6)
![License](https://img.shields.io/badge/license-MIT-green)

## Features

- Webcam + [MediaPipe](https://developers.google.com/mediapipe) hand tracking
- **Relative** mouse mode with One Euro filtering (smooth, low jitter)
- **Clutch** — tuck thumb to freeze the cursor and reposition your hand
- **Dwell click** — hold still to click without pinching
- Gestures for Alt+Tab, volume mute, and more
- Interactive **calibration** for your hand and screen

## Requirements

- **Windows 10/11**
- **Python 3.9–3.12** (64-bit recommended; 3.11 works well)
- Webcam
- Good lighting helps tracking

## Quick start

```powershell
git clone https://github.com/muhib360/hand-jarvis.git
cd hand-jarvis

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python calibrate.py
python main.py
```

On first run, the hand model (~3 MB) downloads into `models/`.

## Gestures

| Gesture | Action |
|--------|--------|
| **Open palm** (hold **5 s**) | Toggle control **ON** / **OFF** |
| **Point** (index extended) | Move mouse |
| **Thumb tucked** + index out | **Clutch** — cursor frozen |
| **Pinky out** while pointing | **Precision** — slower cursor |
| **Index + middle** (hold briefly) | Jump cursor to aim position |
| **Pinch** | Left click |
| **Hold cursor still ~0.45 s** | Dwell click |
| **Fist** (index not extended) | Volume mute |
| **Swipe left / right** | Alt+Shift+Tab / Alt+Tab |

Control starts **OFF**. Hold an open palm until the on-screen bar completes.

## Keys (preview window)

| Key | Action |
|-----|--------|
| **Q** | Quit |
| **C** | Run calibration |

Move the mouse to a **screen corner** for pyautogui failsafe abort.

## Configuration

After calibration, settings are saved to `config.json` (gitignored — personal to your hand).

Copy [`config.example.json`](config.example.json) for reference or reset defaults. Tune the `pointer` section for gain, dead zone, dwell click, etc.

```powershell
python main.py --absolute
```

Use `--absolute` for legacy full-screen finger mapping (relative mode is default).

## Project layout

```
hand-jarvis/
├── main.py           # App entry
├── calibrate.py      # One-time setup wizard
├── pointer.py        # Mouse control (filter, clutch, dwell)
├── gestures.py       # Gesture detection
├── gesture_state.py  # Hysteresis / dropout
├── tracking.py       # MediaPipe wrapper + smoothing
├── config_store.py   # Load/save config
└── requirements.txt
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Webcam won't open | Close other apps using the camera; set `CAMERA_INDEX` in `main.py` |
| Mouse feels jittery | Lower `relative_gain` or raise `dead_zone_px` in `config.json` |
| Bottom of frame fails | Re-run `calibrate.py`; only index needs to be visible when pointing down |
| MediaPipe install fails | Use 64-bit Python 3.11; `python -m pip install -U pip` |
| Console `clearcut` errors | Harmless MediaPipe telemetry — safe to ignore |

## Limitations

This is a **webcam + 2D landmarks** project. It works well for shortcuts and casual control, but holding a hand in the air is more tiring than a physical mouse. Use **clutch** and **dwell click** for longer sessions.

## License

MIT — see [LICENSE](LICENSE).

## Contributing

Issues and PRs welcome. Please run calibration on your machine before tuning defaults.
