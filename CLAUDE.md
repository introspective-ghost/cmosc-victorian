# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Victorian Photo Booth for the Children's Museum of Sonoma County. A two-Raspberry Pi system where a "leader" Pi runs the main greenscreen capture pipeline and a "follower" Pi displays captured photos on an external screen.

## Hardware Architecture

**Leader Pi** (`src/main.py` + `src/buttonHandler.py` + `src/piFileTransfer.py`):
- Drives a dual-monitor pygame window (two 1920x1080 displays side-by-side, total 3840x1080)
- Captures live camera feed via `picamera2`, applies real-time greenscreen compositing with OpenCV
- Physical button on GPIO pin 17 triggers photo capture
- On capture: saves `pic0.jpg` (leader's own display), and `rsync`s `pic1.jpg`/`pic2.jpg` to the follower Pi over LAN

**Follower Pi** (`src/followerPiPicPoller.py`):
- Also drives a dual-monitor pygame window
- Polls `~/pics/` for new/updated JPEG files (checks mtime + JPEG end marker `\xff\xd9`)
- Displays `pic1.jpg` on monitor0 and `pic2.jpg` on monitor1

## Key Configuration (in `src/main.py`)

```python
CANVAS_WIDTH, CANVAS_HEIGHT = 1920, 1080   # per-monitor resolution
FRAME_WIDTH, FRAME_HEIGHT   = 1350, 1080   # cropped camera region
BUTTON_PIN = 17                             # GPIO BCM pin
cropX = 212                                 # horizontal crop offset into camera frame
# HSV greenscreen thresholds
hLow, sLow, vLow   = 35, 40, 40
hHigh, sHigh, vHigh = 95, 255, 255
```

## File Transfer

`piFileTransfer.py` uses `rsync` over SSH with key-based auth (`/home/cmosc/.ssh/id_ed25519`). The follower Pi is reached at `victorian1.local` (static LAN hostname), SSH user `cmosc`.

## Running the Code

Both scripts must be run on their respective Raspberry Pis:

```bash
# Leader Pi
cd ~/cmosc-victorian/src
python main.py

# Follower Pi
cd ~/cmosc-victorian/src
python followerPiPicPoller.py
```

The leader Pi expects the repo at `~/cmosc-victorian/` and background images at `~/cmosc-victorian/backgroundImages/`.

Logs are written to `~/cmosc-victorian/logs/` (gitignored). Captured photos go to `~/cmosc-victorian/pics/` (gitignored).

## Naming Conventions

- **Source files**: `upperCamelCase` for functions, classes, and variables
- **Test files** in `testFiles/`: informal/experimental scripts, not production code

## Dependencies (Raspberry Pi only)

`picamera2`, `libcamera`, `RPi.GPIO`, `opencv-python` (`cv2`), `numpy`, `pygame`, `screeninfo` (follower only)

## Watchdog Pattern

`main.py` wraps `runPipeline()` in a watchdog loop — `RuntimeError` triggers a restart after `WATCHDOG_DELAY` seconds; other exceptions exit cleanly. Signal handlers for `SIGINT`/`SIGTERM` ensure camera, GPIO, and SSH resources are released.
