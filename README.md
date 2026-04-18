# Squat Trainer — AI Coach

Real-time squat form analysis using MediaPipe pose detection and Groq-powered AI coaching cues, served through a web UI.

## How it works

A webcam feed is processed frame-by-frame by MediaPipe's pose landmarker model. Key joint angles (knee, hip, torso) are extracted and used to:

- Count reps automatically via a state machine (up → down → up)
- Detect form issues: squat depth, knee cave (valgus), back rounding
- Every 3 seconds (or on each new rep), send the angle snapshot to **Llama 3.1 8B on Groq** for a short, direct coaching cue ("Heels flat on the floor", "Push your knees out")

The processed frame streams to the browser as MJPEG. Live stats and AI feedback stream via WebSocket.

## Stack

| Layer | Tech |
|---|---|
| Pose detection | MediaPipe Pose Landmarker (heavy float16) |
| Backend | FastAPI + Uvicorn |
| AI coaching | Groq API — `llama-3.1-8b-instant` |
| Video stream | MJPEG over HTTP |
| Live data | WebSocket (10 Hz) |
| Frontend | Vanilla HTML/CSS/JS |

## Setup

```bash
# 1. Install dependencies into the project venv
.venv/bin/python -m pip install fastapi "uvicorn[standard]" groq

# 2. Set your Groq API key
export GROQ_API_KEY=your_key_here

# 3. Run
.venv/bin/python server.py
```

Open `http://localhost:8000` in a browser. Stand in front of the webcam with your full body visible.

The pose model (`pose_landmarker.task`) is downloaded automatically on first run if missing.

## Controls

| Action | How |
|---|---|
| Reset rep counter | Click **Reset** in the header, or `POST /reset` |
| Quit original desktop app | `q` key (only in `pose_webcam.py`) |

## Files

```
├── server.py            # FastAPI server — pose processing + Groq loop + MJPEG + WS
├── pose_webcam.py       # Original standalone OpenCV desktop app
├── static/
│   └── index.html       # Web UI
└── pose_landmarker.task # MediaPipe model (auto-downloaded)
```

## Form checks

| Check | Method |
|---|---|
| **Squat depth** | Knee angle < 95° = parallel reached |
| **Knee cave** | Knee closer to image centre than ankle (mirror-agnostic) |
| **Back rounding** | Torso lean > 50° from vertical |
