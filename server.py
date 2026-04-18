#!/usr/bin/env python3
"""FastAPI server for Squat Trainer — MJPEG stream + WebSocket + Groq AI coaching."""

import asyncio
import json
import os
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from groq import AsyncGroq

# ── Model ──────────────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).parent
MODEL_PATH = BASE_DIR / "pose_landmarker.task"

if not MODEL_PATH.exists():
    print("Downloading pose model…")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
        str(MODEL_PATH),
    )

# ── Math ───────────────────────────────────────────────────────────────────────

def _angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc  = a - b, c - b
    cos     = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _torso_lean(shoulder_mid, hip_mid):
    dx = shoulder_mid[0] - hip_mid[0]
    dy = hip_mid[1] - shoulder_mid[1]
    return abs(float(np.degrees(np.arctan2(abs(dx), max(dy, 1e-6)))))


def _compute_angles(lms):
    """Return a dict of all relevant angles for a frame."""
    def p(i): return [lms[i].x, lms[i].y]

    knee_l = _angle(p(23), p(25), p(27))   # hip-knee-ankle left
    knee_r = _angle(p(24), p(26), p(28))   # hip-knee-ankle right
    hip_l  = _angle(p(11), p(23), p(25))   # shoulder-hip-knee left
    hip_r  = _angle(p(12), p(24), p(26))   # shoulder-hip-knee right

    s_mid = [(lms[11].x + lms[12].x) / 2, (lms[11].y + lms[12].y) / 2]
    h_mid = [(lms[23].x + lms[24].x) / 2, (lms[23].y + lms[24].y) / 2]
    torso = _torso_lean(s_mid, h_mid)

    # Knee cave — mirror-agnostic: is the knee closer to the image centre than the ankle?
    # If the ankle is further from centre than the knee → knee has caved inward.
    right_cave_delta = abs(lms[28].x - 0.5) - abs(lms[26].x - 0.5)  # positive = caved
    left_cave_delta  = abs(lms[27].x - 0.5) - abs(lms[25].x - 0.5)
    right_caved = right_cave_delta > 0.04
    left_caved  = left_cave_delta  > 0.04

    return {
        "knee_l":      round(knee_l, 1),
        "knee_r":      round(knee_r, 1),
        "hip_l":       round(hip_l,  1),
        "hip_r":       round(hip_r,  1),
        "torso_lean":  round(torso,  1),
        "right_caved": right_caved,
        "left_caved":  left_caved,
    }


# ── Hardcoded form checks (fallback while LLM warms up) ───────────────────────

def _check_form(angles, stage):
    issues = {}
    knee_avg = (angles["knee_l"] + angles["knee_r"]) / 2

    if stage != "down" and knee_avg > 140:
        return issues

    # Depth
    if knee_avg < 120:
        issues["depth"] = (knee_avg <= 95, "Go deeper — aim for parallel")

    # Knees (fixed rule)
    caved = angles["right_caved"] or angles["left_caved"]
    issues["knees"] = (not caved, "Knees caving in — push them out")

    # Back
    issues["back"] = (angles["torso_lean"] <= 50,
                      f"Back leaning too far ({angles['torso_lean']:.0f}°)")

    return issues


# ── Drawing ────────────────────────────────────────────────────────────────────

CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(27,29),(27,31),
    (24,26),(26,28),(28,30),(28,32),
]

def _draw_pose(frame, lms, issues):
    h, w  = frame.shape[:2]
    pts   = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
    bad   = any(not v[0] for v in issues.values())
    color = (50, 80, 255) if bad else (80, 220, 80)

    for a, b in CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], color, 2)

    knee_ok = issues.get("knees", (True,))[0]
    for ki in [25, 26]:
        if ki < len(pts):
            cv2.circle(frame, pts[ki], 8, (0,60,255) if not knee_ok else (80,220,80), -1)

    back_ok = issues.get("back", (True,))[0]
    for si in [11, 12]:
        if si < len(pts):
            cv2.circle(frame, pts[si], 8, (0,60,255) if not back_ok else (80,220,80), -1)


# ── Shared state ───────────────────────────────────────────────────────────────

_workout: dict = {
    "reps": 0, "stage": None, "knee_angle": 180.0,
    "form": {}, "active": False, "_reset": False,
    "angles": {},
    "llm_feedback": None,   # current AI coaching message
    "llm_loading":  False,  # True while Groq is generating
}
_workout_lock = threading.Lock()

_latest_result = None
_result_lock   = threading.Lock()

_latest_frame  = None
_frame_lock    = threading.Lock()

# ── MediaPipe ──────────────────────────────────────────────────────────────────

BaseOptions           = mp.tasks.BaseOptions
PoseLandmarker        = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode


def _on_result(result, _img, _ts):
    global _latest_result
    with _result_lock:
        _latest_result = result


_mp_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=_on_result,
)


# ── Processing thread ──────────────────────────────────────────────────────────

def _processing_loop():
    global _latest_frame
    cap      = cv2.VideoCapture(0)
    reps     = 0
    stage    = None
    issues   = {}
    angles   = {}
    knee_avg = 180.0

    with PoseLandmarker.create_from_options(_mp_options) as lmkr:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.02)
                continue

            ts     = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            lmkr.detect_async(mp_img, ts)

            with _result_lock:
                result = _latest_result

            active = False
            if result and result.pose_landmarks:
                active = True
                lms    = result.pose_landmarks[0]
                angles = _compute_angles(lms)
                knee_avg = (angles["knee_l"] + angles["knee_r"]) / 2

                if knee_avg > 160:
                    if stage == "down":
                        reps += 1
                    stage = "up"
                elif knee_avg < 100:
                    stage = "down"

                issues = _check_form(angles, stage)
                _draw_pose(frame, lms, issues)

            with _workout_lock:
                if _workout["_reset"]:
                    reps   = 0
                    stage  = None
                    issues = {}
                    angles = {}
                    _workout["_reset"]       = False
                    _workout["llm_feedback"] = None

                _workout["reps"]       = reps
                _workout["stage"]      = stage
                _workout["knee_angle"] = round(knee_avg, 1)
                _workout["form"]       = {k: {"ok": v[0], "msg": v[1]} for k, v in issues.items()}
                _workout["active"]     = active
                _workout["angles"]     = {**angles, "stage": stage, "reps": reps}

            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with _frame_lock:
                _latest_frame = jpeg.tobytes()

    cap.release()


# ── Groq AI coaching loop ──────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a real-time squat coach. Output EXACTLY one short cue — maximum 6 words. "
    "No punctuation at the end. No filler words. No praise unless form is perfect. "
    "Sound like a coach calling out a cue mid-rep. "
    "Examples of good outputs: "
    "'Heels flat on the floor' | "
    "'Push your knees out' | "
    "'Chest up, stop rounding' | "
    "'Go deeper, hit parallel' | "
    "'Drive through your heels' | "
    "'Brace your core' | "
    "'Keep your back straight' | "
    "'Great depth, keep it up'. "
    "Pick the single most critical fix. If form is good, give a positive drive cue."
)


def _build_user_prompt(angles: dict) -> str:
    stage = angles.get("stage") or "up"
    reps  = angles.get("reps",  0)
    kl    = angles.get("knee_l", 180)
    kr    = angles.get("knee_r", 180)
    torso = angles.get("torso_lean", 0)
    hl    = angles.get("hip_l", 180)
    hr    = angles.get("hip_r", 180)

    issues = []
    if angles.get("right_caved") or angles.get("left_caved"):
        issues.append("knees caving inward")
    if torso > 50:
        issues.append(f"torso leaning {torso:.0f}° forward (back rounding)")
    if stage == "down" and min(kl, kr) > 95:
        issues.append(f"not deep enough — knee angle {min(kl,kr):.0f}° (need <90°)")
    asym = abs(kl - kr)
    if asym > 12:
        issues.append(f"left/right asymmetry: {kl:.0f}° vs {kr:.0f}°")

    issues_str = "; ".join(issues) if issues else "none — form looks good"

    return (
        f"Rep {reps}, phase {stage}. "
        f"Knee angles L={kl:.0f}° R={kr:.0f}°, torso lean={torso:.0f}°, "
        f"hip flexion L={hl:.0f}° R={hr:.0f}°. "
        f"Issues detected: {issues_str}. "
        f"Give the single most important 6-word cue right now."
    )


async def _llm_loop():
    client       = AsyncGroq()
    last_called  = 0.0
    last_reps    = -1
    COOLDOWN     = 3.0  # seconds between calls (unless new rep)

    while True:
        await asyncio.sleep(0.4)

        with _workout_lock:
            active = _workout["active"]
            reps   = _workout["reps"]
            angles = dict(_workout["angles"])

        if not active or not angles:
            continue

        now       = time.time()
        new_rep   = reps != last_reps
        timed_out = (now - last_called) >= COOLDOWN

        if not (new_rep or timed_out):
            continue

        last_called = now
        last_reps   = reps

        with _workout_lock:
            _workout["llm_loading"] = True

        try:
            resp = await client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": _build_user_prompt(angles)},
                ],
                max_tokens=80,
                temperature=0.5,
            )
            feedback = resp.choices[0].message.content.strip()
        except Exception as exc:
            print(f"[Groq] error: {exc}")
            feedback = None

        with _workout_lock:
            if feedback:
                _workout["llm_feedback"] = feedback
            _workout["llm_loading"] = False


# ── FastAPI ────────────────────────────────────────────────────────────────────

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.on_event("startup")
async def _startup():
    threading.Thread(target=_processing_loop, daemon=True).start()
    asyncio.create_task(_llm_loop())


@app.get("/")
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


async def _frame_generator():
    while True:
        with _frame_lock:
            frame = _latest_frame
        if frame:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        await asyncio.sleep(0.033)


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        _frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/reset")
async def reset():
    with _workout_lock:
        _workout["_reset"] = True
    return {"ok": True}


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            with _workout_lock:
                data = {
                    "reps":         _workout["reps"],
                    "stage":        _workout["stage"],
                    "knee_angle":   _workout["knee_angle"],
                    "form":         _workout["form"],
                    "active":       _workout["active"],
                    "llm_feedback": _workout["llm_feedback"],
                    "llm_loading":  _workout["llm_loading"],
                }
            await websocket.send_text(json.dumps(data))
            await asyncio.sleep(0.1)
    except (WebSocketDisconnect, Exception):
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
