#!/usr/bin/env python3
"""FastAPI server for Squat Trainer — MJPEG stream + WebSocket + Groq AI coaching."""

import asyncio
import json
import os
import threading
import time
import urllib.request
from pathlib import Path
from typing import Optional

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
    "llm_cue":      None,   # short coaching command
    "llm_reason":   None,   # explanation of why
    "llm_loading":  False,
}
_workout_lock = threading.Lock()

# Rep angle buffer — collects snapshots during the down phase of each rep
_rep_buffer:      list[dict] = []
_rep_buffer_lock: threading.Lock = threading.Lock()

# Completed rep stats waiting to be sent to Groq (set by processing thread)
_pending_rep:      dict | None = None
_pending_rep_lock: threading.Lock = threading.Lock()

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


# ── Rep aggregation ────────────────────────────────────────────────────────────

def _aggregate_rep(buf: list[dict], rep_num: int) -> dict:
    """Summarise the worst-case metrics across all frames of a single rep."""
    return {
        "reps":           rep_num,
        "min_knee_l":     min(f["knee_l"]    for f in buf),
        "min_knee_r":     min(f["knee_r"]    for f in buf),
        "max_torso_lean": max(f["torso_lean"] for f in buf),
        "knee_caved_l":   any(f["left_caved"]  for f in buf),
        "knee_caved_r":   any(f["right_caved"] for f in buf),
        "max_asym":       max(abs(f["knee_l"] - f["knee_r"]) for f in buf),
        "min_hip_l":      min(f["hip_l"] for f in buf),
        "min_hip_r":      min(f["hip_r"] for f in buf),
    }


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
                        # Rep complete — aggregate buffer and queue for Groq
                        with _rep_buffer_lock:
                            if _rep_buffer:
                                stats = _aggregate_rep(_rep_buffer.copy(), reps)
                                with _pending_rep_lock:
                                    global _pending_rep
                                    _pending_rep = stats
                                _rep_buffer.clear()
                    stage = "up"
                elif knee_avg < 100:
                    stage = "down"

                # Collect angle snapshots during descent
                if stage == "down" and angles:
                    with _rep_buffer_lock:
                        _rep_buffer.append(dict(angles))

                issues = _check_form(angles, stage)
                _draw_pose(frame, lms, issues)

            with _workout_lock:
                if _workout["_reset"]:
                    reps   = 0
                    stage  = None
                    issues = {}
                    angles = {}
                    _workout["_reset"]     = False
                    _workout["llm_cue"]    = None
                    _workout["llm_reason"] = None
                    with _rep_buffer_lock:
                        _rep_buffer.clear()

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


# ── Groq AI coaching ───────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert squat coach analyzing a completed rep. The athlete can see your feedback on screen.
Return ONLY a JSON object — no markdown, no explanation outside the JSON.

Format:
{"cue": "...", "reason": "..."}

Rules:
- "cue": The single most important fix as a direct command. Max 6 words, no trailing punctuation.
  If form is good: a short motivational drive cue (e.g. "Keep that depth up").
- "reason": One sentence (max 18 words) that explains the detected issue with specific numbers,
  or genuine encouragement if form is clean.
  Bad: "Your form needs work."
  Good: "Left knee angle only reached 102° — parallel needs below 90°."
  Good: "Torso leaned 58° forward — brace core to keep chest up."
  Good: "Both knees hit 87° with perfect alignment — excellent depth."
"""


def _build_rep_prompt(s: dict) -> str:
    kl, kr = s["min_knee_l"], s["min_knee_r"]
    depth_l = "✓ parallel" if kl <= 90 else f"✗ only {kl:.0f}° (need ≤90°)"
    depth_r = "✓ parallel" if kr <= 90 else f"✗ only {kr:.0f}° (need ≤90°)"

    cave_parts = []
    if s["knee_caved_l"]: cave_parts.append("left")
    if s["knee_caved_r"]: cave_parts.append("right")
    cave_str = f"{' & '.join(cave_parts)} knee(s) caved inward" if cave_parts else "none"

    torso = s["max_torso_lean"]
    back_str = f"{torso:.0f}° ({'✓ OK' if torso <= 50 else '✗ too far, limit 50°'})"

    asym = s["max_asym"]
    asym_str = f"{asym:.0f}° ({'✓ balanced' if asym <= 10 else '✗ significant imbalance'})"

    return (
        f"Rep #{s['reps']} just completed. Worst-case angles during the entire rep:\n"
        f"Left knee depth:  {depth_l}\n"
        f"Right knee depth: {depth_r}\n"
        f"Torso lean:       {back_str}\n"
        f"Knee cave:        {cave_str}\n"
        f"L/R asymmetry:    {asym_str}\n"
        f"Hip flexion:      L={s['min_hip_l']:.0f}°  R={s['min_hip_r']:.0f}°\n\n"
        f"Return the JSON coaching feedback."
    )


async def _llm_loop():
    global _pending_rep
    client = AsyncGroq()

    while True:
        await asyncio.sleep(0.15)

        # Grab pending rep data if available
        with _pending_rep_lock:
            rep_data = _pending_rep
            if rep_data:
                _pending_rep = None

        if not rep_data:
            continue

        with _workout_lock:
            _workout["llm_loading"] = True

        cue = reason = None
        try:
            resp = await client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": _build_rep_prompt(rep_data)},
                ],
                max_tokens=120,
                temperature=0.4,
                response_format={"type": "json_object"},
            )
            raw  = resp.choices[0].message.content.strip()
            parsed = json.loads(raw)
            cue    = parsed.get("cue",    "").strip()
            reason = parsed.get("reason", "").strip()
            print(f"[Groq] rep {rep_data['reps']}: {cue} | {reason}")
        except Exception as exc:
            print(f"[Groq] error: {exc}")

        with _workout_lock:
            if cue:
                _workout["llm_cue"]    = cue
                _workout["llm_reason"] = reason
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
                    "reps":        _workout["reps"],
                    "stage":       _workout["stage"],
                    "knee_angle":  _workout["knee_angle"],
                    "form":        _workout["form"],
                    "active":      _workout["active"],
                    "llm_cue":     _workout["llm_cue"],
                    "llm_reason":  _workout["llm_reason"],
                    "llm_loading": _workout["llm_loading"],
                }
            await websocket.send_text(json.dumps(data))
            await asyncio.sleep(0.1)
    except (WebSocketDisconnect, Exception):
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
