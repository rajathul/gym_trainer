#!/usr/bin/env python3
"""FastAPI server for TRAINER — MJPEG stream + WebSocket + Groq AI coaching.
Supports squat, pushup, and lunge with exercise-specific prompts from plans/trainer_form_coach_prompt.md
"""

import asyncio
import json
import os
import re
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
from pydantic import BaseModel

# ── Model ──────────────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).parent
MODEL_PATH = BASE_DIR / "pose_landmarker.task"

if not MODEL_PATH.exists():
    print("Downloading pose model…")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
        str(MODEL_PATH),
    )

# ── Load exercise prompts from markdown file ───────────────────────────────────

def _load_exercise_prompts() -> dict[str, str]:
    prompt_file = BASE_DIR / "plans" / "trainer_form_coach_prompt.md"
    if not prompt_file.exists():
        print("[Prompts] trainer_form_coach_prompt.md not found — using fallback prompts")
        return {}
    content = prompt_file.read_text()
    blocks = re.findall(r"```\n(.*?)```", content, re.DOTALL)
    prompts = {
        "squat": blocks[0].strip() if len(blocks) > 0 else "",
        "multi": blocks[1].strip() if len(blocks) > 1 else "",
    }
    print(f"[Prompts] Loaded {len(blocks)} prompt block(s) from trainer_form_coach_prompt.md")
    return prompts

_PROMPTS = _load_exercise_prompts()

_FALLBACK_SYSTEM_PROMPT = """\
You are TRAINER, a silent gym coach. Analyze the provided rep data and return coaching feedback.
Return ONLY: {"cue": "<5-word command or empty>", "error": "<tag or null>"}
"""

def _get_system_prompt(exercise: str) -> str:
    if exercise == "squat":
        return _PROMPTS.get("squat") or _FALLBACK_SYSTEM_PROMPT
    return _PROMPTS.get("multi") or _FALLBACK_SYSTEM_PROMPT


# Human-readable reasons for each error code returned by the file prompts
_ERROR_REASONS: dict[str, str] = {
    "back_rounded":      "Torso leaning too far forward — brace core and keep chest up",
    "knees_in":          "Knees caving inward — push them out over your toes",
    "heels_up":          "Heels lifting off the ground — plant them firmly",
    "not_deep_enough":   "Not reaching parallel — drive hips lower",
    "hips_sag":          "Hips dropping below the body line — squeeze your core",
    "hips_pike":         "Hips rising above the body line — flatten your back",
    "elbows_flared":     "Elbows flaring out too wide — tuck them at 45°",
    "shallow_pushup":    "Not lowering enough — bring chest closer to the floor",
    "knee_collapse":     "Front knee collapsing inward — track it over your toes",
    "torso_lean":        "Torso leaning forward — keep it upright",
    "back_leg_driving":  "Driving through back leg — push through your front heel",
}

# ── Exercise state ─────────────────────────────────────────────────────────────

_current_exercise: str = "squat"
_exercise_lock = threading.Lock()

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
    """Return all relevant angles for the current frame (all exercises)."""
    def p(i): return [lms[i].x, lms[i].y]

    # Squat / lunge
    knee_l = _angle(p(23), p(25), p(27))
    knee_r = _angle(p(24), p(26), p(28))
    hip_l  = _angle(p(11), p(23), p(25))
    hip_r  = _angle(p(12), p(24), p(26))

    s_mid = [(lms[11].x + lms[12].x) / 2, (lms[11].y + lms[12].y) / 2]
    h_mid = [(lms[23].x + lms[24].x) / 2, (lms[23].y + lms[24].y) / 2]
    torso = _torso_lean(s_mid, h_mid)

    right_cave_delta = abs(lms[28].x - 0.5) - abs(lms[26].x - 0.5)
    left_cave_delta  = abs(lms[27].x - 0.5) - abs(lms[25].x - 0.5)

    # Pushup
    elbow_l = _angle(p(11), p(13), p(15))   # shoulder-elbow-wrist
    elbow_r = _angle(p(12), p(14), p(16))
    body_line  = _angle(p(11), p(23), p(27))  # shoulder-hip-ankle alignment
    elbow_torso = _angle(p(13), p(11), p(23)) # elbow-shoulder-hip (flare angle)

    # Hip sag/pike relative to shoulder-ankle line
    sh = np.array([lms[11].x, lms[11].y])
    an = np.array([lms[27].x, lms[27].y])
    hi = np.array([lms[23].x, lms[23].y])
    line = an - sh
    t = float(np.dot(hi - sh, line) / (np.dot(line, line) + 1e-6))
    proj_y = float(sh[1] + t * line[1])
    hip_offset = float(hi[1] - proj_y)  # positive = sag, negative = pike

    # Stance width: ratio of ankle-to-ankle distance vs shoulder-to-shoulder distance
    ankle_width    = abs(lms[27].x - lms[28].x)
    shoulder_width = abs(lms[11].x - lms[12].x) + 1e-6
    stance_ratio   = float(ankle_width / shoulder_width)

    return {
        # squat / lunge
        "knee_l":       round(knee_l, 1),
        "knee_r":       round(knee_r, 1),
        "hip_l":        round(hip_l,  1),
        "hip_r":        round(hip_r,  1),
        "torso_lean":   round(torso,  1),
        "right_caved":  right_cave_delta > 0.04,
        "left_caved":   left_cave_delta  > 0.04,
        "stance_ratio": round(stance_ratio, 2),
        # pushup
        "elbow_l":      round(elbow_l,    1),
        "elbow_r":      round(elbow_r,    1),
        "body_line":    round(body_line,  1),
        "elbow_torso":  round(elbow_torso, 1),
        "elbow_asym":   round(abs(elbow_l - elbow_r), 1),
        "hip_sag":      hip_offset > 0.05,
        "hip_pike":     hip_offset < -0.05,
    }


# ── Rep detection helpers ──────────────────────────────────────────────────────

def _primary_angle(angles: dict, exercise: str) -> float:
    if exercise == "pushup":
        return (angles["elbow_l"] + angles["elbow_r"]) / 2
    return (angles["knee_l"] + angles["knee_r"]) / 2


def _is_down_threshold(angle: float, exercise: str) -> bool:
    if exercise == "pushup":
        return angle < 100
    return angle < 100  # squat and lunge share the same threshold


def _is_up_threshold(angle: float, exercise: str) -> bool:
    if exercise == "pushup":
        return angle > 150
    elif exercise == "lunge":
        return angle > 140
    return angle > 160


# ── Real-time form checks ──────────────────────────────────────────────────────

def _check_form(angles: dict, stage: Optional[str], exercise: str) -> dict:
    issues: dict = {}

    if exercise == "pushup":
        elbow_avg = (angles["elbow_l"] + angles["elbow_r"]) / 2
        if stage != "down" and elbow_avg > 140:
            return issues
        if angles["hip_sag"]:
            issues["hips"]    = (False, "Hips sagging — squeeze core and glutes")
        elif angles["hip_pike"]:
            issues["hips"]    = (False, "Hips too high — lower them to plank")
        else:
            issues["hips"]    = (True,  "Body line straight")
        issues["elbows"]      = (angles["elbow_torso"] <= 75,
                                 f"Elbows flaring {angles['elbow_torso']:.0f}° — tuck to 45°")
        if stage == "down" and elbow_avg > 100:
            issues["depth"]   = (False, f"Go lower — elbows at {elbow_avg:.0f}° (need ≤90°)")
        asym = angles["elbow_asym"]
        if asym > 15:
            side = "Left" if angles["elbow_l"] > angles["elbow_r"] else "Right"
            issues["elbow_asym"] = (False, f"{side} arm lagging — {asym:.0f}° difference")
        else:
            issues["elbow_asym"] = (True, "Both arms bending evenly")
        return issues

    elif exercise == "lunge":
        knee_avg = (angles["knee_l"] + angles["knee_r"]) / 2
        if stage != "down" and knee_avg > 140:
            return issues
        front_knee = min(angles["knee_l"], angles["knee_r"])
        caved_l, caved_r = angles["left_caved"], angles["right_caved"]
        if caved_l and caved_r:
            issues["knee"] = (False, "Both knees caving — push them out over toes")
        elif caved_l:
            issues["knee"] = (False, "Left knee caving in — drive it outward")
        elif caved_r:
            issues["knee"] = (False, "Right knee caving in — drive it outward")
        else:
            issues["knee"] = (True, "Front knee tracking over toes")
        lean = angles["torso_lean"]
        issues["back"]  = (lean <= 30, f"Lean forward {lean:.0f}° — keep chest tall")
        if stage == "down" and front_knee > 100:
            issues["depth"] = (False, f"Front knee at {front_knee:.0f}° — go lower")
        return issues

    else:  # squat
        knee_avg = (angles["knee_l"] + angles["knee_r"]) / 2
        if stage != "down" and knee_avg > 140:
            return issues

        # Depth
        if knee_avg < 120:
            issues["depth"] = (knee_avg <= 95,
                               f"Depth {knee_avg:.0f}° — drive hips lower (need ≤90°)")

        # Knee cave — left/right specific
        caved_l, caved_r = angles["left_caved"], angles["right_caved"]
        if caved_l and caved_r:
            issues["knees"] = (False, "Both knees caving in — spread the floor apart")
        elif caved_l:
            issues["knees"] = (False, "Left knee caving in — push it out over toes")
        elif caved_r:
            issues["knees"] = (False, "Right knee caving in — push it out over toes")
        else:
            issues["knees"] = (True, "Knees tracking wide over toes")

        # Back / torso lean
        lean = angles["torso_lean"]
        issues["back"] = (lean <= 50,
                          f"Torso {lean:.0f}° — chest up, stay more upright" if lean > 50
                          else "Back straight, chest tall")

        # Knee symmetry — is right knee parallel to left?
        kl, kr = angles["knee_l"], angles["knee_r"]
        asym = abs(kl - kr)
        if asym > 15:
            lagging = "Left" if kl > kr else "Right"
            issues["asym"] = (False,
                              f"{lagging} knee not matching — {asym:.0f}° difference (L:{kl:.0f}° R:{kr:.0f}°)")
        else:
            issues["asym"] = (True, f"Knees even — L:{kl:.0f}°  R:{kr:.0f}°")

        # Stance width — are legs wide enough?
        ratio = angles.get("stance_ratio", 1.0)
        if ratio < 0.7:
            issues["stance"] = (False,
                                f"Stance too narrow ({ratio:.1f}× shoulders) — widen your feet")
        elif ratio > 2.0:
            issues["stance"] = (False,
                                f"Stance very wide ({ratio:.1f}× shoulders) — bring feet closer")
        else:
            issues["stance"] = (True, f"Stance width good ({ratio:.1f}× shoulders)")

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

    # Highlight problem joints
    knee_ok = issues.get("knees", issues.get("knee", (True,)))[0]
    for ki in [25, 26]:
        if ki < len(pts):
            cv2.circle(frame, pts[ki], 8, (0,60,255) if not knee_ok else (80,220,80), -1)

    back_ok = issues.get("back", (True,))[0]
    for si in [11, 12]:
        if si < len(pts):
            cv2.circle(frame, pts[si], 8, (0,60,255) if not back_ok else (80,220,80), -1)


# ── Shared state ───────────────────────────────────────────────────────────────

_workout: dict = {
    "reps": 0, "stage": None, "primary_angle": 180.0,
    "form": {}, "active": False, "_reset": False,
    "angles": {},
    "exercise":    "squat",
    "llm_cue":     None,
    "llm_reason":  None,
    "llm_loading": False,
}
_workout_lock = threading.Lock()

_rep_buffer:      list[dict] = []
_rep_buffer_lock: threading.Lock = threading.Lock()

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

def _aggregate_rep(buf: list[dict], rep_num: int, exercise: str) -> dict:
    base: dict = {"reps": rep_num, "exercise": exercise}

    if exercise == "pushup":
        base.update({
            "min_elbow_l":     min(f["elbow_l"]    for f in buf),
            "min_elbow_r":     min(f["elbow_r"]    for f in buf),
            "min_body_line":   min(f["body_line"]  for f in buf),
            "max_elbow_torso": max(f["elbow_torso"] for f in buf),
            "hip_sag_any":     any(f["hip_sag"]    for f in buf),
            "hip_pike_any":    any(f["hip_pike"]   for f in buf),
        })
    else:  # squat and lunge share the same landmark set
        base.update({
            "min_knee_l":     min(f["knee_l"]    for f in buf),
            "min_knee_r":     min(f["knee_r"]    for f in buf),
            "max_torso_lean": max(f["torso_lean"] for f in buf),
            "knee_caved_l":   any(f["left_caved"]  for f in buf),
            "knee_caved_r":   any(f["right_caved"] for f in buf),
            "max_asym":       max(abs(f["knee_l"] - f["knee_r"]) for f in buf),
            "min_hip_l":      min(f["hip_l"] for f in buf),
            "min_hip_r":      min(f["hip_r"] for f in buf),
        })

    return base


# ── Processing thread ──────────────────────────────────────────────────────────

def _processing_loop():
    global _latest_frame, _pending_rep
    cap         = cv2.VideoCapture(0)
    reps        = 0
    stage       = None
    issues: dict = {}
    angles: dict = {}
    primary_ang = 180.0

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

            with _exercise_lock:
                exercise = _current_exercise

            active = False
            if result and result.pose_landmarks:
                active      = True
                lms         = result.pose_landmarks[0]
                angles      = _compute_angles(lms)
                primary_ang = _primary_angle(angles, exercise)

                if _is_up_threshold(primary_ang, exercise):
                    if stage == "down":
                        reps += 1
                        with _rep_buffer_lock:
                            if _rep_buffer:
                                stats = _aggregate_rep(_rep_buffer.copy(), reps, exercise)
                                with _pending_rep_lock:
                                    _pending_rep = stats
                                _rep_buffer.clear()
                    stage = "up"
                elif _is_down_threshold(primary_ang, exercise):
                    stage = "down"

                if stage == "down" and angles:
                    with _rep_buffer_lock:
                        _rep_buffer.append(dict(angles))

                issues = _check_form(angles, stage, exercise)
                _draw_pose(frame, lms, issues)

            with _workout_lock:
                if _workout["_reset"]:
                    reps        = 0
                    stage       = None
                    issues      = {}
                    angles      = {}
                    _workout["_reset"]     = False
                    _workout["llm_cue"]    = None
                    _workout["llm_reason"] = None
                    with _rep_buffer_lock:
                        _rep_buffer.clear()

                _workout["reps"]          = reps
                _workout["stage"]         = stage
                _workout["primary_angle"] = round(primary_ang, 1)
                _workout["form"]          = {k: {"ok": v[0], "msg": v[1]} for k, v in issues.items()}
                _workout["active"]        = active
                _workout["angles"]        = {**angles, "stage": stage, "reps": reps}
                _workout["exercise"]      = exercise

            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with _frame_lock:
                _latest_frame = jpeg.tobytes()

    cap.release()


# ── Rep prompt builder — matches input format from trainer_form_coach_prompt.md ─

def _build_rep_prompt(s: dict) -> str:
    exercise = s.get("exercise", "squat")

    if exercise == "pushup":
        el = min(s["min_elbow_l"], s["min_elbow_r"])
        payload = {
            "exercise": "pushup",
            "phase": "bottom",
            "rep_count": s["reps"],
            "angles": {
                "elbow_angle":              round(el, 1),
                "shoulder_hip_ankle_angle": round(s["min_body_line"], 1),
                "elbow_torso_angle":        round(s["max_elbow_torso"], 1),
            },
            "flags": {
                "hip_below_line": s["hip_sag_any"],
                "hip_above_line": s["hip_pike_any"],
            },
            "last_cue":          None,
            "ms_since_last_cue": 99999,
        }

    elif exercise == "lunge":
        kl, kr     = s["min_knee_l"], s["min_knee_r"]
        front_knee = min(kl, kr)
        back_knee  = max(kl, kr)
        payload = {
            "exercise": "lunge",
            "phase": "bottom",
            "rep_count": s["reps"],
            "angles": {
                "hip_knee_angle":    round(front_knee, 1),
                "torso_vs_vertical": round(s["max_torso_lean"], 1),
                "back_knee_angle":   round(back_knee, 1),
            },
            "flags": {
                "front_knee_inside_ankle": s["knee_caved_l"] or s["knee_caved_r"],
                "back_leg_dominant":       False,
            },
            "last_cue":          None,
            "ms_since_last_cue": 99999,
        }

    else:  # squat
        kl, kr = s["min_knee_l"], s["min_knee_r"]
        payload = {
            "exercise": "squat",
            "phase": "bottom",
            "rep_count": s["reps"],
            "angles": {
                "hip_knee_angle":    round(min(kl, kr), 1),
                "torso_vs_vertical": round(s["max_torso_lean"], 1),
            },
            "flags": {
                "knees_inside_ankles": s["knee_caved_l"] or s["knee_caved_r"],
                "heels_off_ground":    False,
            },
            "last_cue":          None,
            "ms_since_last_cue": 99999,
        }

    return (
        f"Rep #{s['reps']} just completed. Worst-case values captured during descent:\n"
        + json.dumps(payload, indent=2)
        + "\n\nReturn the JSON coaching feedback."
    )


# ── Groq AI coaching ───────────────────────────────────────────────────────────

async def _llm_loop():
    global _pending_rep
    client = AsyncGroq()

    while True:
        await asyncio.sleep(0.15)

        with _pending_rep_lock:
            rep_data  = _pending_rep
            if rep_data:
                _pending_rep = None

        if not rep_data:
            continue

        with _workout_lock:
            _workout["llm_loading"] = True

        exercise = rep_data.get("exercise", "squat")
        system_prompt = _get_system_prompt(exercise)

        cue = reason = None
        try:
            resp = await client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": _build_rep_prompt(rep_data)},
                ],
                max_tokens=120,
                temperature=0.4,
                response_format={"type": "json_object"},
            )
            raw    = resp.choices[0].message.content.strip()
            parsed = json.loads(raw)
            cue    = parsed.get("cue",    "").strip()
            error  = parsed.get("error",  None)
            # File prompts return `error` tag; map it to a human-readable reason
            reason = _ERROR_REASONS.get(error, "") if error else parsed.get("reason", "")
            print(f"[Groq/{exercise}] rep {rep_data['reps']}: {cue} | {reason}")
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


class ExerciseRequest(BaseModel):
    exercise: str


@app.post("/set_exercise")
async def set_exercise(req: ExerciseRequest):
    global _current_exercise
    allowed = {"squat", "pushup", "lunge"}
    if req.exercise not in allowed:
        return {"ok": False, "error": f"exercise must be one of {allowed}"}
    with _exercise_lock:
        _current_exercise = req.exercise
    # Also reset workout state when switching exercise
    with _workout_lock:
        _workout["_reset"]    = True
        _workout["exercise"]  = req.exercise
    print(f"[Exercise] switched to {req.exercise}")
    return {"ok": True, "exercise": req.exercise}


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
                    "reps":          _workout["reps"],
                    "stage":         _workout["stage"],
                    "primary_angle": _workout["primary_angle"],
                    "form":          _workout["form"],
                    "active":        _workout["active"],
                    "exercise":      _workout["exercise"],
                    "llm_cue":       _workout["llm_cue"],
                    "llm_reason":    _workout["llm_reason"],
                    "llm_loading":   _workout["llm_loading"],
                }
            await websocket.send_text(json.dumps(data))
            await asyncio.sleep(0.1)
    except (WebSocketDisconnect, Exception):
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
