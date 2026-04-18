import cv2
import mediapipe as mp
import numpy as np
import threading
import os, urllib.request

if not os.path.exists("pose_landmarker.task"):
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
        "pose_landmarker.task"
    )

# ── Math ─────────────────────────────────────────────────────────────────────

def angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

def torso_lean_angle(shoulder_mid, hip_mid):
    """Angle of torso from vertical in degrees. 0 = upright."""
    dx = shoulder_mid[0] - hip_mid[0]
    dy = hip_mid[1] - shoulder_mid[1]  # y flips: down is positive
    return abs(np.degrees(np.arctan2(abs(dx), max(dy, 1e-6))))

# ── Form checks ───────────────────────────────────────────────────────────────

def check_form(lms, knee_angle, stage):
    """Only run checks during the down phase."""
    issues = {}

    if stage != "down" and knee_angle > 140:
        return issues  # don't flag when standing

    def pt(i): return [lms[i].x, lms[i].y]

    # 1. Depth — only flag at the bottom of the squat
    if knee_angle < 120:
        too_shallow = knee_angle > 95
        issues["depth"] = (
            not too_shallow,
            "Go deeper — aim for parallel"
        )

    # 2. Knee cave (valgus collapse)
    # Right knee (26) should stay to the right of right ankle (28) — higher x
    # Left knee (25) should stay to the left of left ankle (27) — lower x
    right_caved = (lms[28].x - lms[26].x) > 0.04  # ankle is more right than knee
    left_caved  = (lms[25].x - lms[27].x) > 0.04  # knee is more right than ankle
    caved = right_caved or left_caved
    issues["knees"] = (
        not caved,
        "Knees caving in — push them out"
    )

    # 3. Back rounding — torso lean from vertical
    shoulder_mid = [(lms[11].x + lms[12].x) / 2, (lms[11].y + lms[12].y) / 2]
    hip_mid      = [(lms[23].x + lms[24].x) / 2, (lms[23].y + lms[24].y) / 2]
    lean = torso_lean_angle(shoulder_mid, hip_mid)
    # Some lean is fine — flag only excessive (> 50°)
    issues["back"] = (
        lean <= 50,
        f"Back rounding / leaning too far ({lean:.0f}°)"
    )

    return issues

# ── Drawing ───────────────────────────────────────────────────────────────────

CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(27,29),(27,31),
    (24,26),(26,28),(28,30),(28,32)
]

def draw_pose(frame, lms, issues):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]

    any_issue = any(not v[0] for v in issues.values())
    skel_color = (0, 80, 255) if any_issue else (0, 220, 80)

    for a, b in CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], skel_color, 2)

    knee_ok = issues.get("knees", (True,))[0]
    for ki in [25, 26]:
        cv2.circle(frame, pts[ki], 8, (0, 0, 255) if not knee_ok else (0, 220, 80), -1)

    back_ok = issues.get("back", (True,))[0]
    for si in [11, 12]:
        cv2.circle(frame, pts[si], 8, (0, 0, 255) if not back_ok else (0, 220, 80), -1)

def draw_ui(frame, reps, stage, issues, knee_angle):
    h, w = frame.shape[:2]

    # Rep counter
    cv2.rectangle(frame, (0, 0), (190, 70), (0, 0, 0), -1)
    cv2.putText(frame, "SQUATS", (12, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    cv2.putText(frame, str(reps), (12, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)

    # Stage + knee angle
    stage_color = (0, 220, 80) if stage == "up" else (0, 120, 255)
    cv2.putText(frame, (stage or "").upper(), (200, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, stage_color, 2)
    cv2.putText(frame, f"Knee: {knee_angle:.0f}deg", (200, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 0), 2)

    # Form panel bottom-left
    labels = {
        "depth": "Depth",
        "knees": "Knees out",
        "back":  "Back straight",
    }
    panel_h = len(labels) * 38 + 20
    panel_y = h - panel_h
    cv2.rectangle(frame, (0, panel_y), (340, h), (0, 0, 0), -1)

    for i, (key, label) in enumerate(labels.items()):
        state = issues.get(key)
        y = panel_y + 28 + i * 38
        if state is None:
            cv2.putText(frame, f"--  {label}", (14, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
            continue
        passed, msg = state
        color = (0, 210, 80) if passed else (0, 80, 255)
        tick = "OK" if passed else "!!"
        cv2.putText(frame, f"{tick}  {label}", (14, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if not passed:
            cv2.putText(frame, msg, (14, y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 160, 255), 1)

# ── Main ──────────────────────────────────────────────────────────────────────

latest_result = None
result_lock = threading.Lock()

BaseOptions           = mp.tasks.BaseOptions
PoseLandmarker        = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

def on_result(result, output_image, timestamp_ms):
    global latest_result
    with result_lock:
        latest_result = result

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="pose_landmarker.task"),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=on_result
)

cap = cv2.VideoCapture(0)
cv2.namedWindow("Squat Trainer", cv2.WINDOW_NORMAL)

reps = 0
stage = None
issues = {}
knee_angle = 180.0

print("Press 'q' to quit | 'r' to reset reps")

with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        ts = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        landmarker.detect_async(mp_img, ts)

        with result_lock:
            result = latest_result

        if result and result.pose_landmarks:
            lms = result.pose_landmarks[0]

            knee_angle_r = angle([lms[24].x, lms[24].y],
                                 [lms[26].x, lms[26].y],
                                 [lms[28].x, lms[28].y])
            knee_angle_l = angle([lms[23].x, lms[23].y],
                                 [lms[25].x, lms[25].y],
                                 [lms[27].x, lms[27].y])
            knee_angle = (knee_angle_r + knee_angle_l) / 2

            # Rep state machine
            if knee_angle > 160:
                if stage == "down":
                    reps += 1
                stage = "up"
            elif knee_angle < 100:
                stage = "down"

            issues = check_form(lms, knee_angle, stage)
            draw_pose(frame, lms, issues)

        draw_ui(frame, reps, stage, issues, knee_angle)

        cv2.imshow("Squat Trainer", frame)
        cv2.setWindowProperty("Squat Trainer", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            reps = 0
            stage = None
            issues = {}

cap.release()
cv2.destroyAllWindows()