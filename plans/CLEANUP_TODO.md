# Cleanup TODO — `gym_trainer`

This is the remaining security work on `rajathul/gym_trainer`. The hygiene tasks from the earlier plan (gitignore, remove notebook, remove committed binaries, remove `__pycache__`) are already done — do **not** re-do them.

## How to use this document

- Each task has an ID (`SEC-NN`), severity, files touched, problem, fix, and acceptance check.
- Work top-to-bottom. The order is deliberate — cheapest high-impact items first.
- **One commit per SEC-NN.** Keep the history reviewable.
- Use `grep -n` to find the lines referenced — line numbers may shift after earlier fixes land.
- If a fix doesn't apply cleanly (e.g. the code has drifted), stop and ask before improvising.

## Branch note

The repo has two branches: `main` (squat-only) and `prompt_md_rework` (adds pushup/lunge and loads prompts from `plans/trainer_form_coach_prompt.md`). **Apply every fix to both branches** unless a task explicitly says otherwise. The rework branch has some extra endpoints (`/exercise`) that need the same treatment as `/reset`.

## Scope

**In scope:** `server.py`, `pose_webcam.py`, `static/index.html`, `README.md`, `.env.example` (new), `pyproject.toml` (new).

**Out of scope — do NOT touch without asking:**
- `plans/trainer_form_coach_prompt.md` — this file defines the LLM system prompts. Changing it changes model behavior.
- Form-detection thresholds and rep-counting logic (the math in `_compute_angles`, `_check_form`, rep state machine).
- Any feature work. This is a security pass, not a refactor.

---

## SEC-01 — Bind server to localhost by default

**Severity:** Critical
**Files:** `server.py`

**Problem:** `uvicorn.run(app, host="0.0.0.0", ...)` exposes the live webcam MJPEG stream to everyone on the local network. No auth, no TLS.

**Fix:** Replace the `if __name__ == "__main__":` block at the bottom of `server.py` with:

```python
if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port, log_level="info")
```

**Acceptance:**
- Running `python server.py` binds to `127.0.0.1:8000`.
- From a second machine on the same LAN, `curl http://<host-LAN-IP>:8000/` returns connection refused.

---

## SEC-02 — Tighten CORS

**Severity:** High
**Files:** `server.py`

**Problem:** Current config:
```python
CORSMiddleware(allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
```

**Fix:** Replace the `app.add_middleware(CORSMiddleware, ...)` call with:

```python
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:8000,http://127.0.0.1:8000",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Auth-Token"],
)
```

**Acceptance:**
- `curl -I -H "Origin: https://evil.example" -X OPTIONS http://localhost:8000/reset` does not return `Access-Control-Allow-Origin: *`.
- Normal same-origin requests from the UI still work.

---

## SEC-03 — Add auth token on mutating and streaming endpoints

**Severity:** High
**Files:** `server.py`, `static/index.html`

**Problem:** `/reset`, `/video_feed`, `/ws` (and `/exercise` on rework branch) accept anyone. Once the UI lives on localhost, that's less severe, but still: any process on the machine can hit them.

**Fix:**

1. Near the top of `server.py`, after imports:
```python
import secrets
from fastapi import Depends, HTTPException, Request
from fastapi.responses import HTMLResponse

AUTH_TOKEN = os.getenv("AUTH_TOKEN") or secrets.token_urlsafe(32)
if "AUTH_TOKEN" not in os.environ:
    print(f"[auth] AUTH_TOKEN not set — generated ephemeral: {AUTH_TOKEN}")


async def require_token(request: Request):
    token = request.query_params.get("token") or request.headers.get("x-auth-token")
    if token != AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="invalid token")
```

2. Protect HTTP endpoints by adding `dependencies=[Depends(require_token)]`:
```python
@app.get("/video_feed", dependencies=[Depends(require_token)])
@app.post("/reset", dependencies=[Depends(require_token)])
# Also on rework branch:
@app.post("/exercise", dependencies=[Depends(require_token)])
```

3. Update `/` (the root route) to inject the token into the HTML:
```python
@app.get("/")
async def root():
    html = (STATIC_DIR / "index.html").read_text()
    return HTMLResponse(html.replace("{{AUTH_TOKEN}}", AUTH_TOKEN))
```

4. In `static/index.html`, add this to `<head>`:
```html
<meta name="auth-token" content="{{AUTH_TOKEN}}">
```

5. In the `<script>` block of `static/index.html`, near the top:
```javascript
const AUTH_TOKEN = document.querySelector('meta[name="auth-token"]').content;
```

6. Update `resetSession()` in `static/index.html`:
```javascript
await fetch('/reset', {
  method: 'POST',
  headers: { 'X-Auth-Token': AUTH_TOKEN },
});
```

7. Update the `<img src="/video_feed">` element in the HTML body to include the token:
```html
<img id="videoFeed" src="/video_feed?token={{AUTH_TOKEN}}" />
```
(The `{{AUTH_TOKEN}}` placeholder is replaced server-side in step 3.)

8. Update `connectWS()` to include the token:
```javascript
function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const ws = new WebSocket(`${proto}//${location.host}/ws?token=${encodeURIComponent(AUTH_TOKEN)}`);
  ws.onmessage = (e) => updateUI(JSON.parse(e.data));
  ws.onclose   = () => setTimeout(connectWS, 1500);
  ws.onerror   = () => ws.close();
}
```

9. Update the WebSocket handler in `server.py`:
```python
@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    token = websocket.query_params.get("token")
    if token != AUTH_TOKEN:
        await websocket.close(code=4401)
        return
    await websocket.accept()
    try:
        while True:
            with _workout_lock:
                data = { ... }  # unchanged
            await websocket.send_text(json.dumps(data))
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("ws handler error")  # see SEC-08
```

**Acceptance:**
- `curl http://localhost:8000/video_feed` → 401.
- `curl -H "X-Auth-Token: $AUTH_TOKEN" http://localhost:8000/video_feed` streams MJPEG.
- Loading `/` in a browser still displays the UI and video feed (token is injected server-side).
- WebSocket without `?token=...` closes with code 4401; with the right token, streams normally.

---

## SEC-04 — Gate webcam capture behind explicit start/stop

**Severity:** High
**Files:** `server.py`, `static/index.html`

**Problem:** `cv2.VideoCapture(0)` runs from process startup until the process dies. No user consent model beyond "the server is running." Combined with SEC-01 this is fine for localhost, but the webcam LED being permanently on is surprising behavior.

**Fix:**

1. In `server.py`, above `_processing_loop`, add:
```python
_capture_event = threading.Event()  # set = capture, clear = idle
```

2. Modify `_processing_loop` so the capture handle is created/released based on the event:
```python
def _processing_loop():
    global _latest_frame
    cap = None
    reps, stage, issues, angles, knee_avg = 0, None, {}, {}, 180.0

    with PoseLandmarker.create_from_options(_mp_options) as lmkr:
        while True:
            if not _capture_event.is_set():
                if cap is not None:
                    cap.release()
                    cap = None
                    with _frame_lock:
                        _latest_frame = None
                time.sleep(0.1)
                continue

            if cap is None:
                cap = cv2.VideoCapture(0)

            ret, frame = cap.read()
            if not ret:
                time.sleep(0.02)
                continue
            # ... rest of the existing loop body unchanged
```

3. Add endpoints:
```python
@app.post("/capture/start", dependencies=[Depends(require_token)])
async def capture_start():
    _capture_event.set()
    return {"ok": True, "capturing": True}


@app.post("/capture/stop", dependencies=[Depends(require_token)])
async def capture_stop():
    _capture_event.clear()
    return {"ok": True, "capturing": False}
```

4. Default state: `_capture_event` is NOT set. So the webcam stays off until the user clicks Start.

5. In `static/index.html`, add a Start/Stop button to the header (next to Reset) and wire it:
```javascript
async function startCapture() {
  await fetch('/capture/start', { method: 'POST', headers: { 'X-Auth-Token': AUTH_TOKEN } });
}
async function stopCapture() {
  await fetch('/capture/stop',  { method: 'POST', headers: { 'X-Auth-Token': AUTH_TOKEN } });
}
```

**Acceptance:**
- Starting the server does NOT activate the webcam LED.
- Clicking "Start" in the UI activates the LED and begins streaming.
- Clicking "Stop" deactivates the LED.
- `curl -X POST -H "X-Auth-Token: $AUTH_TOKEN" http://localhost:8000/capture/start` works; without the token returns 401.

---

## SEC-05 — Fix XSS in Session Log history entries

**Severity:** High
**Files:** `static/index.html`

**Problem:** In `addHistoryEntry()`, `cue` and `reason` (both LLM-sourced strings) are interpolated into an `innerHTML` template literal:
```javascript
entry.innerHTML = `
  <div class="history-meta">...</div>
  <div style="flex:1;min-width:0;">
    <div class="history-text">${cue}</div>
    ${reason ? `<div class="history-reason">${reason}</div>` : ''}
  </div>`;
```
The LLM output is not a security boundary. A malformed or adversarial response can inject markup. This is on both `main` and `prompt_md_rework`.

**Fix:** Replace the whole `addHistoryEntry` body with explicit DOM construction:

```javascript
function addHistoryEntry(cue, reason, reps) {
  const elapsed = Math.floor((Date.now() - sessionStart) / 1000);
  const mm = String(Math.floor(elapsed / 60)).padStart(2, '0');
  const ss = String(elapsed % 60).padStart(2, '0');

  document.getElementById('historyEmpty')?.remove();

  const entry = document.createElement('div');
  entry.className = 'history-entry';

  const meta = document.createElement('div');
  meta.className = 'history-meta';
  const rep = document.createElement('span');
  rep.className = 'history-rep';
  rep.textContent = `Rep\u00A0${reps}`;
  const time = document.createElement('span');
  time.className = 'history-time';
  time.textContent = `${mm}:${ss}`;
  meta.append(rep, time);

  const body = document.createElement('div');
  body.style.cssText = 'flex:1;min-width:0;';
  const text = document.createElement('div');
  text.className = 'history-text';
  text.textContent = cue;
  body.append(text);
  if (reason) {
    const reasonEl = document.createElement('div');
    reasonEl.className = 'history-reason';
    reasonEl.textContent = reason;
    body.append(reasonEl);
  }

  entry.append(meta, body);
  document.getElementById('historyList').prepend(entry);

  historyCount++;
  document.getElementById('historyCount').textContent = historyCount;
}
```

**Do not** change the developer-controlled `innerHTML` usages elsewhere (e.g. the `cfg.cards.map(...).join('')` assignment on the rework branch is static data — leave it).

**Acceptance:**
- `grep -n "innerHTML" static/index.html` — any remaining hits assign only developer-authored static strings, never server-sourced data.
- Test: manually send a WebSocket message where `llm_cue` is `<img src=x onerror=alert(1)>` (use browser devtools to simulate). The text must render as literal characters, not execute.

---

## SEC-06 — Move `GROQ_API_KEY` to `.env`

**Severity:** Medium
**Files:** `server.py`, `.env.example` (new), `README.md`, `pyproject.toml` (from SEC-07)

**Problem:** README tells users to `export GROQ_API_KEY=...` in their shell. Keys end up in `.zshrc` / shell history.

**Fix:**

1. Add `python-dotenv` to dependencies in `pyproject.toml` (SEC-07).

2. At the top of `server.py`, before reading any env vars:
```python
from dotenv import load_dotenv
load_dotenv()
```

3. Add a startup check just before `AsyncGroq()` is instantiated (or at the top of `_llm_loop`):
```python
if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError(
        "GROQ_API_KEY is required. Copy .env.example to .env and fill it in."
    )
```

4. Create `.env.example` at the repo root:
```
# Groq API key — get one at https://console.groq.com/
GROQ_API_KEY=your_key_here

# Auth token for UI. Generate with: python -c "import secrets; print(secrets.token_urlsafe(32))"
# If blank, server generates an ephemeral token and prints it to stdout on startup.
AUTH_TOKEN=

# Network binding
HOST=127.0.0.1
PORT=8000
ALLOWED_ORIGINS=http://localhost:8000,http://127.0.0.1:8000
```

5. Update `README.md` setup section:
```markdown
## Setup

```bash
# 1. Create venv and install dependencies
python -m venv .venv
.venv/bin/pip install -e .

# 2. Configure secrets
cp .env.example .env
# edit .env and set GROQ_API_KEY (and optionally AUTH_TOKEN)

# 3. Run
.venv/bin/python server.py
```

Open `http://localhost:8000` in a browser.
```
`.env` is already listed in `.gitignore` — verify it's still there.

**Acceptance:**
- Running without a `.env` (or without `GROQ_API_KEY` set) fails with a clear message pointing at `.env.example`.
- With a valid `.env`, server starts normally.
- `.env` is not tracked by git.

---

## SEC-07 — Pin dependencies with `pyproject.toml`

**Severity:** Medium
**Files:** `pyproject.toml` (new), `README.md`

**Problem:** No dependency manifest. README has a one-liner pip install with no versions.

**Fix:** Create `pyproject.toml` at the repo root:

```toml
[project]
name = "gym-trainer"
version = "0.1.0"
description = "Real-time squat form analysis with MediaPipe + Groq."
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115,<0.116",
    "uvicorn[standard]>=0.32,<0.33",
    "groq>=0.13,<0.14",
    "mediapipe>=0.10.14,<0.11",
    "opencv-python>=4.10,<5.0",
    "numpy>=1.26,<3.0",
    "python-dotenv>=1.0,<2.0",
    "slowapi>=0.1.9,<0.2",
]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
py-modules = ["server", "pose_webcam"]
```

Update README install step to `pip install -e .` (already done in SEC-06).

**Acceptance:** From a fresh venv, `pip install -e .` installs everything needed; `python server.py` runs without `ModuleNotFoundError`.

---

## SEC-08 — Replace `print` and bare `except` with structured logging

**Severity:** Low
**Files:** `server.py`

**Problem:**
- `except (WebSocketDisconnect, Exception): pass` swallows all errors silently.
- `print(f"[Groq] error: {exc}")` is not log-aggregator-friendly.
- Several other `print(...)` calls for status.

**Fix:**

1. Near the top of `server.py`:
```python
import logging

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("gym_trainer")
```

2. Replace every `print(...)` in `server.py` with `log.info(...)`, `log.warning(...)`, or `log.error(...)` as appropriate.

3. Replace the WebSocket handler's `except (WebSocketDisconnect, Exception): pass` with:
```python
except WebSocketDisconnect:
    log.info("ws client disconnected")
except Exception:
    log.exception("ws handler error")
```

4. Replace the Groq loop's `print(f"[Groq] error: {exc}")` with:
```python
log.exception("groq call failed (rep=%s)", rep_data.get("reps"))
```

**Acceptance:** `grep -n "print(" server.py` returns zero hits (or only intentional one-time banners like the ephemeral-token notice in SEC-03, which is OK to leave as `print` since it runs once pre-logging-setup).

---

## SEC-09 — Migrate `@app.on_event("startup")` to lifespan

**Severity:** Low
**Files:** `server.py`

**Problem:** `on_event` is deprecated in FastAPI 0.110+.

**Fix:** Replace the startup block:
```python
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # startup
    threading.Thread(target=_processing_loop, daemon=True).start()
    llm_task = asyncio.create_task(_llm_loop())
    try:
        yield
    finally:
        # shutdown
        llm_task.cancel()
        try:
            await llm_task
        except asyncio.CancelledError:
            pass


app = FastAPI(lifespan=lifespan)

# ... CORS middleware, static mount, etc.
```

Delete the old `@app.on_event("startup")` function.

**Acceptance:** No `DeprecationWarning` on startup. `Ctrl+C` shuts down cleanly without hanging.

---

## SEC-10 — Rate-limit mutating endpoints

**Severity:** Low
**Files:** `server.py`

**Problem:** State-mutating endpoints are unthrottled. Defense-in-depth behind the auth token.

**Fix:**
1. `slowapi` is already in `pyproject.toml` from SEC-07.
2. Set up the limiter:
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
```
3. Apply to endpoints:
```python
@app.post("/reset", dependencies=[Depends(require_token)])
@limiter.limit("30/minute")
async def reset(request: Request):
    ...

@app.post("/capture/start", dependencies=[Depends(require_token)])
@limiter.limit("10/minute")
async def capture_start(request: Request):
    ...

# Same for /capture/stop and (on rework branch) /exercise
```

Note: SlowAPI requires the endpoint to accept a `request: Request` parameter.

**Acceptance:** `for i in {1..40}; do curl -X POST -H "X-Auth-Token: $AUTH_TOKEN" http://localhost:8000/reset; done` produces `429 Too Many Requests` after the first 30.

---

## SEC-11 — Pin MediaPipe model SHA256

**Severity:** Low
**Files:** `server.py`

**Problem:** `urllib.request.urlretrieve(...)` downloads `pose_landmarker_heavy.task` with no integrity check.

**Fix:** Replace the model-download block:

```python
import hashlib

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
# After the first successful download, compute: shasum -a 256 pose_landmarker.task
# and paste the result here. Update on intentional model version changes only.
MODEL_SHA256 = "PASTE_HASH_HERE"


def _verify_model(path: Path) -> bool:
    if MODEL_SHA256 == "PASTE_HASH_HERE":
        return True  # skip check until hash is filled in
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest() == MODEL_SHA256


if not MODEL_PATH.exists() or not _verify_model(MODEL_PATH):
    if MODEL_PATH.exists():
        log.warning("model SHA256 mismatch — re-downloading")
        MODEL_PATH.unlink()
    log.info("downloading pose model…")
    urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))
    if MODEL_SHA256 != "PASTE_HASH_HERE" and not _verify_model(MODEL_PATH):
        MODEL_PATH.unlink()
        raise RuntimeError("downloaded model failed SHA256 verification")
```

Then: run the server once so it downloads the model, `shasum -a 256 pose_landmarker.task`, paste the hash into `MODEL_SHA256`, commit.

**Acceptance:** Tampering (`echo x >> pose_landmarker.task`) causes the next startup to delete and re-download the file.

---

## SEC-12 — Extract shared logic into `squat_core.py`

**Severity:** Low (maintainability; listed here because duplicated code drifts, including security-relevant detection logic)
**Files:** `squat_core.py` (new), `server.py`, `pose_webcam.py`

**Problem:** `_angle`, `_torso_lean`, `_compute_angles`, `_check_form`, `CONNECTIONS`, `_draw_pose` are duplicated across the two scripts with subtle differences. The `main` branch's `server.py` knee-cave check is mirror-agnostic; `pose_webcam.py`'s version assumes a fixed mirror. The `server.py` version is correct — use it as canonical.

**Fix:**

1. Create `squat_core.py` containing the canonical versions of:
   - `angle(a, b, c)`
   - `torso_lean(shoulder_mid, hip_mid)`
   - `compute_angles(lms)`
   - `check_form(angles, stage)`
   - `CONNECTIONS`
   - `draw_pose(frame, lms, issues)`

2. Import from it in both `server.py` and `pose_webcam.py`. Delete the duplicated private `_` versions.

3. For the rework branch: the multi-exercise angle logic (pushup/lunge) stays in `server.py` for now — only the squat-shared pieces move to `squat_core.py`. Don't expand scope.

**Acceptance:**
- `grep -n "def angle\|def _angle\|def compute_angles\|def _compute_angles" *.py` — each function defined exactly once.
- `python server.py` still runs; `python pose_webcam.py` still runs and produces the same rep counts on the same input.

---

## Final verification checklist

Before marking this TODO done, confirm:

- [ ] `curl http://localhost:8000/video_feed` returns 401.
- [ ] `curl -X POST http://localhost:8000/reset` returns 401.
- [ ] WebSocket connection without a token is refused with close code 4401.
- [ ] Starting the server does NOT activate the webcam LED; clicking Start in the UI does.
- [ ] The UI still works end-to-end (video streams, reps count, AI coach shows cues, Session Log accumulates).
- [ ] `grep -n "innerHTML" static/index.html` — only static, developer-authored strings.
- [ ] `grep -n "print(" server.py` — zero hits (or just the intentional token banner).
- [ ] `grep -n "0.0.0.0" server.py` — zero hits.
- [ ] `grep -n "allow_origins=\[\"\\*\"\]" server.py` — zero hits.
- [ ] `grep -n "on_event" server.py` — zero hits.
- [ ] Starting the server without `GROQ_API_KEY` fails with a clear error pointing at `.env.example`.
- [ ] `pip install -e .` in a fresh venv installs everything needed.
- [ ] Rate limit of 30/minute on `/reset` triggers `429` after 30 rapid calls.
- [ ] `.env` is git-ignored and untracked.
- [ ] Both branches (`main` and `prompt_md_rework`) have all applicable fixes applied.

## If anything is unclear

Stop and ask before guessing. Specifically:
- If the code has drifted and a snippet doesn't apply cleanly.
- If a fix to the rework branch would affect the multi-exercise prompt loading logic (`_load_exercise_prompts` and the `plans/trainer_form_coach_prompt.md` parser).
- If introducing the auth token breaks an integration I wasn't told about.
