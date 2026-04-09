import cv2
import numpy as np
import json
import os
import time
import sys
from datetime import datetime
from collections import deque
from numpy.linalg import norm
from openpyxl import load_workbook, Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# ── SSL workaround for corporate/school proxies ──────────
# Disables certificate verification so model downloads work
# behind firewalls that use self-signed certificates.
import ssl
import urllib3
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

# ══════════════════════════════════════════════════════════
#   AXIOM — Face Recognition Attendance System
#   Engine: InsightFace ArcFace (512-dim neural embeddings)
#
#   What changed from the old version:
#     OLD: MediaPipe Face Mesh → 18 geometric ratios → custom math
#     NEW: RetinaFace detection → ArcFace embedding → cosine similarity
#
#   What stayed:
#     ✅ Vote buffer (deque-based rolling votes)
#     ✅ Confirmation timer (hold face N seconds)
#     ✅ Excel attendance logging (per-session sheets)
#     ✅ HUD overlay (FPS, present, unknown, time)
#     ✅ View/delete registered persons
#     ✅ Main menu loop
# ══════════════════════════════════════════════════════════

# ── Configuration ─────────────────────────────────────────
PROFILES_FILE    = "face_profiles.json"
ATTENDANCE_FILE  = "attendance.xlsx"

# Cosine similarity threshold: 0.0 (no match) to 1.0 (identical)
# Typical: 0.3-0.4 for verification, 0.5+ for very confident
MATCH_THRESHOLD  = 0.35    # below this = unknown
MATCH_THRESHOLD_HIGH = 0.50  # above this = very confident match

UNKNOWN_COOLDOWN = 10      # seconds between logging same unknown

# Confirmation: face must be held steady for this many seconds
CONFIRM_SECONDS  = 2.5

# Rolling vote: must match in N of M consecutive frames
VOTE_WINDOW      = 12
VOTE_REQUIRE     = 8

# Registration: how many embedding samples to capture
REGISTER_SAMPLES = 20      # captures from multiple angles
REGISTER_DELAY   = 0.25    # seconds between captures (forces diversity)

# Camera resolution — request the highest your webcam supports
# InsightFace handles any resolution; bigger = farther detection
CAMERA_WIDTH     = 640
CAMERA_HEIGHT    = 480

# InsightFace detection size — larger = detects smaller faces at distance
# but uses more GPU/CPU. 640 is default, 960 or 1280 for long range.
DET_SIZE         = (320, 320)

# ── InsightFace Initialization ────────────────────────────

def init_face_analyzer():
    """
    Initialize InsightFace with the buffalo_l model.
    First run auto-downloads ~335MB model to ~/.insightface/models/
    """
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        print("=" * 50)
        print("❌ InsightFace is not installed!")
        print("   Run: pip install insightface onnxruntime")
        print("=" * 50)
        sys.exit(1)

    print("🔄 Loading InsightFace model (buffalo_sc)...")
    print("   (First run downloads ~16MB model — be patient)")

    # Patch requests to skip SSL verification (for proxied networks)
    import requests
    _original_get = requests.get
    def _patched_get(*args, **kwargs):
        kwargs.setdefault("verify", False)
        return _original_get(*args, **kwargs)
    requests.get = _patched_get

    app = FaceAnalysis(
        name="buffalo_sc",
        # providers: CUDA if GPU available, else CPU
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    app.prepare(ctx_id=0, det_size=DET_SIZE)

    print("✅ InsightFace loaded successfully!")
    return app

# Lazy-load: only initialize when needed
_face_app = None

def get_face_app():
    global _face_app
    if _face_app is None:
        _face_app = init_face_analyzer()
    return _face_app

# ── Embedding Comparison ──────────────────────────────────

def cosine_similarity(emb1, emb2):
    """
    Cosine similarity between two 512-dim embedding vectors.
    Returns: float in [-1, 1] where 1 = identical faces.
    """
    return float(np.dot(emb1, emb2) / (norm(emb1) * norm(emb2)))


def compare_faces(stored_embedding, live_embedding):
    """
    Compare stored vs live face embedding.
    Returns: (similarity_score, is_match)
      - similarity_score: 0.0 to 1.0 (percentage-like display)
      - is_match: True if similarity exceeds threshold
    """
    sim = cosine_similarity(stored_embedding, live_embedding)

    # Convert to a 0-100 display score
    # Cosine sim for same person is typically 0.3-0.7
    # Map 0.2..0.7 → 0..100 for display purposes
    display_score = max(0, min(100, (sim - 0.15) / 0.55 * 100))

    is_match = sim >= MATCH_THRESHOLD
    return round(display_score, 1), is_match, sim


# ── Profile Storage ───────────────────────────────────────

def load_profiles():
    if os.path.exists(PROFILES_FILE):
        with open(PROFILES_FILE, "r") as f:
            data = json.load(f)
            # Convert embedding lists back to numpy arrays
            for name, person in data.items():
                if "embedding" in person and isinstance(person["embedding"], list):
                    person["embedding"] = np.array(person["embedding"], dtype=np.float32)
            return data
    return {}


def save_profiles(data):
    # Convert numpy arrays to lists for JSON serialization
    serializable = {}
    for name, person in data.items():
        p = dict(person)
        if "embedding" in p and isinstance(p["embedding"], np.ndarray):
            p["embedding"] = p["embedding"].tolist()
        serializable[name] = p

    with open(PROFILES_FILE, "w") as f:
        json.dump(serializable, f, indent=4)


# ── Excel Attendance (per-session sheets) ─────────────────

SESSION_SHEET_PREFIX = "Session_"

def get_session_sheet_name():
    """Each run gets its own sheet: Session_2025-07-14_143022"""
    return SESSION_SHEET_PREFIX + datetime.now().strftime("%Y-%m-%d_%H%M%S")

def create_attendance_workbook(filepath):
    wb = Workbook()
    ws = wb.active
    ws.title = "Summary"
    ws["A1"] = "AXIOM Face Attendance System"
    ws["A1"].font = Font(bold=True, size=14)
    ws["A2"] = "Each session is logged in its own sheet."
    ws["A2"].font = Font(color="888888", size=10)
    wb.save(filepath)
    return wb

def init_session_sheet(filepath, sheet_name):
    """Create a new sheet for this session with a formatted header."""
    if not os.path.exists(filepath):
        create_attendance_workbook(filepath)

    wb = load_workbook(filepath)
    ws = wb.create_sheet(title=sheet_name)

    headers = ["Name", "Date", "Time", "Status", "Confidence", "Confirmed After (s)"]
    header_fill   = PatternFill("solid", fgColor="1F4E79")
    header_font   = Font(bold=True, color="FFFFFF", name="Arial", size=11)
    header_align  = Alignment(horizontal="center", vertical="center")
    thin_border   = Border(
        bottom=Side(style="thin", color="AAAAAA"),
        right=Side(style="thin",  color="DDDDDD"),
    )

    for col_idx, header in enumerate(headers, start=1):
        cell = ws.cell(row=1, column=col_idx, value=header)
        cell.font      = header_font
        cell.fill      = header_fill
        cell.alignment = header_align
        cell.border    = thin_border

    ws.row_dimensions[1].height = 22
    col_widths = [22, 14, 12, 12, 14, 20]
    for i, width in enumerate(col_widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = width

    wb.save(filepath)
    return sheet_name


def append_attendance_row(filepath, sheet_name, name, status, score, confirm_secs=""):
    """Append one attendance row to the session sheet with alternating row color."""
    for attempt in range(3):
        try:
            wb = load_workbook(filepath)
            ws = wb[sheet_name]

            now      = datetime.now()
            next_row = ws.max_row + 1

            row_fill = PatternFill("solid", fgColor="EBF3FB" if next_row % 2 == 0 else "FFFFFF")

            values = [
                name,
                now.strftime("%Y-%m-%d"),
                now.strftime("%H:%M:%S"),
                status,
                score,
                confirm_secs,
            ]
            status_colors = {"PRESENT": "1E7A1E", "UNKNOWN": "C00000"}

            for col_idx, val in enumerate(values, start=1):
                cell = ws.cell(row=next_row, column=col_idx, value=val)
                cell.fill      = row_fill
                cell.alignment = Alignment(horizontal="center")
                cell.font      = Font(name="Arial", size=10)
                if col_idx == 4 and val in status_colors:
                    cell.font = Font(
                        name="Arial", size=10,
                        bold=True, color=status_colors[val]
                    )

            # Update summary sheet
            if "Summary" in wb.sheetnames:
                sw = wb["Summary"]
                found = False
                for row in sw.iter_rows(min_row=4):
                    if row[0].value == sheet_name:
                        if status == "PRESENT":
                            row[1].value = (row[1].value or 0) + 1
                        found = True
                        break
                if not found:
                    sr = sw.max_row + 1
                    sw.cell(row=sr, column=1, value=sheet_name)
                    sw.cell(row=sr, column=2, value=1 if status == "PRESENT" else 0)
                    sw.cell(row=sr, column=3, value=datetime.now().strftime("%Y-%m-%d %H:%M"))

            wb.save(filepath)
            return True

        except PermissionError:
            print(f"⚠️  Close Excel and press Enter... ({attempt+1}/3)")
            input()
        except Exception as e:
            print(f"⚠️  Save error: {e}")
            return False
    return False


# ══════════════════════════════════════════════════════════
# OPTION 1 — REGISTER NEW PERSON
# ══════════════════════════════════════════════════════════

def register_live():
    print("\n" + "=" * 50)
    print("          REGISTER NEW PERSON")
    print("=" * 50)

    name = input("Enter your name: ").strip()
    if not name:
        print("❌ Name cannot be empty!")
        return

    data = load_profiles()

    if name in data:
        confirm = input(f"⚠️  '{name}' already exists. Re-register? (y/n): ")
        if confirm.lower() != 'y':
            print("Cancelled.")
            return

    app = get_face_app()

    print(f"\n👤 Registering: {name}")
    print(f"📸 The system will capture {REGISTER_SAMPLES} face snapshots.")
    print(f"   Slowly turn your head left, right, up, down while looking at camera.")
    print(f"   This gives the AI multiple angles to learn your face.")
    print(f"   Press Q to cancel.\n")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        print("❌ Cannot open camera!")
        return

    embeddings     = []
    last_capture   = 0
    frame_count    = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        h, w = frame.shape[:2]
        now_t = time.time()

        # Detect faces in frame
        faces = app.get(frame)

        # ── HUD: Top bar ──────────────────────────────────
        cv2.rectangle(frame, (0, 0), (w, 60), (18, 18, 18), cv2.FILLED)
        cv2.putText(frame, f"Registering: {name}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 215, 60), 2)

        progress_pct = len(embeddings) / REGISTER_SAMPLES
        bar_x1, bar_x2 = w - 210, w - 10
        cv2.rectangle(frame, (bar_x1, 8), (bar_x2, 22), (50, 50, 50), cv2.FILLED)
        filled = int((bar_x2 - bar_x1) * progress_pct)
        if filled > 0:
            cv2.rectangle(frame, (bar_x1, 8), (bar_x1 + filled, 22),
                          (0, 200, 100), cv2.FILLED)
        cv2.putText(frame, f"{len(embeddings)}/{REGISTER_SAMPLES}",
                    (bar_x1 + 4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (220, 255, 220), 1)

        if faces:
            face = faces[0]  # Use first detected face
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox

            # Quality check: face must be reasonably sized
            face_w = x2 - x1
            face_h = y2 - y1
            face_area = face_w * face_h
            min_face_area = (w * h) * 0.008  # face must be at least 0.8% of frame

            quality_ok = face_area >= min_face_area
            det_score = face.det_score if hasattr(face, 'det_score') else 0

            if quality_ok and det_score > 0.5:
                # Draw green box around face
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 100), 2)

                # Draw glowing hacker landmarks for the hackathon judges!
                if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
                    for pt in face.landmark_2d_106:
                        cv2.circle(frame, (int(pt[0]), int(pt[1])), 1, (255, 255, 0), -1)
                elif hasattr(face, 'kps') and face.kps is not None:
                    # Fallback: draw the 5 keypoints (eyes, nose, mouth corners)
                    for pt in face.kps:
                        cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (255, 255, 0), -1)

                # Capture embedding with delay for diversity
                if now_t - last_capture >= REGISTER_DELAY:
                    embedding = face.normed_embedding
                    if embedding is not None:
                        embeddings.append(embedding)
                        last_capture = now_t

                status_text = f"Capturing... ({len(embeddings)}/{REGISTER_SAMPLES})"
                if len(embeddings) < REGISTER_SAMPLES // 3:
                    hint = "Look straight, then slowly turn left/right"
                elif len(embeddings) < 2 * REGISTER_SAMPLES // 3:
                    hint = "Now tilt chin up and down slightly"
                else:
                    hint = "Almost done! Any remaining angles"

                cv2.putText(frame, status_text, (10, 46),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 215, 160), 1)
                cv2.putText(frame, hint, (10, h - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

                # Detection confidence
                cv2.putText(frame, f"det: {det_score:.2f}",
                            (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (0, 200, 100), 1)
            else:
                # Face too small or low confidence
                if not quality_ok:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 2)
                    cv2.putText(frame, "Move closer", (10, 46),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 140, 255), 1)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 100, 200), 2)
                    cv2.putText(frame, "Low quality — adjust position", (10, 46),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 100, 200), 1)
        else:
            cv2.putText(frame, "No face detected — look at camera", (10, 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 200), 1)

        # Bottom hints
        cv2.putText(frame, "S = save now   |   Q = cancel",
                    (w // 2 - 140, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (90, 90, 90), 1)

        cv2.imshow("AXIOM — Face Registration", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), ord('Q')]:
            print("Registration cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            return
        elif key in [ord('s'), ord('S')]:
            if len(embeddings) < 5:
                print(f"⚠️  Only {len(embeddings)} samples — need at least 5. Keep going!")
            else:
                print(f"\n💾 Saving with {len(embeddings)} samples...")
                break

        # Auto-save when target reached
        if len(embeddings) >= REGISTER_SAMPLES:
            print(f"\n✅ {REGISTER_SAMPLES} samples captured! Saving...")
            time.sleep(0.3)
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(embeddings) == 0:
        print("❌ No samples collected.")
        return

    # Average all embeddings → robust face identity
    avg_embedding = np.mean(embeddings, axis=0)
    # Re-normalize after averaging
    avg_embedding = avg_embedding / norm(avg_embedding)

    data[name] = {
        "name"          : name,
        "embedding"     : avg_embedding,
        "total_samples" : len(embeddings),
        "registered_on" : datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    save_profiles(data)

    # Show quality stats
    # Compute how consistent the captures were (lower spread = better)
    sims = [cosine_similarity(avg_embedding, e) for e in embeddings]
    avg_sim = np.mean(sims)
    min_sim = np.min(sims)

    print(f"\n✅ {name} registered successfully!")
    print(f"   Samples collected : {len(embeddings)}")
    print(f"   Avg consistency   : {avg_sim:.3f}")
    print(f"   Min consistency   : {min_sim:.3f}")
    if min_sim < 0.5:
        print(f"   ⚠️  Some captures were inconsistent — consider re-registering")
    else:
        print(f"   🎯 Profile quality: {'Excellent' if avg_sim > 0.75 else 'Good'}")


# ══════════════════════════════════════════════════════════
# OPTION 2 — MARK ATTENDANCE (with vote buffer + confirmation)
# ══════════════════════════════════════════════════════════

def mark_attendance():
    data = load_profiles()

    if not data:
        print("\n❌ No profiles registered yet!")
        return

    app = get_face_app()

    print(f"\n📋 {len(data)} registered persons loaded")
    print(f"⏱  A face must be held for {CONFIRM_SECONDS}s to be marked")
    print("Controls: Q = Quit (cancel session) | S = Save & Quit")
    print("=" * 50)

    # Setup Excel session sheet
    sheet_name = get_session_sheet_name()
    init_session_sheet(ATTENDANCE_FILE, sheet_name)
    print(f"📄 Logging to sheet: {sheet_name}")

    today = datetime.now().strftime("%Y-%m-%d")
    marked_today = set()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        print("❌ Cannot open camera!")
        return

    # Per-face tracking state
    vote_buffer    = {}    # name → deque of recent bool matches
    confirm_start  = {}    # name → time when continuous match streak started
    confirmed_set  = set() # names that have been marked present

    unknown_count       = 0
    unknown_last_logged = {}

    frame_count   = 0
    fps           = 0
    fps_timer     = time.time()
    frame_counter = 0

    last_locations  = []
    last_labels     = []
    last_colors     = []
    last_progresses = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count   += 1
        frame_counter += 1
        now            = datetime.now()
        h, w           = frame.shape[:2]

        if time.time() - fps_timer >= 1.0:
            fps           = frame_counter
            frame_counter = 0
            fps_timer     = time.time()

        # ── Face detection + recognition ──────────────────
        # Process every frame for maximum responsiveness
        faces = app.get(frame)

        last_locations  = []
        last_labels     = []
        last_colors     = []
        last_progresses = []

        seen_this_frame = set()

        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Draw glowing hacker landmarks for the hackathon judges!
            if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
                for pt in face.landmark_2d_106:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 1, (255, 255, 0), -1)
            elif hasattr(face, 'kps') and face.kps is not None:
                # Fallback: draw the 5 keypoints (eyes, nose, mouth corners)
                for pt in face.kps:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (255, 255, 0), -1)

            live_embedding = face.normed_embedding
            if live_embedding is None:
                continue

            # Compare against all registered profiles
            best_name  = None
            best_score = 0
            best_sim   = 0.0

            for pname, person in data.items():
                if "embedding" not in person or person["embedding"] is None:
                    continue
                stored_emb = person["embedding"]
                if isinstance(stored_emb, list):
                    stored_emb = np.array(stored_emb, dtype=np.float32)

                display_score, is_match, raw_sim = compare_faces(
                    stored_emb, live_embedding
                )

                if is_match and display_score > best_score:
                    best_score = display_score
                    best_name  = pname
                    best_sim   = raw_sim

            if best_name:
                seen_this_frame.add(best_name)

                # ── Vote buffer ───────────────────────────
                if best_name not in vote_buffer:
                    vote_buffer[best_name] = deque(maxlen=VOTE_WINDOW)
                vote_buffer[best_name].append(True)

                # Feed False into other person's buffers to prevent stale votes
                for other in list(vote_buffer.keys()):
                    if other != best_name:
                        vote_buffer[other].append(False)

                vote_ok = (
                    len(vote_buffer[best_name]) >= VOTE_WINDOW and
                    sum(vote_buffer[best_name]) >= VOTE_REQUIRE
                )

                # ── Confirmation timer ────────────────────
                if vote_ok:
                    if best_name not in confirm_start:
                        confirm_start[best_name] = time.time()
                    elapsed  = time.time() - confirm_start[best_name]
                    progress = min(elapsed / CONFIRM_SECONDS, 1.0)
                else:
                    confirm_start.pop(best_name, None)
                    elapsed  = 0
                    progress = 0

                already_confirmed = best_name in confirmed_set

                # ── Visual state ──────────────────────────
                if already_confirmed:
                    color    = (0, 200, 0)
                    label    = f"{best_name} ✓ ({best_score:.0f}%)"
                    progress = 1.0
                elif vote_ok and elapsed >= CONFIRM_SECONDS:
                    # ── CONFIRMED! Mark attendance ────────
                    confirmed_set.add(best_name)
                    marked_today.add(best_name)
                    elapsed_str = f"{elapsed:.1f}"
                    append_attendance_row(
                        ATTENDANCE_FILE, sheet_name,
                        best_name, "PRESENT", f"{best_score:.0f}%", elapsed_str
                    )
                    print(f"✅ {best_name} — PRESENT at "
                          f"{now.strftime('%H:%M:%S')} "
                          f"({best_score:.0f}% | sim={best_sim:.3f}) "
                          f"after {elapsed_str}s")
                    color    = (0, 200, 0)
                    label    = f"{best_name} ✓ ({best_score:.0f}%)"
                    progress = 1.0
                elif vote_ok:
                    color = (0, 200, 180)
                    label = f"{best_name} ({best_score:.0f}%) — hold..."
                else:
                    color = (0, 160, 100)
                    label = f"{best_name}? ({best_score:.0f}%)"

            else:
                # ── No hard match this frame ──────────────
                # Check if any person has a strong partial vote buffer
                soft_name  = None
                soft_votes = 0
                for pname, buf in vote_buffer.items():
                    v = sum(buf)
                    if v > soft_votes and v >= VOTE_REQUIRE - 2:
                        soft_votes = v
                        soft_name  = pname

                if soft_name and soft_name not in confirmed_set:
                    color    = (0, 140, 80)
                    label    = f"{soft_name}? (weak frame)"
                    progress = 0
                    seen_this_frame.add(soft_name)
                elif soft_name and soft_name in confirmed_set:
                    color    = (0, 200, 0)
                    label    = f"{soft_name} ✓"
                    progress = 1.0
                    seen_this_frame.add(soft_name)
                else:
                    # Genuinely unknown
                    color    = (0, 0, 200)
                    label    = "UNKNOWN"
                    loc_key  = f"{x1 // 30}_{y1 // 30}"
                    last_log = unknown_last_logged.get(loc_key)
                    if (last_log is None or
                            (now - last_log).seconds > UNKNOWN_COOLDOWN):
                        unknown_count += 1
                        unknown_last_logged[loc_key] = now
                        print(f"🚨 Unknown at {now.strftime('%H:%M:%S')}")
                progress = 0

            last_locations.append((x1, y1, x2, y2))
            last_labels.append(label)
            last_colors.append(color)
            last_progresses.append(progress if best_name else 0)

        # ── Decay vote buffers for faces not seen this frame ──
        for name_key in list(vote_buffer.keys()):
            if name_key not in seen_this_frame:
                vote_buffer[name_key].append(False)
                still_ok = (
                    len(vote_buffer[name_key]) >= VOTE_WINDOW and
                    sum(vote_buffer[name_key]) >= VOTE_REQUIRE
                )
                if not still_ok:
                    confirm_start.pop(name_key, None)

        # ── Draw face boxes + confirmation bar ────────────
        for i, (x1, y1, x2, y2) in enumerate(last_locations):
            color    = last_colors[i]    if i < len(last_colors)    else (128, 128, 128)
            label    = last_labels[i]    if i < len(last_labels)    else ""
            progress = last_progresses[i] if i < len(last_progresses) else 0

            # Face box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label bar below face
            cv2.rectangle(frame, (x1, y2), (x2, y2 + 28), color, cv2.FILLED)
            cv2.putText(frame, label, (x1 + 4, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Confirmation progress bar
            if 0 < progress < 1.0:
                bar_w  = x2 - x1
                filled = int(bar_w * progress)
                cv2.rectangle(frame, (x1, y2 + 28), (x2, y2 + 36),
                              (30, 30, 30), cv2.FILLED)
                cv2.rectangle(frame, (x1, y2 + 28), (x1 + filled, y2 + 36),
                              (0, 220, 180), cv2.FILLED)
            elif progress >= 1.0:
                cv2.rectangle(frame, (x1, y2 + 28), (x2, y2 + 36),
                              (0, 200, 0), cv2.FILLED)

        # ── HUD bar ───────────────────────────────────────
        hud_w = min(w, 900)
        cv2.rectangle(frame, (0, 0), (hud_w, 65), (25, 25, 25), cv2.FILLED)
        cv2.putText(frame, f"FPS:{fps}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        cv2.putText(frame, f"Present: {len(marked_today)}",
                    (90, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 0), 2)
        cv2.putText(frame, f"Unknown: {unknown_count}",
                    (290, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 80, 255), 2)
        cv2.putText(frame, now.strftime("%H:%M:%S"),
                    (490, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, f"Faces: {len(faces)} | Hold {CONFIRM_SECONDS}s | Q=Cancel | S=Save",
                    (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1)

        cv2.imshow("AXIOM — Attendance System", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), ord('Q')]:
            print("\n❌ Session cancelled. Removing saved data...")
            try:
                wb = load_workbook(ATTENDANCE_FILE)
                if sheet_name in wb.sheetnames:
                    del wb[sheet_name]
                if "Summary" in wb.sheetnames:
                    sw = wb["Summary"]
                    for row in sw.iter_rows(min_row=4):
                        if row[0].value == sheet_name:
                            sw.delete_rows(row[0].row, 1)
                            break
                wb.save(ATTENDANCE_FILE)
            except Exception:
                pass
            cap.release()
            cv2.destroyAllWindows()
            return
        elif key in [ord('s'), ord('S')]:
            break

    cap.release()
    cv2.destroyAllWindows()

    # ── Session summary ───────────────────────────────────
    print("\n" + "=" * 50)
    print(f"✅ Present this session : {len(marked_today)}")
    for pname in marked_today:
        print(f"     {pname}")
    print(f"🚨 Unknown detected     : {unknown_count}")
    print(f"💾 Saved to             : {ATTENDANCE_FILE}  →  sheet '{sheet_name}'")
    print("=" * 50)


# ══════════════════════════════════════════════════════════
# OPTION 3 — VIEW & DELETE REGISTERED PERSONS
# ══════════════════════════════════════════════════════════

def view_registered():
    while True:
        data = load_profiles()

        if not data:
            print("\n❌ No one registered yet!")
            return

        names = list(data.keys())

        print("\n" + "=" * 50)
        print(f"  📋 REGISTERED PERSONS  ({len(names)} total)")
        print("=" * 50)
        for i, name in enumerate(names, start=1):
            info = data[name]
            emb = info.get("embedding")
            emb_status = "✅ 512-dim ArcFace" if emb is not None else "❌ Missing"
            print(f"  {i:>2}. {name}")
            print(f"       Registered : {info.get('registered_on', 'N/A')}")
            print(f"       Samples    : {info.get('total_samples', 'N/A')}")
            print(f"       Embedding  : {emb_status}")
            print()

        print("-" * 50)
        print("  Enter a number to delete that person")
        print("  Press Enter / type 'back' to return")
        print("-" * 50)

        choice = input("  Choice: ").strip().lower()

        if choice == "" or choice == "back":
            return

        if not choice.isdigit():
            print("❌ Enter a number from the list.")
            continue

        idx = int(choice) - 1
        if idx < 0 or idx >= len(names):
            print(f"❌ Enter a number between 1 and {len(names)}.")
            continue

        target = names[idx]
        print(f"\n  ⚠️  Delete '{target}'?")
        print(f"       This removes their face profile permanently.")
        confirm = input("  Type their name to confirm, or press Enter to cancel: ").strip()

        if confirm == target:
            del data[target]
            save_profiles(data)
            print(f"\n  ✅ '{target}' has been deleted.")
        else:
            print("  Cancelled — name didn't match.")


# ══════════════════════════════════════════════════════════
# OPTION 4 — QUICK TEST (verify recognition works)
# ══════════════════════════════════════════════════════════

def quick_test():
    """Quick recognition test — shows similarity scores without marking attendance."""
    data = load_profiles()

    if not data:
        print("\n❌ No profiles registered! Register someone first.")
        return

    app = get_face_app()

    print(f"\n🔬 Quick test mode — {len(data)} profiles loaded")
    print("   Shows raw similarity scores. Press Q to quit.\n")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        print("❌ Cannot open camera!")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        faces = app.get(frame)

        cv2.rectangle(frame, (0, 0), (w, 35), (18, 18, 18), cv2.FILLED)
        cv2.putText(frame, f"Quick Test | {len(faces)} face(s) | Q=Quit",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            live_emb = face.normed_embedding
            if live_emb is None:
                continue

            # Score against all registered
            scores = []
            for pname, person in data.items():
                stored = person.get("embedding")
                if stored is None:
                    continue  # skip old profiles without ArcFace embedding
                if isinstance(stored, list):
                    stored = np.array(stored, dtype=np.float32)
                sim = cosine_similarity(stored, live_emb)
                scores.append((pname, sim))

            scores.sort(key=lambda x: x[1], reverse=True)
            best_name, best_sim = scores[0]

            if best_sim >= MATCH_THRESHOLD:
                color = (0, 200, 0)
                label = f"{best_name} sim={best_sim:.3f}"
            elif best_sim >= MATCH_THRESHOLD - 0.05:
                color = (0, 180, 255)
                label = f"{best_name}? sim={best_sim:.3f}"
            else:
                color = (0, 0, 200)
                label = f"UNKNOWN best={best_sim:.3f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(frame, (x1, y2), (x2, y2 + 24), color, cv2.FILLED)
            cv2.putText(frame, label, (x1 + 4, y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            # Show all scores
            for j, (pname, sim) in enumerate(scores[:5]):
                y_offset = y1 - 20 - j * 18
                if y_offset < 15:
                    continue
                scolor = (0, 200, 0) if sim >= MATCH_THRESHOLD else (100, 100, 100)
                cv2.putText(frame, f"{pname}: {sim:.3f}",
                            (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, scolor, 1)

        cv2.imshow("AXIOM — Quick Test", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), ord('Q')]:
            break

    cap.release()
    cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════
# MAIN MENU
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    while True:
        print("\n" + "=" * 50)
        print("    ╔═══════════════════════════════════╗")
        print("    ║   AXIOM — Face Attendance System  ║")
        print("    ║   Engine: ArcFace Neural Network  ║")
        print("    ╚═══════════════════════════════════╝")
        print("=" * 50)
        print("  1. Register New Person (Live Camera)")
        print("  2. Mark Attendance")
        print("  3. View / Delete Registered Persons")
        print("  4. Quick Test (see similarity scores)")
        print("  5. Exit")
        print("=" * 50)

        choice = input("  Choose option (1-5): ").strip()

        if choice == "1":
            register_live()

        elif choice == "2":
            mark_attendance()

        elif choice == "3":
            view_registered()

        elif choice == "4":
            quick_test()

        elif choice == "5":
            print("\nGoodbye! 👋")
            break

        else:
            print("❌ Invalid choice!")