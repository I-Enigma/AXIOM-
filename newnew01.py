import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import json
import os
import time
from datetime import datetime
from collections import deque
from openpyxl import load_workbook, Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# ── Configuration ─────────────────────────────────────────
PROFILES_FILE    = "face_profiles.json"
ATTENDANCE_FILE  = "attendance.xlsx"
MATCH_THRESHOLD  = 0.06
UNKNOWN_COOLDOWN = 10

# Confirmation: face must be held steady for this many seconds
CONFIRM_SECONDS  = 3.0
# Rolling vote: must match in N of M consecutive frames
VOTE_WINDOW      = 10
VOTE_REQUIRE     = 7

mp_face_mesh = mp.solutions.face_mesh

# ── Ratio weights (stable ratios count more) ───────────────
RATIO_WEIGHTS = {
    "golden_ratio"     : 1.0,
    "eye_ratio"        : 1.4,
    "inner_eye_ratio"  : 1.4,
    "nose_ratio"       : 1.0,
    "lip_ratio"        : 0.8,
    "jaw_ratio"        : 0.9,
    "brow_ratio"       : 0.7,
    "nose_width_ratio" : 1.2,
    "fWHR"             : 0.8,
}
SYMMETRY_WEIGHT = 0.6

# ── Facial Geometry ───────────────────────────────────────

def euclidean(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def get_pt(landmarks, idx, w, h):
    lm = landmarks[idx]
    return (int(lm.x * w), int(lm.y * h))

def check_pose_quality(landmarks, w, h):
    """Returns True only if face is frontal, level, and large enough."""
    def pt(i):
        lm = landmarks[i]
        return (lm.x * w, lm.y * h)

    nose      = pt(1)
    l_eye     = pt(33)
    r_eye     = pt(263)
    chin      = pt(152)
    forehead  = pt(10)

    eye_dist  = abs(r_eye[0] - l_eye[0])
    eye_tilt  = abs(l_eye[1] - r_eye[1])
    if eye_dist == 0 or eye_tilt / eye_dist > 0.15:
        return False

    nose_offset = abs(nose[0] - (l_eye[0] + r_eye[0]) / 2)
    if eye_dist > 0 and nose_offset / eye_dist > 0.2:
        return False

    face_h = abs(chin[1] - forehead[1])
    if face_h < h * 0.18:
        return False

    return True

def extract_profile(landmarks, w, h):
    def pt(i): return get_pt(landmarks, i, w, h)

    forehead    = pt(10)
    chin        = pt(152)
    left_cheek  = pt(234)
    right_cheek = pt(454)
    nose_tip    = pt(1)
    left_eye_o  = pt(33)
    right_eye_o = pt(263)
    left_eye_i  = pt(133)
    right_eye_i = pt(362)
    left_lip    = pt(61)
    right_lip   = pt(291)
    left_jaw    = pt(172)
    right_jaw   = pt(397)
    left_brow   = pt(70)
    right_brow  = pt(300)

    face_height    = euclidean(forehead, chin)
    face_width     = euclidean(left_cheek, right_cheek)
    eye_distance   = euclidean(left_eye_o, right_eye_o)
    inner_eye_dist = euclidean(left_eye_i, right_eye_i)
    nose_to_chin   = euclidean(nose_tip, chin)
    lip_width      = euclidean(left_lip, right_lip)
    jaw_width      = euclidean(left_jaw, right_jaw)
    brow_width     = euclidean(left_brow, right_brow)
    nose_width     = euclidean(pt(129), pt(358))

    if face_width == 0 or face_height == 0:
        return None

    nose_center  = pt(1)
    mirror_pairs = [(33,263),(130,359),(234,454),(61,291),(70,300),(172,397)]
    diffs = []
    for l, r in mirror_pairs:
        lp = pt(l)
        rp = pt(r)
        diffs.append(abs(abs(lp[0]-nose_center[0]) - abs(rp[0]-nose_center[0])))
    symmetry = max(0, 1-(np.mean(diffs)/face_width)) * 100

    return {
        "golden_ratio"     : round(face_height / face_width, 4),
        "eye_ratio"        : round(eye_distance / face_width, 4),
        "inner_eye_ratio"  : round(inner_eye_dist / face_width, 4),
        "nose_ratio"       : round(nose_to_chin / face_height, 4),
        "lip_ratio"        : round(lip_width / face_width, 4),
        "jaw_ratio"        : round(jaw_width / face_width, 4),
        "brow_ratio"       : round(brow_width / face_width, 4),
        "nose_width_ratio" : round(nose_width / face_width, 4),
        "fWHR"             : round(face_width / (face_height/2), 4),
        "symmetry"         : round(symmetry, 2),
    }

def build_robust_profile(profiles):
    """Median-based profile — outlier frames can't skew it."""
    keys = list(profiles[0].keys())
    avg_profile = {}
    std_profile = {}
    for key in keys:
        values = [p[key] for p in profiles]
        avg_profile[key] = round(float(np.median(values)), 4)
        std_profile[key] = round(float(np.std(values)), 4)
    return avg_profile, std_profile

def compare_profiles(stored, live, stored_std=None):
    weighted_diffs = []
    total_weight   = 0.0

    for key, weight in RATIO_WEIGHTS.items():
        ref = stored.get(key, 0)
        if ref == 0:
            continue
        raw_diff = abs(ref - live[key]) / ref

        if stored_std and stored_std.get(key, 0) > 0:
            tolerance_factor = 1 + (stored_std[key] / ref) * 3
            raw_diff /= tolerance_factor

        weighted_diffs.append(raw_diff * weight)
        total_weight += weight

    sym_stored = stored.get("symmetry", 50)
    sym_live   = live.get("symmetry", 50)
    if sym_stored > 0:
        sym_diff = abs(sym_stored - sym_live) / 100
        weighted_diffs.append(sym_diff * SYMMETRY_WEIGHT)
        total_weight += SYMMETRY_WEIGHT

    if total_weight == 0:
        return 0, False

    avg_diff    = sum(weighted_diffs) / total_weight
    match_score = round(max(0, 1 - avg_diff) * 100, 2)
    is_match    = avg_diff < MATCH_THRESHOLD
    return match_score, is_match

# ── Profile Storage ───────────────────────────────────────

def load_profiles():
    if os.path.exists(PROFILES_FILE):
        with open(PROFILES_FILE, "r") as f:
            return json.load(f)
    return {}

def save_profiles(data):
    with open(PROFILES_FILE, "w") as f:
        json.dump(data, f, indent=4)

# ── Excel Attendance (per-session sheets) ─────────────────

SESSION_SHEET_PREFIX = "Session_"

def get_session_sheet_name():
    """Each run gets its own sheet: Session_2025-07-14_143022"""
    return SESSION_SHEET_PREFIX + datetime.now().strftime("%Y-%m-%d_%H%M%S")

def create_attendance_workbook(filepath):
    wb = Workbook()
    ws = wb.active
    ws.title = "Summary"
    ws["A1"] = "Face Attendance System"
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

    # Header row styling
    headers = ["Name", "Date", "Time", "Status", "Match Score", "Confirmed After (s)"]
    header_fill   = PatternFill("solid", fgColor="1F4E79")
    header_font   = Font(bold=True, color="FFFFFF", name="Arial", size=11)
    header_align  = Alignment(horizontal="center", vertical="center")
    thin_border   = Border(
        bottom=Side(style="thin", color="AAAAAA"),
        right=Side(style="thin",  color="DDDDDD"),
    )

    for col_idx, header in enumerate(headers, start=1):
        cell = ws.cell(row=1, column=col_idx, value=header)
        cell.font    = header_font
        cell.fill    = header_fill
        cell.alignment = header_align
        cell.border  = thin_border

    ws.row_dimensions[1].height = 22
    col_widths = [20, 14, 12, 12, 14, 20]
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

            # Alternating row background
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
                # Find or create the row for this session
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

# ── Angle detection helpers ───────────────────────────────

# Each zone: (short_label, instruction_line1, instruction_line2, yaw_min, yaw_max, pitch_min, pitch_max)
# yaw > 0 = turned right, < 0 = turned left
# pitch > 0 = chin up,    < 0 = chin down
CAPTURE_ZONES = [
    ("Front",        "Face the camera",        "look straight ahead",  -0.08,  0.08, -0.08,  0.08),
    ("Left",         "Turn your head",          "slightly to the LEFT", -0.22, -0.08, -0.08,  0.08),
    ("Right",        "Turn your head",          "slightly to the RIGHT", 0.08,  0.22, -0.08,  0.08),
    ("Up",           "Tilt your chin",          "slightly UPWARD",      -0.08,  0.08,  0.08,  0.22),
    ("Down",         "Tilt your chin",          "slightly DOWNWARD",    -0.08,  0.08, -0.22, -0.08),
    ("Left+Up",      "Turn LEFT and",           "tilt chin UP",         -0.22, -0.08,  0.08,  0.22),
    ("Right+Up",     "Turn RIGHT and",          "tilt chin UP",          0.08,  0.22,  0.08,  0.22),
    ("Left+Down",    "Turn LEFT and",           "tilt chin DOWN",       -0.22, -0.08, -0.22, -0.08),
]
SAMPLES_PER_ZONE = 5   # 8 zones × 5 = 40 total


def get_face_angles(landmarks, w, h):
    """Returns (yaw, pitch) as normalised ratios."""
    def pt(i):
        lm = landmarks[i]
        return (lm.x * w, lm.y * h)

    nose     = pt(1)
    l_eye    = pt(33)
    r_eye    = pt(263)
    chin     = pt(152)
    forehead = pt(10)

    eye_mid  = ((l_eye[0] + r_eye[0]) / 2, (l_eye[1] + r_eye[1]) / 2)
    eye_dist = abs(r_eye[0] - l_eye[0])
    face_h   = abs(chin[1] - forehead[1])

    yaw   = (nose[0] - eye_mid[0]) / eye_dist if eye_dist > 0 else 0
    pitch = (nose[1] - eye_mid[1]) / face_h   if face_h  > 0 else 0

    return round(yaw, 3), round(pitch, 3)


def classify_zone(yaw, pitch):
    """Return index of matching zone, or None."""
    for i, (_, _l1, _l2, y0, y1, p0, p1) in enumerate(CAPTURE_ZONES):
        if y0 <= yaw <= y1 and p0 <= pitch <= p1:
            return i
    return None


def draw_registration_hud(frame, name, zone_buckets, current_zone_idx, next_zone_idx):
    """
    Full registration HUD:
      - Top bar:  name + live capture status
      - Middle:   big centred "NEXT ANGLE" instruction (only when needed)
      - Bottom strip: 8 zone pills in a single row
      - Very bottom: key hints
    """
    h, w = frame.shape[:2]
    zone_counts   = [len(b) for b in zone_buckets]
    total_needed  = len(CAPTURE_ZONES) * SAMPLES_PER_ZONE
    total_done    = sum(zone_counts)
    zones_complete = sum(1 for c in zone_counts if c >= SAMPLES_PER_ZONE)

    # ── TOP BAR (name + progress bar) ────────────────────
    cv2.rectangle(frame, (0, 0), (w, 54), (18, 18, 18), cv2.FILLED)
    cv2.putText(frame, f"Registering: {name}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 215, 60), 2)
    prog_label = f"{total_done}/{total_needed} samples   {zones_complete}/{len(CAPTURE_ZONES)} zones"
    cv2.putText(frame, prog_label,
                (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1)
    # progress bar
    bar_x1, bar_y1, bar_x2, bar_y2 = w - 160, 10, w - 10, 24
    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (50, 50, 50), cv2.FILLED)
    filled = int((bar_x2 - bar_x1) * total_done / total_needed)
    if filled > 0:
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + filled, bar_y2), (0, 200, 100), cv2.FILLED)
    pct_text = f"{int(100 * total_done / total_needed)}%"
    cv2.putText(frame, pct_text, (bar_x1 + 2, bar_y2 - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 255, 220), 1)

    # ── CENTRE INSTRUCTION (next pending zone) ────────────
    # Show only when user is NOT currently in the right zone
    if next_zone_idx is not None and current_zone_idx != next_zone_idx:
        _, line1, line2, *_ = CAPTURE_ZONES[next_zone_idx]
        box_y = h // 2 - 52
        cv2.rectangle(frame, (w//2 - 200, box_y), (w//2 + 200, box_y + 80),
                      (10, 10, 40), cv2.FILLED)
        cv2.rectangle(frame, (w//2 - 200, box_y), (w//2 + 200, box_y + 80),
                      (60, 100, 200), 1)
        cv2.putText(frame, line1,
                    (w//2 - 190, box_y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (200, 220, 255), 2)
        cv2.putText(frame, line2,
                    (w//2 - 190, box_y + 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (100, 200, 255), 2)

    # ── LIVE CAPTURE FLASH ────────────────────────────────
    # When currently capturing the active zone, show a bright flash label
    if current_zone_idx is not None:
        count = zone_counts[current_zone_idx]
        done  = count >= SAMPLES_PER_ZONE
        if done:
            flash_text  = f"  {CAPTURE_ZONES[current_zone_idx][0]} done!"
            flash_color = (0, 230, 80)
        else:
            flash_text  = f"  Capturing {CAPTURE_ZONES[current_zone_idx][0]}  {count}/{SAMPLES_PER_ZONE}"
            flash_color = (0, 210, 160)
        # Solid bar across the top-centre so it's impossible to miss
        flash_y = 58
        cv2.rectangle(frame, (0, flash_y), (w, flash_y + 34), (0, 30, 20), cv2.FILLED)
        cv2.putText(frame, flash_text,
                    (10, flash_y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, flash_color, 2)
    else:
        # No face / no zone — show a plain prompt
        flash_y = 58
        cv2.rectangle(frame, (0, flash_y), (w, flash_y + 34), (30, 10, 10), cv2.FILLED)
        cv2.putText(frame, "  Position your face in the camera",
                    (10, flash_y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (80, 80, 200), 2)

    # ── BOTTOM ZONE PILLS ─────────────────────────────────
    strip_h    = 44
    strip_y    = h - strip_h - 26   # leave room for key hints below
    cv2.rectangle(frame, (0, strip_y - 4), (w, strip_y + strip_h + 4),
                  (18, 18, 18), cv2.FILLED)

    n          = len(CAPTURE_ZONES)
    pill_w     = (w - 16) // n
    pill_gap   = 2

    for i, (short_label, *_) in enumerate(CAPTURE_ZONES):
        count   = zone_counts[i]
        done    = count >= SAMPLES_PER_ZONE
        active  = (i == current_zone_idx)
        is_next = (i == next_zone_idx) and not done

        px = 8 + i * pill_w
        py = strip_y

        # Pill background
        if done:
            bg = (0, 70, 0)
            border = (0, 180, 60)
        elif active:
            bg = (0, 50, 60)
            border = (0, 200, 160)
        elif is_next:
            bg = (40, 30, 0)
            border = (200, 160, 0)
        else:
            bg = (30, 30, 30)
            border = (70, 70, 70)

        cv2.rectangle(frame, (px, py), (px + pill_w - pill_gap, py + strip_h), bg, cv2.FILLED)
        cv2.rectangle(frame, (px, py), (px + pill_w - pill_gap, py + strip_h), border, 1)

        # Zone short label — centred in pill
        label_scale = 0.38
        (lw, lh), _ = cv2.getTextSize(short_label, cv2.FONT_HERSHEY_SIMPLEX, label_scale, 1)
        lx = px + (pill_w - pill_gap - lw) // 2
        if done:
            lbl_color = (100, 255, 120)
            cv2.putText(frame, "OK", (lx + 4, py + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 200, 80), 1)
        elif is_next:
            lbl_color = (255, 200, 60)
        elif active:
            lbl_color = (80, 230, 200)
        else:
            lbl_color = (100, 100, 100)

        cv2.putText(frame, short_label, (lx, py + 16 if done else py + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, label_scale, lbl_color, 1)

        # Mini count bar at bottom of pill
        bar_py = py + strip_h - 7
        cv2.rectangle(frame, (px + 2, bar_py), (px + pill_w - pill_gap - 2, bar_py + 5),
                      (50, 50, 50), cv2.FILLED)
        filled_w = int((pill_w - pill_gap - 4) * min(count / SAMPLES_PER_ZONE, 1.0))
        bar_col  = (0, 180, 60) if done else ((0, 180, 160) if active else (100, 100, 40))
        if filled_w > 0:
            cv2.rectangle(frame, (px + 2, bar_py), (px + 2 + filled_w, bar_py + 5),
                          bar_col, cv2.FILLED)

    # ── KEY HINTS ─────────────────────────────────────────
    hint_y = h - 10
    cv2.putText(frame, "S = save now   |   Q = cancel",
                (w // 2 - 130, hint_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (90, 90, 90), 1)


# ══════════════════════════════════════════════════════════
# OPTION 1 — REGISTER NEW PERSON
# ══════════════════════════════════════════════════════════

def register_live():
    print("\n" + "="*50)
    print("          REGISTER NEW PERSON")
    print("="*50)

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

    total_needed = len(CAPTURE_ZONES) * SAMPLES_PER_ZONE
    print(f"\n👤 Registering: {name}")
    print(f"📸 The camera will guide you — follow the on-screen angle prompts.")
    print(f"   {len(CAPTURE_ZONES)} zones × {SAMPLES_PER_ZONE} samples = {total_needed} total.")
    print("   Saves automatically when all zones are done.  S = save early.  Q = cancel.\n")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("❌ Cannot open camera!")
        return

    zone_buckets = [[] for _ in CAPTURE_ZONES]

    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results  = face_mesh.process(rgb)

            zone_counts      = [len(b) for b in zone_buckets]
            current_zone_idx = None

            # Find the first incomplete zone to prompt next
            next_zone_idx = next(
                (i for i, c in enumerate(zone_counts) if c < SAMPLES_PER_ZONE),
                None
            )

            if results.multi_face_landmarks:
                lms        = results.multi_face_landmarks[0].landmark
                profile    = extract_profile(lms, w, h)
                yaw, pitch = get_face_angles(lms, w, h)
                zone_idx   = classify_zone(yaw, pitch)

                if profile:
                    # Draw subtle green mesh
                    for lm in lms:
                        cv2.circle(frame,
                                   (int(lm.x * w), int(lm.y * h)),
                                   1, (0, 180, 80), -1)

                    if zone_idx is not None:
                        current_zone_idx = zone_idx
                        if zone_counts[zone_idx] < SAMPLES_PER_ZONE:
                            zone_buckets[zone_idx].append(profile)

            draw_registration_hud(frame, name, zone_buckets, current_zone_idx, next_zone_idx)

            cv2.imshow("Face Registration", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Registration cancelled.")
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord('s'):
                all_so_far = [p for b in zone_buckets for p in b]
                if len(all_so_far) < SAMPLES_PER_ZONE:
                    print(f"⚠️  Only {len(all_so_far)} samples — need at least {SAMPLES_PER_ZONE} to save.")
                else:
                    print(f"\n💾 Saving with {len(all_so_far)} samples...")
                    break

            # Auto-save when every zone is complete
            if all(len(b) >= SAMPLES_PER_ZONE for b in zone_buckets):
                print("\n✅ All zones captured! Saving...")
                time.sleep(0.4)
                break

    cap.release()
    cv2.destroyAllWindows()

    all_profiles = [p for bucket in zone_buckets for p in bucket]

    if len(all_profiles) == 0:
        print("❌ No samples collected.")
        return

    avg_profile, std_profile = build_robust_profile(all_profiles)

    data[name] = {
        "name"          : name,
        "profile"       : avg_profile,
        "std"           : std_profile,
        "total_samples" : len(all_profiles),
        "zones_captured": {CAPTURE_ZONES[i][0]: len(zone_buckets[i])
                           for i in range(len(CAPTURE_ZONES))},
        "registered_on" : datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    save_profiles(data)

    print(f"\n✅ {name} registered successfully!")
    print(f"   Total samples : {len(all_profiles)}")
    print(f"   Golden Ratio  : {avg_profile['golden_ratio']}")
    print(f"   Symmetry      : {avg_profile['symmetry']}%")
    zones_done = sum(1 for b in zone_buckets if len(b) >= SAMPLES_PER_ZONE)
    print(f"   Zones filled  : {zones_done}/{len(CAPTURE_ZONES)}")

# ══════════════════════════════════════════════════════════
# OPTION 2 — MARK ATTENDANCE (with confirmation timer)
# ══════════════════════════════════════════════════════════

def mark_attendance():
    data = load_profiles()

    if not data:
        print("\n❌ No profiles registered yet!")
        return

    print(f"\n📋 {len(data)} registered persons loaded")
    print(f"⏱  A face must be held for {CONFIRM_SECONDS}s to be marked")
    print("Controls: Q = Quit | S = Show summary")
    print("="*50)

    # Setup Excel session sheet
    sheet_name = get_session_sheet_name()
    init_session_sheet(ATTENDANCE_FILE, sheet_name)
    print(f"📄 Logging to sheet: {sheet_name}")

    today = datetime.now().strftime("%Y-%m-%d")
    marked_today = set()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("❌ Cannot open camera!")
        return

    # Per-face tracking state
    # vote_buffer[name]    = deque of recent bool matches
    # confirm_start[name]  = time when continuous match streak started
    # confirmed[name]      = True once attendance has been logged
    vote_buffer    = {}
    confirm_start  = {}
    confirmed_set  = set()

    unknown_count       = 0
    unknown_last_logged = {}

    frame_count   = 0
    fps           = 0
    fps_timer     = time.time()
    frame_counter = 0

    last_locations = []
    last_labels    = []
    last_colors    = []
    last_progresses = []

    with mp_face_mesh.FaceMesh(
            max_num_faces=3,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count   += 1
            frame_counter += 1
            now            = datetime.now()

            if time.time() - fps_timer >= 1.0:
                fps           = frame_counter
                frame_counter = 0
                fps_timer     = time.time()

            if frame_count % 2 == 0:
                rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)

                last_locations  = []
                last_labels     = []
                last_colors     = []
                last_progresses = []

                seen_this_frame = set()

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        lms          = face_landmarks.landmark
                        live_profile = extract_profile(lms, w := frame.shape[1], h := frame.shape[0])
                        if not live_profile:
                            continue

                        xs = [int(lm.x*w) for lm in lms]
                        ys = [int(lm.y*h) for lm in lms]
                        x1 = max(0, min(xs)-10)
                        x2 = min(w, max(xs)+10)
                        y1 = max(0, min(ys)-10)
                        y2 = min(h, max(ys)+10)

                        # Match against all profiles
                        best_name  = None
                        best_score = 0

                        for name, person in data.items():
                            score, is_match = compare_profiles(
                                person["profile"],
                                live_profile,
                                person.get("std")
                            )
                            if is_match and score > best_score:
                                best_score = score
                                best_name  = name

                        if best_name:
                            seen_this_frame.add(best_name)

                            # Rolling vote buffer
                            if best_name not in vote_buffer:
                                vote_buffer[best_name] = deque(maxlen=VOTE_WINDOW)
                            vote_buffer[best_name].append(True)

                            vote_ok = (
                                len(vote_buffer[best_name]) >= VOTE_WINDOW and
                                sum(vote_buffer[best_name]) >= VOTE_REQUIRE
                            )

                            # Confirmation timer — only starts when vote is stable
                            if vote_ok:
                                if best_name not in confirm_start:
                                    confirm_start[best_name] = time.time()
                                elapsed  = time.time() - confirm_start[best_name]
                                progress = min(elapsed / CONFIRM_SECONDS, 1.0)
                            else:
                                # Reset timer if vote drops
                                confirm_start.pop(best_name, None)
                                elapsed  = 0
                                progress = 0

                            already_confirmed = best_name in confirmed_set

                            if already_confirmed:
                                color = (0, 200, 0)
                                label = f"{best_name} ✓ ({best_score}%)"
                                progress = 1.0
                            elif vote_ok and elapsed >= CONFIRM_SECONDS:
                                # ── CONFIRMED ────────────────────────────────
                                confirmed_set.add(best_name)
                                marked_today.add(best_name)
                                elapsed_str = f"{elapsed:.1f}"
                                append_attendance_row(
                                    ATTENDANCE_FILE, sheet_name,
                                    best_name, "PRESENT", f"{best_score}%", elapsed_str
                                )
                                print(f"✅ {best_name} — PRESENT at "
                                      f"{now.strftime('%H:%M:%S')} "
                                      f"({best_score}%) after {elapsed_str}s")
                                color    = (0, 200, 0)
                                label    = f"{best_name} ✓ ({best_score}%)"
                                progress = 1.0
                            elif vote_ok:
                                # Timing — show progress bar filling up
                                color = (0, 200, 180)
                                label = f"{best_name} ({best_score}%) — hold..."
                            else:
                                # Vote not stable yet
                                color = (0, 160, 100)
                                label = f"{best_name}? ({best_score}%)"

                        else:
                            # Unknown face
                            color    = (0, 0, 220)
                            progress = 0

                            # Reset any partial match state (face left the frame)
                            for k in list(vote_buffer.keys()):
                                if k not in seen_this_frame:
                                    vote_buffer[k].clear()
                                    confirm_start.pop(k, None)

                            loc_key  = f"{x1//30}_{y1//30}"
                            last_log = unknown_last_logged.get(loc_key)

                            if (last_log is None or
                                    (now - last_log).seconds > UNKNOWN_COOLDOWN):
                                unknown_count += 1
                                append_attendance_row(
                                    ATTENDANCE_FILE, sheet_name,
                                    f"UNKNOWN #{unknown_count}", "UNKNOWN", "N/A"
                                )
                                unknown_last_logged[loc_key] = now
                                print(f"🚨 Unknown at {now.strftime('%H:%M:%S')}")

                            label = "UNKNOWN"

                        last_locations.append((x1, y1, x2, y2))
                        last_labels.append(label)
                        last_colors.append(color)
                        last_progresses.append(progress if best_name else 0)

                # Reset vote buffers for faces not seen this frame
                for name_key in list(vote_buffer.keys()):
                    if name_key not in seen_this_frame:
                        vote_buffer[name_key].clear()
                        confirm_start.pop(name_key, None)

            # ── Draw face boxes + confirmation bar ────────
            h, w, _ = frame.shape
            for i, (x1, y1, x2, y2) in enumerate(last_locations):
                color    = last_colors[i]    if i < len(last_colors)    else (128,128,128)
                label    = last_labels[i]    if i < len(last_labels)    else ""
                progress = last_progresses[i] if i < len(last_progresses) else 0

                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

                # Label bar
                cv2.rectangle(frame, (x1, y2), (x2, y2+28), color, cv2.FILLED)
                cv2.putText(frame, label, (x1+4, y2+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

                # Confirmation progress bar (below label)
                if progress > 0 and progress < 1.0:
                    bar_w = x2 - x1
                    filled = int(bar_w * progress)
                    cv2.rectangle(frame, (x1, y2+28), (x2, y2+36), (30,30,30), cv2.FILLED)
                    cv2.rectangle(frame, (x1, y2+28), (x1+filled, y2+36), (0,220,180), cv2.FILLED)
                elif progress >= 1.0:
                    bar_w = x2 - x1
                    cv2.rectangle(frame, (x1, y2+28), (x2, y2+36), (0,200,0), cv2.FILLED)

            # ── HUD bar ───────────────────────────────────
            cv2.rectangle(frame, (0,0), (640,65), (25,25,25), cv2.FILLED)
            cv2.putText(frame, f"FPS:{fps}",
                        (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150,150,150), 1)
            cv2.putText(frame, f"Present: {len(marked_today)}",
                        (80,25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,220,0), 2)
            cv2.putText(frame, f"Unknown: {unknown_count}",
                        (270,25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,80,255), 2)
            cv2.putText(frame, now.strftime("%H:%M:%S"),
                        (490,25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
            cv2.putText(frame, f"Hold {CONFIRM_SECONDS}s to confirm | Q=Quit | S=Summary",
                        (10,52), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120,120,120), 1)

            cv2.imshow("Attendance System", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                print("\n" + "="*40)
                print(f"  Session: {sheet_name}")
                for name in marked_today:
                    print(f"  ✅ {name}")
                print(f"  🚨 Unknowns: {unknown_count}")
                print("="*40 + "\n")

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "="*50)
    print(f"✅ Present this session : {len(marked_today)}")
    for name in marked_today:
        print(f"     {name}")
    print(f"🚨 Unknown detected     : {unknown_count}")
    print(f"💾 Saved to             : {ATTENDANCE_FILE}  →  sheet '{sheet_name}'")
    print("="*50)

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

        print("\n" + "="*50)
        print(f"  📋 REGISTERED PERSONS  ({len(names)} total)")
        print("="*50)
        for i, name in enumerate(names, start=1):
            info = data[name]
            zones_info = ""
            if "zones_captured" in info:
                zones_done = sum(1 for v in info["zones_captured"].values() if v >= SAMPLES_PER_ZONE)
                zones_info = f"  [{zones_done}/{len(CAPTURE_ZONES)} zones]"
            print(f"  {i:>2}. {name}")
            print(f"       Registered : {info['registered_on']}")
            print(f"       Samples    : {info['total_samples']}{zones_info}")
            print(f"       Symmetry   : {info['profile']['symmetry']}%  |  "
                  f"Golden Ratio: {info['profile']['golden_ratio']}")
            print()

        print("-"*50)
        print("  Enter a number to delete that person")
        print("  Press Enter / type 'back' to return")
        print("-"*50)

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
# MAIN MENU
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    while True:
        print("\n" + "="*50)
        print("       FACE ATTENDANCE SYSTEM")
        print("    (Facial Geometry & Symmetry)")
        print("="*50)
        print("1. Register New Person (Live Camera)")
        print("2. Mark Attendance")
        print("3. View Registered Persons")
        print("4. Exit")
        print("="*50)

        choice = input("Choose option (1/2/3/4): ").strip()

        if choice == "1":
            register_live()

        elif choice == "2":
            mark_attendance()

        elif choice == "3":
            view_registered()

        elif choice == "4":
            print("\nGoodbye! 👋")
            break

        else:
            print("❌ Invalid choice!")