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
MATCH_THRESHOLD         = 0.14   # maximum slack for extreme corners like Left+Up
MATCH_THRESHOLD_FRONTAL = 0.04   # frontal — MUCH tighter to avoid false positives
UNKNOWN_COOLDOWN = 10

# Confirmation: face must be held steady for this many seconds
CONFIRM_SECONDS  = 3.0
# Rolling vote: must match in N of M consecutive frames
VOTE_WINDOW      = 10
VOTE_REQUIRE     = 7

mp_face_mesh = mp.solutions.face_mesh

# ── Ratio weights (stable ratios count more) ───────────────
# Used for frontal faces
RATIO_WEIGHTS = {
    "golden_ratio"      : 1.0,
    "eye_ratio"         : 1.4,
    "inner_eye_ratio"   : 1.4,
    "nose_ratio"        : 1.0,
    "lip_ratio"         : 0.8,
    "jaw_ratio"         : 0.9,
    "brow_ratio"        : 0.7,
    "nose_width_ratio"  : 1.2,
    "fWHR"              : 0.8,
    "eye_height_ratio"  : 1.1,
    "philtrum_ratio"    : 0.9,
    "chin_width_ratio"  : 1.0,
    "cheekbone_ratio"   : 1.0,
    "temple_ratio"      : 0.8,
    "nose_bridge_ratio" : 1.1,
    "mouth_height_ratio": 0.7,
    "nose_length_ratio" : 1.0,
    "nose_base_ratio"   : 1.1,
}

# Used for tilted faces — heavily down-weights ratios that shift with head turn.
# face_width, jaw_ratio, brow_ratio, cheekbone_ratio, fWHR all change when
# the head rotates because one side of the face becomes foreshortened.
# Ratios that stay stable: eye_height, nose_length, philtrum, inner eye distance.
TILTED_RATIO_WEIGHTS = {
    "golden_ratio"      : 0.5,   # face_width changes on turn — unreliable
    "eye_ratio"         : 0.8,   # outer eye points shift on turn
    "inner_eye_ratio"   : 1.4,   # inner corners more stable
    "nose_ratio"        : 1.2,   # nose-to-chin mostly stable
    "lip_ratio"         : 0.5,   # lip width foreshortens
    "jaw_ratio"         : 0.4,   # jaw foreshortens heavily
    "brow_ratio"        : 0.4,   # brow width foreshortens
    "nose_width_ratio"  : 0.9,   # nostril width reasonably stable
    "fWHR"              : 0.3,   # very pose-sensitive — nearly ignore
    "eye_height_ratio"  : 1.3,   # eye height doesn't change with yaw
    "philtrum_ratio"    : 1.2,   # philtrum length stable
    "chin_width_ratio"  : 0.5,   # foreshortens
    "cheekbone_ratio"   : 0.4,   # heavily affected by yaw
    "temple_ratio"      : 0.4,   # foreshortens
    "nose_bridge_ratio" : 1.0,   # bridge width stable
    "mouth_height_ratio": 0.7,
    "nose_length_ratio" : 1.3,   # nose length very stable across yaw
    "nose_base_ratio"   : 1.1,
}
SYMMETRY_WEIGHT = 0.8

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
    """
    Extracts 14 facial geometry ratios + symmetry score.
    All measurements are normalised by face_width or face_height
    so they are scale- and distance-independent.

    New landmarks added (Fix #4):
      - eye_height_ratio : how tall each eye is — very person-specific
      - philtrum_ratio   : nose-tip to upper lip, normalised by face height
      - chin_width_ratio : narrow vs wide chin shape
      - cheekbone_ratio  : cheekbone prominence relative to face width
      - temple_ratio     : forehead width at temple level
    """
    def pt(i): return get_pt(landmarks, i, w, h)

    forehead     = pt(10)
    chin         = pt(152)
    left_cheek   = pt(234)
    right_cheek  = pt(454)
    nose_tip     = pt(1)
    left_eye_o   = pt(33)
    right_eye_o  = pt(263)
    left_eye_i   = pt(133)
    right_eye_i  = pt(362)
    left_lip     = pt(61)
    right_lip    = pt(291)
    left_jaw     = pt(172)
    right_jaw    = pt(397)
    left_brow    = pt(70)
    right_brow   = pt(300)
    # Existing extra landmarks
    upper_lip    = pt(13)
    left_eye_top = pt(159)
    left_eye_bot = pt(145)
    left_temple  = pt(162)
    right_temple = pt(389)
    left_cheekb  = pt(116)
    right_cheekb = pt(345)
    chin_left    = pt(136)
    chin_right   = pt(365)
    # New landmarks (issue 4)
    nose_bridge  = pt(6)    # top of nose bridge — stable bone point
    left_nostril = pt(129)
    right_nostril= pt(358)
    lower_lip    = pt(17)   # bottom of lower lip
    mouth_top    = pt(0)    # top of mouth opening
    mouth_bot    = pt(17)   # bottom of mouth opening (same as lower_lip, used for height)
    left_brow_i  = pt(107)  # inner brow left
    right_brow_i = pt(336)  # inner brow right
    nose_base    = pt(94)   # base of nose between nostrils

    face_height    = euclidean(forehead, chin)
    face_width     = euclidean(left_cheek, right_cheek)

    if face_width == 0 or face_height == 0:
        return None

    eye_distance   = euclidean(left_eye_o, right_eye_o)
    inner_eye_dist = euclidean(left_eye_i, right_eye_i)
    nose_to_chin   = euclidean(nose_tip, chin)
    lip_width      = euclidean(left_lip, right_lip)
    jaw_width      = euclidean(left_jaw, right_jaw)
    brow_width     = euclidean(left_brow, right_brow)
    nose_width     = euclidean(pt(129), pt(358))
    # New measurements (issue 4)
    eye_height      = euclidean(left_eye_top, left_eye_bot)
    philtrum        = euclidean(nose_tip, upper_lip)
    chin_width      = euclidean(chin_left, chin_right)
    cheekbone_w     = euclidean(left_cheekb, right_cheekb)
    temple_w        = euclidean(left_temple, right_temple)
    nose_bridge_w   = euclidean(left_brow_i, right_brow_i)   # brow-to-brow gap (proxies bridge)
    mouth_height    = euclidean(upper_lip, lower_lip)         # how open/tall the mouth region is
    nose_length     = euclidean(nose_bridge, nose_tip)        # nose bridge to tip length
    nose_base_w     = euclidean(left_nostril, right_nostril)  # nostril width
    brow_inner_gap  = euclidean(left_brow_i, right_brow_i)    # inner brow separation

    # Symmetry: how evenly balanced left/right landmarks are around nose centre
    nose_center  = pt(1)
    mirror_pairs = [(33,263),(130,359),(234,454),(61,291),(70,300),(172,397),
                    (116,345),(162,389),(136,365),(107,336),(129,358)]
    diffs = []
    for l, r in mirror_pairs:
        lp = pt(l)
        rp = pt(r)
        diffs.append(abs(abs(lp[0]-nose_center[0]) - abs(rp[0]-nose_center[0])))
    symmetry = max(0, 1 - (np.mean(diffs) / face_width)) * 100

    return {
        # Original ratios
        "golden_ratio"      : round(face_height / face_width, 4),
        "eye_ratio"         : round(eye_distance / face_width, 4),
        "inner_eye_ratio"   : round(inner_eye_dist / face_width, 4),
        "nose_ratio"        : round(nose_to_chin / face_height, 4),
        "lip_ratio"         : round(lip_width / face_width, 4),
        "jaw_ratio"         : round(jaw_width / face_width, 4),
        "brow_ratio"        : round(brow_width / face_width, 4),
        "nose_width_ratio"  : round(nose_width / face_width, 4),
        "fWHR"              : round(face_width / (face_height / 2), 4),
        # Previous extra ratios
        "eye_height_ratio"  : round(eye_height / face_height, 4),
        "philtrum_ratio"    : round(philtrum / face_height, 4),
        "chin_width_ratio"  : round(chin_width / face_width, 4),
        "cheekbone_ratio"   : round(cheekbone_w / face_width, 4),
        "temple_ratio"      : round(temple_w / face_width, 4),
        # New ratios (issue 4)
        "nose_bridge_ratio" : round(nose_bridge_w / face_width, 4),
        "mouth_height_ratio": round(mouth_height / face_height, 4),
        "nose_length_ratio" : round(nose_length / face_height, 4),
        "nose_base_ratio"   : round(nose_base_w / face_width, 4),
        # Symmetry
        "symmetry"          : round(symmetry, 2),
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

def _nearest_zone_profiles(stored_person, live_yaw, live_pitch):
    """
    Instead of only using the exact matched zone, collect up to 3 nearest
    registered zone profiles sorted by angular distance from the live pose.
    This means even if the face is between zones, we always have something
    meaningful to compare against.
    Returns list of (distance, zone_label, profile_dict, std_dict).
    """
    candidates = []

    zone_profiles = stored_person.get("zone_profiles", {})

    for i, (label, _l1, _l2, y0, y1, p0, p1) in enumerate(CAPTURE_ZONES):
        # Zone centre
        cy = (y0 + y1) / 2
        cp = (p0 + p1) / 2
        dist = (live_yaw - cy) ** 2 + (live_pitch - cp) ** 2

        if label in zone_profiles:
            zp  = zone_profiles[label]["profile"]
            zstd = zone_profiles[label].get("std")
        else:
            # Zone not in per-zone profiles — use global as fallback
            zp   = stored_person["profile"]
            zstd = stored_person.get("std")

        candidates.append((dist, label, zp, zstd))

    # Sort by distance — closest zone first
    candidates.sort(key=lambda x: x[0])
    return candidates[:3]   # top 3 nearest zones


def _ratio_score(stored, live, std_dev, weights):
    """
    Compute weighted average difference between stored and live profile.
    Returns (avg_diff, total_weight).
    """
    weighted_diffs = []
    total_weight   = 0.0

    for key, weight in weights.items():
        ref = stored.get(key, 0)
        liv = live.get(key, 0)
        if ref == 0 or liv == 0:
            continue
        raw_diff = abs(ref - liv) / ref

        # Adaptive tolerance: if this ratio was variable during registration,
        # give more slack (up to 4× tolerance)
        if std_dev and std_dev.get(key, 0) > 0:
            tol = 1 + (std_dev[key] / ref) * 4
            raw_diff /= tol

        weighted_diffs.append(raw_diff * weight)
        total_weight += weight

    # Symmetry
    sym_ref = stored.get("symmetry", 50)
    sym_liv = live.get("symmetry", 50)
    if sym_ref > 0:
        sym_diff = abs(sym_ref - sym_liv) / 100.0
        weighted_diffs.append(sym_diff * SYMMETRY_WEIGHT)
        total_weight += SYMMETRY_WEIGHT

    if total_weight == 0:
        return 1.0, 0.0

    return sum(weighted_diffs) / total_weight, total_weight


def compare_profiles(stored_person, live_profile, live_yaw=None, live_pitch=None):
    """
    Robust multi-zone matching strategy:

    1.  Find the 3 nearest registered zone profiles by angular distance —
        not just the exact zone the face falls in.  This eliminates UNKNOWN
        flashes when the head is between zone boundaries.

    2.  Score against each of the 3 candidates, weighting by:
          a. How close the zone is angularly  (closer = higher weight)
          b. Whether the face is frontal or tilted (controls which ratio
             weights to use — tilted ignores foreshortening-sensitive ratios)

    3.  Take a distance-weighted blend of the 3 scores as the final score.
        This smooths out the sharp boundary effect where crossing a zone
        line by 1 pixel used to flip from 92% match to UNKNOWN.

    4.  Dynamic threshold:
          - Frontal     : 0.06  (tight)
          - Slight tilt : 0.09  (medium)
          - Heavy tilt  : 0.12  (loose — many ratios foreshorten)
        Threshold scales continuously with |yaw| so there's no cliff edge.
    """
    yaw   = live_yaw   if live_yaw   is not None else 0.0
    pitch = live_pitch if live_pitch is not None else 0.0

    # How tilted is the face right now?
    tilt_magnitude = (yaw ** 2 + pitch ** 2) ** 0.5

    # Threshold scales smoothly from frontal threshold to tilted threshold
    # The 0.4 multiplier ensures extreme tilts reach the max MATCH_THRESHOLD
    threshold = MATCH_THRESHOLD_FRONTAL + min(tilt_magnitude * 0.4, MATCH_THRESHOLD - MATCH_THRESHOLD_FRONTAL)

    # Choose weights based on tilt
    weights = RATIO_WEIGHTS if tilt_magnitude < 0.12 else TILTED_RATIO_WEIGHTS

    # Get up to 3 nearest zone profiles
    candidates = _nearest_zone_profiles(stored_person, yaw, pitch)

    # Score each candidate and blend by inverse angular distance
    blended_diff  = 0.0
    blend_weight  = 0.0

    for angular_dist, zone_label, zprofile, zstd in candidates:
        diff, _ = _ratio_score(zprofile, live_profile, zstd, weights)

        # Weight this candidate inversely by angular distance from the live pose
        # Add small epsilon to avoid division by zero when perfectly on-centre
        inv_dist = 1.0 / (angular_dist + 0.001)
        blended_diff += diff * inv_dist
        blend_weight += inv_dist

    avg_diff    = blended_diff / blend_weight if blend_weight > 0 else 1.0
    match_score = round(max(0.0, 1.0 - avg_diff) * 100, 2)
    is_match    = avg_diff < threshold

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
#
# FIX 1 — Added missing Right+Down zone (was completely absent before).
# FIX 2 — Zones now overlap by 0.03 on each boundary so there are no
#          dead gaps between tiles where no zone triggers.
# FIX 4 — Widened all yaw ranges from ±0.14 to ±0.28 and pitch ranges
#          from ±0.14 to ±0.22 to match real comfortable head movements.
#
# Each zone: (short_label, line1, line2, yaw_min, yaw_max, pitch_min, pitch_max)
# yaw   > 0 = turned right,  < 0 = turned left
# pitch > 0 = chin up,       < 0 = chin down

CAPTURE_ZONES = [
    # label        instruction line 1          line 2
    # yaw > 0 = YOUR right (nose moves right in mirror = left on screen)
    # yaw < 0 = YOUR left
    # Note: yaw is already negated in get_face_angles() so positive = YOUR right
    ("Front",      "Face the camera",          "look straight ahead",    -0.10,  0.10, -0.10,  0.10),
    ("Your Left",  "Turn YOUR head",           "to YOUR LEFT",           -0.38, -0.09, -0.10,  0.10),
    ("Your Right", "Turn YOUR head",           "to YOUR RIGHT",           0.09,  0.38, -0.10,  0.10),
    ("Up",         "Tilt your chin",           "slightly UPWARD",        -0.10,  0.10,  0.09,  0.28),
    ("Down",       "Tilt your chin",           "slightly DOWNWARD",      -0.10,  0.10, -0.28, -0.09),
    ("Left+Up",    "Turn YOUR LEFT and",       "tilt chin UP",           -0.38, -0.09,  0.09,  0.28),
    ("Right+Up",   "Turn YOUR RIGHT and",      "tilt chin UP",            0.09,  0.38,  0.09,  0.28),
    ("Left+Down",  "Turn YOUR LEFT and",       "tilt chin DOWN",         -0.38, -0.09, -0.28, -0.09),
    ("Right+Down", "Turn YOUR RIGHT and",      "tilt chin DOWN",          0.09,  0.38, -0.28, -0.09),
]
SAMPLES_PER_ZONE  = 8    # 9 zones × 8 = 72 total — richer coverage
ZONE_SAMPLE_DELAY = 0.3  # seconds between accepted samples per zone (forces diversity)


def get_face_angles(landmarks, w, h):
    """
    FIX 3 — Completely rewritten pitch calculation.

    OLD (broken):
        pitch = (nose_y - eye_mid_y) / face_height
        Problem: eye_mid is near the vertical centre of the face,
        so this ratio is tiny and changes very little when you tilt.
        When you look down, face_height also shrinks, making it
        even less reliable.

    NEW (fixed):
        yaw   = (nose_x - eye_mid_x) / eye_distance
                  — how far left/right the nose has shifted
                    relative to the eye width. Stable and clear.
        pitch = (chin_y - nose_y) / (chin_y - forehead_y)
                  — how close the chin is to the nose as a fraction
                    of total face height. When you tilt UP your chin
                    moves AWAY from your nose (ratio rises). When you
                    tilt DOWN your chin comes CLOSER (ratio falls).
                    We subtract 0.5 to centre it around zero.

    FIX 5 — Uses chin-to-nose distance normalised by full face height.
    This is stable even when face foreshortens on down-tilt because
    both the numerator and denominator shrink together.
    """
    def pt(i):
        lm = landmarks[i]
        return (lm.x * w, lm.y * h)

    nose     = pt(1)
    l_eye    = pt(33)
    r_eye    = pt(263)
    chin     = pt(152)
    forehead = pt(10)

    eye_mid_x = (l_eye[0] + r_eye[0]) / 2
    eye_dist  = abs(r_eye[0] - l_eye[0])
    face_h    = abs(chin[1] - forehead[1])

    # Yaw: nose horizontal offset from eye centre, scaled by eye width.
    # NEGATED (Fix #3) so that YOUR left turn = negative yaw,
    # YOUR right turn = positive yaw. Without negation the camera
    # mirror would flip left and right from your perspective.
    yaw = -((nose[0] - eye_mid_x) / eye_dist) if eye_dist > 0 else 0

    # Pitch: chin-to-nose vertical gap as fraction of face height, centred at 0
    # Neutral face: chin_y - nose_y ≈ 0.5 * face_h  → pitch ≈ 0.0
    # Chin up:      gap grows → pitch positive
    # Chin down:    gap shrinks → pitch negative
    chin_nose_gap = chin[1] - nose[1]
    pitch = ((chin_nose_gap / face_h) - 0.5) if face_h > 0 else 0

    return round(yaw, 3), round(pitch, 3)


def classify_zone(yaw, pitch):
    """
    Returns the BEST matching zone when multiple overlap,
    picking the one whose centre is closest to the measured angle.
    """
    best_idx  = None
    best_dist = float("inf")

    for i, (_, _l1, _l2, y0, y1, p0, p1) in enumerate(CAPTURE_ZONES):
        if y0 <= yaw <= y1 and p0 <= pitch <= p1:
            cy   = (y0 + y1) / 2
            cp   = (p0 + p1) / 2
            dist = (yaw - cy) ** 2 + (pitch - cp) ** 2
            if dist < best_dist:
                best_dist = dist
                best_idx  = i

    return best_idx


def draw_registration_hud(frame, name, zone_buckets, current_zone_idx,
                          next_zone_idx, yaw=None, pitch=None):
    """
    Full registration HUD with live yaw/pitch debug readout.
    """
    h, w = frame.shape[:2]
    zone_counts    = [len(b) for b in zone_buckets]
    total_needed   = len(CAPTURE_ZONES) * SAMPLES_PER_ZONE
    total_done     = sum(zone_counts)
    zones_complete = sum(1 for c in zone_counts if c >= SAMPLES_PER_ZONE)

    # ── TOP BAR ───────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 54), (18, 18, 18), cv2.FILLED)
    cv2.putText(frame, f"Registering: {name}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 215, 60), 2)
    prog_label = f"{total_done}/{total_needed} samples   {zones_complete}/{len(CAPTURE_ZONES)} zones"
    cv2.putText(frame, prog_label,
                (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1)

    # Progress bar (top-right)
    bar_x1, bar_y1, bar_x2, bar_y2 = w - 160, 10, w - 10, 24
    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (50, 50, 50), cv2.FILLED)
    filled = int((bar_x2 - bar_x1) * total_done / total_needed)
    if filled > 0:
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + filled, bar_y2),
                      (0, 200, 100), cv2.FILLED)
    cv2.putText(frame, f"{int(100 * total_done / total_needed)}%",
                (bar_x1 + 2, bar_y2 - 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (220, 255, 220), 1)

    # ── DEBUG READOUT (live yaw / pitch) ──────────────────
    # Shows the raw numbers so you can see exactly what the camera
    # is measuring as you move your head.
    if yaw is not None and pitch is not None:
        dbg = f"yaw={yaw:+.3f}  pitch={pitch:+.3f}"
        # Colour: green if inside some zone, orange if in dead-space
        dbg_color = (0, 220, 120) if current_zone_idx is not None else (0, 140, 255)
        cv2.putText(frame, dbg, (w - 260, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, dbg_color, 1)

    # ── LIVE CAPTURE FLASH ────────────────────────────────
    # ALL guidance is in this single strip — nothing drawn over the face
    flash_y = 58
    if current_zone_idx is not None:
        count = zone_counts[current_zone_idx]
        done  = count >= SAMPLES_PER_ZONE
        if done:
            if next_zone_idx is not None and next_zone_idx != current_zone_idx:
                _, nline1, nline2, *_ = CAPTURE_ZONES[next_zone_idx]
                flash_text = f"  Done!  Now: {nline1} {nline2}"
                flash_color, flash_bg = (60, 220, 255), (0, 30, 40)
            else:
                flash_text = "  All zones complete!  Press S to save."
                flash_color, flash_bg = (60, 255, 120), (0, 40, 0)
        else:
            flash_text  = f"  Capturing {CAPTURE_ZONES[current_zone_idx][0]}  {count}/{SAMPLES_PER_ZONE}  — hold this angle"
            flash_color, flash_bg = (0, 215, 160), (0, 28, 18)
        cv2.rectangle(frame, (0, flash_y), (w, flash_y + 36), flash_bg, cv2.FILLED)
        cv2.putText(frame, flash_text, (10, flash_y + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, flash_color, 2)
    else:
        if next_zone_idx is not None:
            _, nline1, nline2, *_ = CAPTURE_ZONES[next_zone_idx]
            flash_text  = f"  {nline1} {nline2}"
            flash_color, flash_bg = (120, 180, 255), (10, 10, 35)
        else:
            flash_text  = "  Position your face in the camera"
            flash_color, flash_bg = (100, 100, 200), (25, 10, 10)
        cv2.rectangle(frame, (0, flash_y), (w, flash_y + 36), flash_bg, cv2.FILLED)
        cv2.putText(frame, flash_text, (10, flash_y + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, flash_color, 2)

    # ── BOTTOM ZONE PILLS ─────────────────────────────────
    strip_h  = 40
    strip_y  = h - strip_h - 24
    cv2.rectangle(frame, (0, strip_y - 4), (w, strip_y + strip_h + 4),
                  (18, 18, 18), cv2.FILLED)

    n        = len(CAPTURE_ZONES)
    pill_w   = (w - 16) // n
    pill_gap = 2

    for i, (short_label, *_) in enumerate(CAPTURE_ZONES):
        count   = zone_counts[i]
        done    = count >= SAMPLES_PER_ZONE
        active  = (i == current_zone_idx)
        is_next = (i == next_zone_idx) and not done

        px = 8 + i * pill_w
        py = strip_y

        if done:
            bg, border = (0, 70, 0),   (0, 180, 60)
        elif active:
            bg, border = (0, 50, 60),  (0, 200, 160)
        elif is_next:
            bg, border = (40, 30, 0),  (200, 160, 0)
        else:
            bg, border = (30, 30, 30), (70, 70, 70)

        cv2.rectangle(frame, (px, py), (px + pill_w - pill_gap, py + strip_h),
                      bg, cv2.FILLED)
        cv2.rectangle(frame, (px, py), (px + pill_w - pill_gap, py + strip_h),
                      border, 1)

        label_scale = 0.35
        (lw, _), _  = cv2.getTextSize(short_label, cv2.FONT_HERSHEY_SIMPLEX, label_scale, 1)
        lx = px + (pill_w - pill_gap - lw) // 2

        if done:
            lbl_color = (100, 255, 120)
            cv2.putText(frame, "OK", (lx + 2, py + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (60, 200, 80), 1)
        else:
            lbl_color = (255, 200, 60) if is_next else \
                        ((80, 230, 200) if active else (100, 100, 100))

        cv2.putText(frame, short_label, (lx, py + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, label_scale, lbl_color, 1)

        # Mini fill bar
        bar_py = py + strip_h - 7
        cv2.rectangle(frame, (px + 2, bar_py),
                      (px + pill_w - pill_gap - 2, bar_py + 5),
                      (50, 50, 50), cv2.FILLED)
        filled_w = int((pill_w - pill_gap - 4) * min(count / SAMPLES_PER_ZONE, 1.0))
        bar_col  = (0, 180, 60) if done else \
                   ((0, 180, 160) if active else (100, 100, 40))
        if filled_w > 0:
            cv2.rectangle(frame, (px + 2, bar_py),
                          (px + 2 + filled_w, bar_py + 5), bar_col, cv2.FILLED)

    # ── KEY HINTS ─────────────────────────────────────────
    cv2.putText(frame, "S = save now   |   Q = cancel",
                (w // 2 - 130, h - 8),
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

    # FIX 6 — Per-zone cooldown: track the last time a sample was
    # accepted for each zone. Samples taken less than ZONE_SAMPLE_DELAY
    # seconds apart are skipped, forcing diversity within each zone.
    zone_last_sample_time = [0.0] * len(CAPTURE_ZONES)

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
            live_yaw         = None
            live_pitch       = None

            # First incomplete zone = the one we prompt for next
            next_zone_idx = next(
                (i for i, c in enumerate(zone_counts) if c < SAMPLES_PER_ZONE),
                None
            )

            if results.multi_face_landmarks:
                lms             = results.multi_face_landmarks[0].landmark
                profile         = extract_profile(lms, w, h)
                live_yaw, live_pitch = get_face_angles(lms, w, h)
                zone_idx        = classify_zone(live_yaw, live_pitch)

                if profile:
                    # Draw subtle green face mesh
                    for lm in lms:
                        cv2.circle(frame,
                                   (int(lm.x * w), int(lm.y * h)),
                                   1, (0, 180, 80), -1)

                    if zone_idx is not None:
                        current_zone_idx = zone_idx
                        now_t = time.time()

                        # Accept sample only if this zone needs more AND
                        # enough time has passed since the last sample here
                        if (zone_counts[zone_idx] < SAMPLES_PER_ZONE and
                                now_t - zone_last_sample_time[zone_idx] >= ZONE_SAMPLE_DELAY):
                            zone_buckets[zone_idx].append(profile)
                            zone_last_sample_time[zone_idx] = now_t

            draw_registration_hud(frame, name, zone_buckets,
                                  current_zone_idx, next_zone_idx,
                                  live_yaw, live_pitch)

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

    # Global profile (median across all angles) — used as fallback
    avg_profile, std_profile = build_robust_profile(all_profiles)

    # Per-zone profiles — used for angle-aware matching (Fix #2)
    zone_profiles = {}
    for i, bucket in enumerate(zone_buckets):
        if len(bucket) == 0:
            continue
        zone_label = CAPTURE_ZONES[i][0]
        zp, zstd   = build_robust_profile(bucket)
        zone_profiles[zone_label] = {"profile": zp, "std": zstd,
                                     "samples": len(bucket)}

    data[name] = {
        "name"          : name,
        "profile"       : avg_profile,        # global fallback
        "std"           : std_profile,
        "zone_profiles" : zone_profiles,       # per-angle profiles
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

                        # Get live angle for angle-aware matching (Fix #2)
                        live_yaw, live_pitch = get_face_angles(lms, w, h)

                        xs = [int(lm.x*w) for lm in lms]
                        ys = [int(lm.y*h) for lm in lms]
                        x1 = max(0, min(xs)-10)
                        x2 = min(w, max(xs)+10)
                        y1 = max(0, min(ys)-10)
                        y2 = min(h, max(ys)+10)

                        best_name  = None
                        best_score = 0
                        best_diff  = 999.0   # track closest miss too

                        for pname, person in data.items():
                            score, is_match = compare_profiles(
                                person,
                                live_profile,
                                live_yaw,
                                live_pitch
                            )
                            if is_match and score > best_score:
                                best_score = score
                                best_name  = pname
                            # Track closest non-match for soft vote
                            if score > (100 - best_diff * 100):
                                best_diff = (100 - score) / 100

                        if best_name:
                            seen_this_frame.add(best_name)

                            if best_name not in vote_buffer:
                                vote_buffer[best_name] = deque(maxlen=VOTE_WINDOW)
                            vote_buffer[best_name].append(True)

                            # Also feed False into other person's buffers
                            # so they can't accumulate stale votes
                            for other in list(vote_buffer.keys()):
                                if other != best_name:
                                    vote_buffer[other].append(False)

                            vote_ok = (
                                len(vote_buffer[best_name]) >= VOTE_WINDOW and
                                sum(vote_buffer[best_name]) >= VOTE_REQUIRE
                            )

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

                            if already_confirmed:
                                color    = (0, 200, 0)
                                label    = f"{best_name} ✓ ({best_score:.0f}%)"
                                progress = 1.0
                            elif vote_ok and elapsed >= CONFIRM_SECONDS:
                                confirmed_set.add(best_name)
                                marked_today.add(best_name)
                                elapsed_str = f"{elapsed:.1f}"
                                append_attendance_row(
                                    ATTENDANCE_FILE, sheet_name,
                                    best_name, "PRESENT", f"{best_score:.0f}%", elapsed_str
                                )
                                print(f"✅ {best_name} — PRESENT at "
                                      f"{now.strftime('%H:%M:%S')} "
                                      f"({best_score:.0f}%) after {elapsed_str}s")
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
                            # ── No hard match this frame ──────────────────
                            # DON'T immediately show UNKNOWN.
                            # Check if any person has a strong partial vote
                            # buffer from recent frames — if so, keep showing
                            # them as a soft candidate rather than flipping to
                            # UNKNOWN for one bad frame.
                            soft_name  = None
                            soft_votes = 0
                            for pname, buf in vote_buffer.items():
                                v = sum(buf)
                                if v > soft_votes and v >= VOTE_REQUIRE - 2:
                                    soft_votes = v
                                    soft_name  = pname

                            if soft_name and soft_name not in confirmed_set:
                                # Still recognising — just a weak frame
                                color    = (0, 140, 80)
                                label    = f"{soft_name}? (weak frame)"
                                progress = 0
                                seen_this_frame.add(soft_name)
                            elif soft_name and soft_name in confirmed_set:
                                # Already confirmed — keep showing green
                                color    = (0, 200, 0)
                                label    = f"{soft_name} ✓"
                                progress = 1.0
                                seen_this_frame.add(soft_name)
                            else:
                                # Genuinely unknown — apply cooldown
                                color    = (0, 0, 200)
                                label    = "UNKNOWN"
                                loc_key  = f"{x1//30}_{y1//30}"
                                last_log = unknown_last_logged.get(loc_key)
                                if (last_log is None or
                                        (now - last_log).seconds > UNKNOWN_COOLDOWN):
                                    unknown_count += 1
                                    # Disabled saving UNKNOWN to Excel per user request
                                    unknown_last_logged[loc_key] = now
                                    print(f"🚨 Unknown at {now.strftime('%H:%M:%S')}")
                            progress = 0

                        last_locations.append((x1, y1, x2, y2))
                        last_labels.append(label)
                        last_colors.append(color)
                        last_progresses.append(progress if best_name else 0)

                # Decay vote buffers for faces not seen this frame.
                # Instead of immediately clearing (which caused UNKNOWN flicker),
                # inject a False vote — the buffer naturally drains over VOTE_WINDOW
                # frames if the person truly left. Confirm timer resets only when
                # the vote count drops below threshold.
                for name_key in list(vote_buffer.keys()):
                    if name_key not in seen_this_frame:
                        vote_buffer[name_key].append(False)
                        still_ok = (
                            len(vote_buffer[name_key]) >= VOTE_WINDOW and
                            sum(vote_buffer[name_key]) >= VOTE_REQUIRE
                        )
                        if not still_ok:
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
            cv2.putText(frame, f"Hold {CONFIRM_SECONDS}s to confirm | Q=Cancel | S=Save & Quit",
                        (10,52), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120,120,120), 1)

            cv2.imshow("Attendance System", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
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
            elif key == ord('s'):
                break

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