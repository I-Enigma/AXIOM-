import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime

# ── Configuration ─────────────────────────────────────────
DATASET_PATH     = "dataset"
PROFILES_FILE    = "face_profiles.json"
ATTENDANCE_FILE  = "attendance.xlsx"
MATCH_THRESHOLD  = 0.06    # 6% difference allowed (lower = stricter)
UNKNOWN_COOLDOWN = 10      # seconds between logging unknowns
SAMPLES_PER_IMG  = 1       # profile extractions per image

mp_face_mesh = mp.solutions.face_mesh

# ── Helper Functions ──────────────────────────────────────

def euclidean(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def get_pt(landmarks, idx, w, h):
    lm = landmarks[idx]
    return (int(lm.x * w), int(lm.y * h))

def extract_profile(landmarks, w, h):
    """Extract all facial ratios from 478 mediapipe landmarks"""

    def pt(i): return get_pt(landmarks, i, w, h)

    # Key facial points
    forehead    = pt(10)
    chin        = pt(152)
    left_cheek  = pt(234)
    right_cheek = pt(454)
    nose_tip    = pt(1)
    top_lip     = pt(13)
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
    nose_base   = pt(2)

    # Core measurements
    face_height    = euclidean(forehead, chin)
    face_width     = euclidean(left_cheek, right_cheek)
    eye_distance   = euclidean(left_eye_o, right_eye_o)
    inner_eye_dist = euclidean(left_eye_i, right_eye_i)
    nose_to_chin   = euclidean(nose_tip, chin)
    lip_to_chin    = euclidean(top_lip, chin)
    lip_width      = euclidean(left_lip, right_lip)
    jaw_width      = euclidean(left_jaw, right_jaw)
    brow_width     = euclidean(left_brow, right_brow)
    nose_width     = euclidean(pt(129), pt(358))

    # Symmetry calculation
    nose_center = pt(1)
    mirror_pairs = [
        (33, 263), (130, 359), (234, 454),
        (61, 291), (70, 300), (172, 397)
    ]
    diffs = []
    for l, r in mirror_pairs:
        lp = pt(l)
        rp = pt(r)
        diffs.append(abs(
            abs(lp[0] - nose_center[0]) - abs(rp[0] - nose_center[0])
        ))
    symmetry = max(0, 1 - (np.mean(diffs) / face_width)) * 100

    # Avoid division by zero
    if face_width == 0 or face_height == 0:
        return None

    return {
        "golden_ratio"    : round(face_height / face_width, 4),
        "eye_ratio"       : round(eye_distance / face_width, 4),
        "inner_eye_ratio" : round(inner_eye_dist / face_width, 4),
        "nose_ratio"      : round(nose_to_chin / face_height, 4),
        "lip_ratio"       : round(lip_width / face_width, 4),
        "jaw_ratio"       : round(jaw_width / face_width, 4),
        "brow_ratio"      : round(brow_width / face_width, 4),
        "nose_width_ratio": round(nose_width / face_width, 4),
        "fWHR"            : round(face_width / (face_height / 2), 4),
        "symmetry"        : round(symmetry, 2),
    }

def compare_profiles(stored, live, threshold=MATCH_THRESHOLD):
    """
    Compare stored profile vs live profile
    Returns: (match_score %, is_match bool, per_key differences)
    """
    keys = [
        "golden_ratio", "eye_ratio", "inner_eye_ratio",
        "nose_ratio", "lip_ratio", "jaw_ratio",
        "brow_ratio", "nose_width_ratio", "fWHR"
    ]
    diffs = []
    for key in keys:
        if stored.get(key, 0) != 0:
            diff = abs(stored[key] - live[key]) / stored[key]
            diffs.append(diff)

    avg_diff    = np.mean(diffs)
    match_score = round(max(0, 1 - avg_diff) * 100, 2)
    is_match    = avg_diff < threshold

    return match_score, is_match

# ── Phase 1: Register from Dataset ───────────────────────

def register_from_dataset():
    print("\n" + "=" * 50)
    print("         REGISTERING FROM DATASET")
    print("=" * 50)

    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH)
        print(f"❌ Created '{DATASET_PATH}' folder.")
        print("   Add subfolders named after each person with images.")
        return

    data = {}
    if os.path.exists(PROFILES_FILE):
        with open(PROFILES_FILE, "r") as f:
            data = json.load(f)

    persons = [
        p for p in os.listdir(DATASET_PATH)
        if os.path.isdir(os.path.join(DATASET_PATH, p))
    ]

    print(f"📂 Found {len(persons)} persons\n")

    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5) as face_mesh:

        for person_name in persons:
            folder   = os.path.join(DATASET_PATH, person_name)
            images   = [
                f for f in os.listdir(folder)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]

            print(f"👤 Processing: {person_name} ({len(images)} images)")

            profiles = []
            failed   = 0

            for img_file in images:
                img_path = os.path.join(folder, img_file)
                frame    = cv2.imread(img_path)

                if frame is None:
                    failed += 1
                    continue

                h, w, _ = frame.shape
                rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results  = face_mesh.process(rgb)

                if results.multi_face_landmarks:
                    lms     = results.multi_face_landmarks[0].landmark
                    profile = extract_profile(lms, w, h)
                    if profile:
                        profiles.append(profile)
                else:
                    failed += 1

            if not profiles:
                print(f"   ⚠️  No faces detected! Skipping.\n")
                continue

            # Average all profiles for a stable identity
            avg_profile = {}
            for key in profiles[0].keys():
                avg_profile[key] = round(
                    np.mean([p[key] for p in profiles]), 4
                )

            data[person_name] = {
                "name"          : person_name,
                "profile"       : avg_profile,
                "total_samples" : len(profiles),
                "failed_images" : failed,
                "registered_on" : datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            print(f"   ✅ Registered! "
                  f"({len(profiles)} valid, {failed} failed)\n")

    with open(PROFILES_FILE, "w") as f:
        json.dump(data, f, indent=4)

    print(f"🎉 Done! {len(data)} persons saved to {PROFILES_FILE}")

# ── Attendance Functions ──────────────────────────────────

def load_attendance():
    expected_cols = ["Name", "Date", "Time", "Type", "Match_Score"]
    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_excel(ATTENDANCE_FILE)
        # Add any missing columns (handles old attendance files)
        for col in expected_cols:
            if col not in df.columns:
                df[col] = "KNOWN" if col == "Type" else ("N/A" if col == "Match_Score" else "")
        return df
    return pd.DataFrame(columns=expected_cols)

def save_attendance(df):
    df.to_excel(ATTENDANCE_FILE, index=False)

def log_entry(df, name, entry_type, score=""):
    now = datetime.now()
    new_row = pd.DataFrame({
        "Name"        : [name],
        "Date"        : [now.strftime("%Y-%m-%d")],
        "Time"        : [now.strftime("%H:%M:%S")],
        "Type"        : [entry_type],
        "Match_Score" : [score]
    })
    return pd.concat([df, new_row], ignore_index=True)

# ── Phase 2: Live Attendance ──────────────────────────────

def mark_attendance():
    if not os.path.exists(PROFILES_FILE):
        print("❌ No profiles found! Run registration first.")
        return

    with open(PROFILES_FILE, "r") as f:
        data = json.load(f)

    print(f"\n📋 Loaded {len(data)} registered persons")
    print("Controls: Q = Quit | S = Summary")
    print("=" * 50)

    df    = load_attendance()
    today = datetime.now().strftime("%Y-%m-%d")

    # Load already marked today
    marked_today = set(
        df[(df["Date"] == today) & (df["Type"] == "KNOWN")]["Name"].tolist()
    )

    if marked_today:
        print(f"📌 Already marked today: {list(marked_today)}")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("❌ Cannot open camera!")
        return

    frame_count         = 0
    unknown_count       = 0
    unknown_last_logged = {}
    last_locations      = []
    last_labels         = []
    last_colors         = []

    with mp_face_mesh.FaceMesh(
            max_num_faces=3,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            now   = datetime.now()
            today = now.strftime("%Y-%m-%d")
            h, w, _ = frame.shape

            # Process every 2nd frame for speed
            if frame_count % 2 == 0:
                rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)

                last_locations = []
                last_labels    = []
                last_colors    = []

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        lms          = face_landmarks.landmark
                        live_profile = extract_profile(lms, w, h)

                        if not live_profile:
                            continue

                        # Get face bounding box from landmarks
                        xs = [int(lm.x * w) for lm in lms]
                        ys = [int(lm.y * h) for lm in lms]
                        x1, x2 = max(0, min(xs)-10), min(w, max(xs)+10)
                        y1, y2 = max(0, min(ys)-10), min(h, max(ys)+10)

                        # Compare with all stored profiles
                        best_name  = None
                        best_score = 0

                        for name, person in data.items():
                            score, is_match = compare_profiles(
                                person["profile"], live_profile
                            )
                            if is_match and score > best_score:
                                best_score = score
                                best_name  = name

                        if best_name:
                            # ── KNOWN ──────────────────────
                            color = (0, 200, 0)
                            label = f"{best_name} ({best_score}%)"

                            if best_name not in marked_today:
                                df = log_entry(df, best_name,
                                               "KNOWN", f"{best_score}%")
                                save_attendance(df)
                                marked_today.add(best_name)
                                print(f"✅ {best_name} marked at "
                                      f"{now.strftime('%H:%M:%S')} "
                                      f"— {best_score}%")
                        else:
                            # ── UNKNOWN ────────────────────
                            color    = (0, 0, 220)
                            loc_key  = f"{x1//30}_{y1//30}"
                            last_log = unknown_last_logged.get(loc_key)

                            if (last_log is None or
                                    (now - last_log).seconds > UNKNOWN_COOLDOWN):
                                unknown_count += 1
                                unknown_last_logged[loc_key] = now
                                print(f"🚨 Unknown #{unknown_count} at "
                                      f"{now.strftime('%H:%M:%S')}")

                            label = f"Unknown #{unknown_count}"

                        last_locations.append((x1, y1, x2, y2))
                        last_labels.append(label)
                        last_colors.append(color)

            # ── Draw boxes ────────────────────────────────
            for i, (x1, y1, x2, y2) in enumerate(last_locations):
                color = last_colors[i] if i < len(last_colors) else (128,128,128)
                label = last_labels[i] if i < len(last_labels) else ""

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(frame, (x1, y2), (x2, y2+28), color, cv2.FILLED)
                cv2.putText(frame, label, (x1+4, y2+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (255,255,255), 2)

            # ── HUD ───────────────────────────────────────
            cv2.rectangle(frame, (0,0), (640, 65), (25,25,25), cv2.FILLED)
            cv2.putText(frame, f"Present : {len(marked_today)}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 220, 0), 2)
            cv2.putText(frame, f"Unknown : {unknown_count}",
                        (10, 52), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 80, 255), 2)
            cv2.putText(frame, now.strftime("%H:%M:%S"),
                        (490, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255,255,255), 2)
            cv2.putText(frame, today,
                        (490, 52), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (180,180,180), 1)

            cv2.imshow("Face Attendance System", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                print("\n" + "="*40)
                known_df = df[(df["Date"]==today) & (df["Type"]=="KNOWN")]
                for _, row in known_df.iterrows():
                    print(f"  ✅ {row['Time']} — {row['Name']} "
                          f"({row['Match_Score']})")
                print(f"\n  🚨 Unknowns: {unknown_count}")
                print("="*40+"\n")

    cap.release()
    cv2.destroyAllWindows()
    save_attendance(df)

    # Final summary
    print("\n" + "="*50)
    known_df   = df[(df["Date"]==today) & (df["Type"]=="KNOWN")]
    unknown_df = df[(df["Date"]==today) & (df["Type"]=="UNKNOWN")]
    print(f"✅ Known present   : {len(known_df)}")
    print(f"🚨 Unknown detected: {len(unknown_df)}")
    print(f"💾 Saved to        : {ATTENDANCE_FILE}")
    print("="*50)

# ── Main Menu ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("      FACE ATTENDANCE SYSTEM")
    print("      (Facial Geometry Based)")
    print("=" * 50)
    print("1. Register from Dataset")
    print("2. Mark Attendance (Live Camera)")
    print("=" * 50)

    choice = input("Choose option (1/2): ").strip()

    if choice == "1":
        register_from_dataset()
    elif choice == "2":
        mark_attendance()
    else:
        print("Invalid choice!")