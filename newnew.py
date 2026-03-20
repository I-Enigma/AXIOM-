import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import json
import os
import time
from datetime import datetime

# ── Configuration ─────────────────────────────────────────
PROFILES_FILE    = "face_profiles.json"
ATTENDANCE_FILE  = "attendance.xlsx"
MATCH_THRESHOLD  = 0.06
UNKNOWN_COOLDOWN = 10

mp_face_mesh = mp.solutions.face_mesh

# ── Facial Geometry ───────────────────────────────────────

def euclidean(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def get_pt(landmarks, idx, w, h):
    lm = landmarks[idx]
    return (int(lm.x * w), int(lm.y * h))

def extract_profile(landmarks, w, h):
    def pt(i): return get_pt(landmarks, i, w, h)

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

def compare_profiles(stored, live):
    keys = [
        "golden_ratio","eye_ratio","inner_eye_ratio",
        "nose_ratio","lip_ratio","jaw_ratio",
        "brow_ratio","nose_width_ratio","fWHR"
    ]
    diffs = []
    for key in keys:
        if stored.get(key, 0) != 0:
            diffs.append(abs(stored[key]-live[key]) / stored[key])

    avg_diff    = np.mean(diffs)
    match_score = round(max(0, 1-avg_diff)*100, 2)
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

# ── Attendance Storage ────────────────────────────────────

def load_attendance():
    expected_cols = ["Name","Date","Time","Status","Match_Score"]
    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_excel(ATTENDANCE_FILE)
        # Handle older attendance files that are missing new columns
        for col in expected_cols:
            if col not in df.columns:
                df[col] = "" 
        return df
    return pd.DataFrame(columns=expected_cols)

def save_attendance_file(df):
    for attempt in range(3):
        try:
            df.to_excel(ATTENDANCE_FILE, index=False)
            return True
        except PermissionError:
            print(f"⚠️  Close Excel file and press Enter... ({attempt+1}/3)")
            input()
        except Exception as e:
            print(f"⚠️  Save error: {e}")
            return False
    return False

def log_entry(df, name, status, score=""):
    now = datetime.now()
    new_row = pd.DataFrame({
        "Name"        : [name],
        "Date"        : [now.strftime("%Y-%m-%d")],
        "Time"        : [now.strftime("%H:%M:%S")],
        "Status"      : [status],
        "Match_Score" : [score]
    })
    return pd.concat([df, new_row], ignore_index=True)

# ══════════════════════════════════════════════════════════
# OPTION 1 — REGISTER NEW PERSON VIA LIVE CAMERA
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

    print(f"\n👤 Registering: {name}")
    print("📸 Look straight at camera. Don't move...")
    print("Press Q to cancel\n")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("❌ Cannot open camera!")
        return

    profiles = []

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

            if results.multi_face_landmarks:
                lms     = results.multi_face_landmarks[0].landmark
                profile = extract_profile(lms, w, h)

                if profile:
                    profiles.append(profile)

                    # Draw green dots on face
                    for lm in lms:
                        cx, cy = int(lm.x*w), int(lm.y*h)
                        cv2.circle(frame, (cx,cy), 1, (0,200,0), -1)

                    # Status text
                    cv2.rectangle(frame, (0,0), (640,60), (20,20,20), -1)
                    cv2.putText(frame, f"Scanning {name}...",
                                (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0,220,0), 2)
                    cv2.putText(frame,
                                f"Samples: {len(profiles)} | Press 'S' to Save",
                                (10,52), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (150,150,150), 1)
            else:
                cv2.rectangle(frame, (0,0), (640,60), (20,20,20), -1)
                cv2.putText(frame, "No face detected — look at camera",
                            (10,35), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0,0,220), 2)

            cv2.putText(frame,
                        f"{len(profiles)} captured",
                        (220,415), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255,255,255), 2)
            cv2.putText(frame, "Press S to Save | Q to Cancel",
                        (150, 445), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2)

            cv2.imshow("Registration — Press S to Save / Q to Cancel", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Registration cancelled.")
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord('s'):
                if len(profiles) < 10:
                    print(f"⚠️ Only got {len(profiles)} samples. Try to get at least 10 before saving.")
                else:
                    print("\nSaving profile...")
                    break

    cap.release()
    cv2.destroyAllWindows()

    if len(profiles) == 0:
        print("❌ No samples collected.")
        return

    # Average all samples → stable face profile
    avg_profile = {}
    for key in profiles[0].keys():
        avg_profile[key] = round(np.mean([p[key] for p in profiles]), 4)

    data[name] = {
        "name"          : name,
        "profile"       : avg_profile,
        "total_samples" : len(profiles),
        "registered_on" : datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    save_profiles(data)

    print(f"\n✅ {name} registered successfully!")
    print(f"   Samples collected : {len(profiles)}")
    print(f"   Golden Ratio      : {avg_profile['golden_ratio']}")
    print(f"   Symmetry Score    : {avg_profile['symmetry']}%")
    print(f"   Eye Ratio         : {avg_profile['eye_ratio']}")

# ══════════════════════════════════════════════════════════
# OPTION 2 — MARK ATTENDANCE
# ══════════════════════════════════════════════════════════

def mark_attendance():
    data = load_profiles()

    if not data:
        print("\n❌ No profiles registered yet!")
        print("   Please register first using Option 1.")
        return

    print(f"\n📋 {len(data)} registered persons loaded")
    print("Controls: Q = Quit | S = Show summary")
    print("="*50)

    df    = load_attendance()
    today = datetime.now().strftime("%Y-%m-%d")

    # Load already marked today — no duplicates
    marked_today = set(
        df[(df["Date"]==today) & (df["Status"]=="PRESENT")]["Name"].tolist()
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
    fps                 = 0
    fps_timer           = time.time()
    frame_counter       = 0

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
            today          = now.strftime("%Y-%m-%d")
            h, w, _        = frame.shape

            if time.time() - fps_timer >= 1.0:
                fps           = frame_counter
                frame_counter = 0
                fps_timer     = time.time()

            # Process every 2nd frame
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

                        # Face bounding box
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
                                person["profile"], live_profile
                            )
                            if is_match and score > best_score:
                                best_score = score
                                best_name  = name

                        if best_name:
                            # ── KNOWN PERSON ───────────────
                            color = (0, 200, 0)
                            label = f"{best_name} ({best_score}%)"

                            if best_name not in marked_today:
                                df = log_entry(df, best_name,
                                               "PRESENT", f"{best_score}%")
                                save_attendance_file(df)
                                marked_today.add(best_name)
                                print(f"✅ {best_name} — PRESENT at "
                                      f"{now.strftime('%H:%M:%S')} "
                                      f"({best_score}%)")
                        else:
                            # ── UNKNOWN PERSON ─────────────
                            color    = (0, 0, 220)
                            loc_key  = f"{x1//30}_{y1//30}"
                            last_log = unknown_last_logged.get(loc_key)

                            if (last_log is None or
                                    (now-last_log).seconds > UNKNOWN_COOLDOWN):
                                unknown_count += 1
                                df = log_entry(
                                    df, f"UNKNOWN #{unknown_count}",
                                    "UNKNOWN", "N/A"
                                )
                                save_attendance_file(df)
                                unknown_last_logged[loc_key] = now
                                print(f"🚨 Unknown person at "
                                      f"{now.strftime('%H:%M:%S')}")

                            label = "UNKNOWN"

                        last_locations.append((x1,y1,x2,y2))
                        last_labels.append(label)
                        last_colors.append(color)

            # ── Draw face boxes ───────────────────────────
            for i, (x1,y1,x2,y2) in enumerate(last_locations):
                color = last_colors[i] if i < len(last_colors) else (128,128,128)
                label = last_labels[i] if i < len(last_labels) else ""

                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                cv2.rectangle(frame, (x1,y2), (x2,y2+28), color, cv2.FILLED)
                cv2.putText(frame, label, (x1+4, y2+20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (255,255,255), 2)

            # ── HUD Bar ───────────────────────────────────
            cv2.rectangle(frame, (0,0), (640,65), (25,25,25), cv2.FILLED)
            cv2.putText(frame, f"FPS:{fps}",
                        (10,25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (150,150,150), 1)
            cv2.putText(frame, f"Present: {len(marked_today)}",
                        (80,25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0,220,0), 2)
            cv2.putText(frame, f"Unknown: {unknown_count}",
                        (270,25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0,80,255), 2)
            cv2.putText(frame, now.strftime("%H:%M:%S"),
                        (490,25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255,255,255), 2)
            cv2.putText(frame, f"Q=Quit | S=Summary | {today}",
                        (10,52), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (120,120,120), 1)

            cv2.imshow("Attendance System", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                print("\n" + "="*40)
                known_df = df[(df["Date"]==today) & (df["Status"]=="PRESENT")]
                for _, row in known_df.iterrows():
                    print(f"  ✅ {row['Time']} — {row['Name']} ({row['Match_Score']})")
                print(f"  🚨 Unknowns logged: {unknown_count}")
                print("="*40+"\n")

    cap.release()
    cv2.destroyAllWindows()
    save_attendance_file(df)

    # Final summary
    print("\n" + "="*50)
    known_df   = df[(df["Date"]==today) & (df["Status"]=="PRESENT")]
    unknown_df = df[(df["Date"]==today) & (df["Status"]=="UNKNOWN")]
    print(f"✅ Present today   : {len(known_df)}")
    for _, row in known_df.iterrows():
        print(f"     {row['Time']} — {row['Name']}")
    print(f"🚨 Unknown detected: {len(unknown_df)}")
    print(f"💾 Saved to        : {ATTENDANCE_FILE}")
    print("="*50)

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
            data = load_profiles()
            if not data:
                print("\n❌ No one registered yet!")
            else:
                print(f"\n📋 {len(data)} Registered Persons:")
                print("-"*40)
                for name, info in data.items():
                    print(f"  👤 {name}")
                    print(f"     Registered : {info['registered_on']}")
                    print(f"     Samples    : {info['total_samples']}")
                    print(f"     Symmetry   : {info['profile']['symmetry']}%")
                    print(f"     Golden Ratio: {info['profile']['golden_ratio']}")
                    print()

        elif choice == "4":
            print("\nGoodbye! 👋")
            break

        else:
            print("❌ Invalid choice!")