import cv2
import mediapipe as mp
import numpy as np
import json
import os
from datetime import datetime

mp_face_mesh = mp.solutions.face_mesh

# ── Helpers ──────────────────────────────────────────────

def euclidean(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def get_coords(landmarks, idx, w, h):
    lm = landmarks[idx]
    return (int(lm.x * w), int(lm.y * h))

def extract_profile(landmarks, w, h):
    """Extract face ratios from landmarks"""

    def pt(i): return get_coords(landmarks, i, w, h)

    forehead    = pt(10)
    chin        = pt(152)
    left_cheek  = pt(234)
    right_cheek = pt(454)
    nose_tip    = pt(1)
    top_lip     = pt(13)
    left_eye    = pt(33)
    right_eye   = pt(263)
    left_lip    = pt(61)
    right_lip   = pt(291)
    nose        = pt(1)

    face_height  = euclidean(forehead, chin)
    face_width   = euclidean(left_cheek, right_cheek)
    eye_distance = euclidean(left_eye, right_eye)
    nose_to_chin = euclidean(nose_tip, chin)
    lip_to_chin  = euclidean(top_lip, chin)
    lip_width    = euclidean(left_lip, right_lip)

    # Symmetry score
    mirror_pairs = [(33,263),(130,359),(234,454),(61,291),(70,300)]
    diffs = []
    for l, r in mirror_pairs:
        lp = pt(l)
        rp = pt(r)
        diffs.append(abs(abs(lp[0]-nose[0]) - abs(rp[0]-nose[0])))
    symmetry = max(0, 1 - (np.mean(diffs) / face_width)) * 100

    return {
        "golden_ratio" : round(face_height / face_width, 4),
        "eye_ratio"    : round(eye_distance / face_width, 4),
        "nose_ratio"   : round(nose_to_chin / face_height, 4),
        "lip_ratio"    : round(lip_width / face_width, 4),
        "fWHR"         : round(face_width / (face_height / 2), 4),
        "symmetry"     : round(symmetry, 2),
    }

def process_image(image_path):
    """Extract face profile from a single image file"""
    frame = cv2.imread(image_path)
    if frame is None:
        return None

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(
            static_image_mode=True,        # important for images!
            max_num_faces=1,
            min_detection_confidence=0.5) as face_mesh:

        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            lms = results.multi_face_landmarks[0].landmark
            return extract_profile(lms, w, h)

    return None  # no face found in image

# ── Phase 1: Register from 40 Sample Images ──────────────

def register_from_dataset(dataset_path="dataset"):
    """
    Reads dataset/ folder
    Each subfolder = one person (format: Name_RollNo)
    Processes all images and saves averaged profile
    """

    if not os.path.exists(dataset_path):
        print(f"❌ Dataset folder '{dataset_path}' not found!")
        return

    data = {}
    if os.path.exists("face_profiles.json"):
        with open("face_profiles.json", "r") as f:
            data = json.load(f)

    persons = os.listdir(dataset_path)
    print(f"\n📂 Found {len(persons)} persons in dataset\n")

    for person_folder in persons:
        folder_path = os.path.join(dataset_path, person_folder)
        if not os.path.isdir(folder_path):
            continue

        # Parse name and roll from folder name
        # Expected format: "Rahul_CS101"
        parts   = person_folder.split("_")
        name    = parts[0]
        roll_no = parts[1] if len(parts) > 1 else person_folder

        print(f"👤 Processing: {name} ({roll_no})")

        image_files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.jpg','.jpeg','.png'))
        ]

        profiles = []
        failed   = 0

        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            profile  = process_image(img_path)

            if profile:
                profiles.append(profile)
            else:
                failed += 1

        if not profiles:
            print(f"   ⚠️  No valid faces found in any image! Skipping.")
            continue

        # Average all profiles for stability
        avg_profile = {}
        for key in profiles[0].keys():
            avg_profile[key] = round(
                np.mean([p[key] for p in profiles]), 4
            )

        data[roll_no] = {
            "name"          : name,
            "roll_no"       : roll_no,
            "profile"       : avg_profile,
            "total_samples" : len(profiles),
            "failed_images" : failed
        }

        print(f"   ✅ Registered with {len(profiles)} valid samples "
              f"({failed} failed)")

    with open("face_profiles.json", "w") as f:
        json.dump(data, f, indent=4)

    print(f"\n🎉 Registration complete! {len(data)} persons saved.")

# ── Phase 2: Live Recognition + Attendance ────────────────

def compare_profiles(stored, live, threshold=0.06):
    """Compare stored vs live profile. Returns score and match bool"""
    keys  = ["golden_ratio", "eye_ratio", "nose_ratio", "lip_ratio", "fWHR"]
    diffs = []

    for key in keys:
        if stored[key] != 0:
            diff = abs(stored[key] - live[key]) / stored[key]
            diffs.append(diff)

    avg_diff    = np.mean(diffs)
    match_score = round(max(0, 1 - avg_diff) * 100, 2)
    is_match    = avg_diff < threshold

    return match_score, is_match

def mark_attendance():
    """Live camera — recognize face and mark attendance"""

    if not os.path.exists("face_profiles.json"):
        print("❌ No profiles found! Run registration first.")
        return

    with open("face_profiles.json", "r") as f:
        data = json.load(f)

    print(f"\n📷 Loaded {len(data)} registered persons")
    print("Press Q to quit\n")

    cap         = cv2.VideoCapture(0)
    marked      = {}   # roll_no: timestamp  (prevent duplicates)
    COOLDOWN    = 5    # seconds between re-marking same person

    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results  = face_mesh.process(rgb)

            label = "No Face Detected"
            color = (128, 128, 128)

            if results.multi_face_landmarks:
                lms          = results.multi_face_landmarks[0].landmark
                live_profile = extract_profile(lms, w, h)

                best_match = None
                best_score = 0

                # Compare with all stored profiles
                for roll_no, person in data.items():
                    score, is_match = compare_profiles(
                        person["profile"], live_profile
                    )
                    if is_match and score > best_score:
                        best_score = score
                        best_match = person

                now = datetime.now()

                if best_match:
                    # ── KNOWN PERSON ──
                    name = best_match["name"]
                    roll = best_match["roll_no"]
                    color = (0, 255, 0)
                    label = f"{name} | {best_score}%"

                    # Check cooldown before marking
                    last_marked = marked.get(roll)
                    if (last_marked is None or
                        (now - last_marked).seconds > COOLDOWN):

                        with open("attendance.csv", "a") as f:
                            f.write(
                                f"{name},{roll},"
                                f"{now.strftime('%Y-%m-%d %H:%M:%S')}\n"
                            )
                        marked[roll] = now
                        print(f"✅ {name} — Attendance Marked")

                else:
                    # ── UNKNOWN PERSON ──
                    color = (0, 0, 255)
                    label = "⚠ UNKNOWN PERSON"
                    print("🚨 Unknown face detected!")

                # Draw box area at top
                cv2.rectangle(frame, (0,0), (w, 60), (0,0,0), -1)
                cv2.putText(frame, label, (10, 40),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           1, color, 2)

                # Show live ratios (small text)
                y = 80
                for key, val in live_profile.items():
                    cv2.putText(frame, f"{key}: {val}",
                               (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                               0.45, (200, 200, 200), 1)
                    y += 20

            cv2.imshow("Attendance System", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# ── Main Menu ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 40)
    print("   FACE RECOGNITION ATTENDANCE SYSTEM")
    print("=" * 40)
    print("1. Register from Dataset (40 samples)")
    print("2. Mark Attendance (Live Camera)")
    print("=" * 40)

    choice = input("Choose option: ")

    if choice == "1":
        register_from_dataset()
    elif choice == "2":
        mark_attendance()
    else:
        print("Invalid choice!")
        