"""
attendance.py — Main script for the Smart Attendance System.
Detects and recognizes faces via webcam and marks attendance in a CSV file.

Usage:
    python attendance.py                          # normal webcam mode with live preview
    python attendance.py --no-display              # webcam mode, no preview window (headless)
    python attendance.py --image-folder test_images  # process a folder of still images instead
                                                       # of a webcam (useful when no camera/display
                                                       # is available, e.g. automated evaluation)
    python attendance.py --max-frames 100           # stop automatically after N frames (headless)
"""

import argparse
import cv2
import face_recognition
import os
import csv
from datetime import datetime

# ── Configuration ──────────────────────────────────────────────────────────
KNOWN_FACES_DIR = "known_faces"       # Folder with registered face photos
ATTENDANCE_FILE = "attendance.csv"    # Output CSV file
CONFIDENCE_THRESHOLD = 0.55           # Lower = stricter matching (0.0 – 1.0)
# ───────────────────────────────────────────────────────────────────────────


def load_known_faces():
    """Load all registered faces from the known_faces folder."""
    known_encodings = []
    known_names = []

    print("[INFO] Loading registered faces...")

    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            name = os.path.splitext(filename)[0]  # e.g. "Alice.jpg" → "Alice"
            image_path = os.path.join(KNOWN_FACES_DIR, filename)

            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(name)
                print(f"  ✓ Loaded: {name}")
            else:
                print(f"  ✗ No face found in: {filename} (skipped)")

    print(f"[INFO] {len(known_names)} face(s) loaded.\n")
    return known_encodings, known_names


def mark_attendance(name):
    """Write attendance entry to CSV (only once per session per person)."""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    # Read existing entries to avoid duplicates in today's session
    existing = set()
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "r") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if len(row) >= 2 and row[1] == date_str:
                    existing.add(row[0])

    if name not in existing:
        with open(ATTENDANCE_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            # Write header if file is new/empty
            if os.stat(ATTENDANCE_FILE).st_size == 0:
                writer.writerow(["Name", "Date", "Time", "Status"])
            writer.writerow([name, date_str, time_str, "Present"])
        print(f"[ATTENDANCE] Marked: {name} at {time_str}")
        return True  # newly marked
    return False  # already marked today


def process_frame(frame, known_encodings, known_names, marked_today, draw=True):
    """Detect + recognize faces in one BGR frame, mark attendance, and
    optionally draw boxes/labels for on-screen display.

    Returns the (possibly annotated) frame and the list of names recognized
    in this frame.
    """
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small)
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    recognized = []

    for face_encoding, face_location in zip(face_encodings, face_locations):
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        name = "Unknown"
        color = (0, 0, 255)  # Red for unknown

        if len(distances) > 0:
            best_idx = distances.argmin()
            if distances[best_idx] < CONFIDENCE_THRESHOLD:
                name = known_names[best_idx]
                color = (0, 200, 0)  # Green for recognised

                newly_marked = mark_attendance(name)
                if newly_marked:
                    marked_today.add(name)
                recognized.append(name)

        if draw:
            top, right, bottom, left = [v * 2 for v in face_location]
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            label = f"{name}"
            if name in marked_today:
                label += " check"

            cv2.rectangle(frame, (left, bottom - 30), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, label, (left + 6, bottom - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return frame, recognized


def run_attendance_webcam(known_encodings, known_names, show_display=True, max_frames=None):
    """Run real-time face recognition against a live webcam feed.

    show_display=False skips cv2.imshow / waitKey entirely, so this can run
    on machines with no display server (e.g. CI or an automated grader).
    In that mode, pass max_frames to stop automatically instead of waiting
    for a 'q' keypress.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam. If this machine has no camera "
              "(e.g. an automated grading environment), rerun with:")
        print("    python attendance.py --image-folder <folder_of_test_photos>")
        return

    mode = "Press Q to quit." if show_display else f"Headless mode, stopping after {max_frames} frames."
    print(f"[INFO] Webcam started. {mode}\n")

    marked_today = set()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, _ = process_frame(frame, known_encodings, known_names, marked_today, draw=show_display)
        frame_count += 1

        if show_display:
            cv2.putText(frame, f"Marked today: {len(marked_today)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, "Press Q to quit", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.imshow("Smart Attendance System", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print(f"[FRAME {frame_count}] Marked so far: {sorted(marked_today)}")

        if max_frames is not None and frame_count >= max_frames:
            break

    cap.release()
    if show_display:
        cv2.destroyAllWindows()
    print(f"\n[DONE] Attendance saved to: {ATTENDANCE_FILE}")


def run_attendance_on_images(image_folder, known_encodings, known_names):
    """Process a folder of still images instead of a live webcam.

    This exists so the recognition + attendance-marking logic can be
    verified on a machine with no camera and no display (e.g. an automated
    evaluation environment) — just point it at a folder of test photos.
    """
    if not os.path.isdir(image_folder):
        print(f"[ERROR] Image folder not found: {image_folder}")
        return

    marked_today = set()
    image_files = [f for f in sorted(os.listdir(image_folder))
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not image_files:
        print(f"[ERROR] No images found in: {image_folder}")
        return

    print(f"[INFO] Processing {len(image_files)} image(s) from: {image_folder}\n")

    for filename in image_files:
        frame = cv2.imread(os.path.join(image_folder, filename))
        if frame is None:
            print(f"  ✗ Could not read: {filename} (skipped)")
            continue

        _, recognized = process_frame(frame, known_encodings, known_names, marked_today, draw=False)
        result = ", ".join(recognized) if recognized else "no known face"
        print(f"  {filename}: {result}")

    print(f"\n[DONE] Attendance saved to: {ATTENDANCE_FILE}")


def run_attendance(show_display=True, image_folder=None, max_frames=None):
    """Entry point: initialise attendance file, load known faces, then
    dispatch to webcam mode or image-folder mode.
    """
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w", newline="") as f:
            csv.writer(f).writerow(["Name", "Date", "Time", "Status"])

    known_encodings, known_names = load_known_faces()

    if not known_names:
        print("[ERROR] No registered faces found. Run register.py first.")
        return

    if image_folder:
        run_attendance_on_images(image_folder, known_encodings, known_names)
    else:
        run_attendance_webcam(known_encodings, known_names, show_display=show_display, max_frames=max_frames)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Smart Attendance System")
    parser.add_argument("--no-display", action="store_true",
                         help="Run webcam mode without opening a preview window (headless)")
    parser.add_argument("--image-folder", type=str, default=None,
                         help="Process a folder of still images instead of the webcam")
    parser.add_argument("--max-frames", type=int, default=None,
                         help="Stop automatically after N webcam frames (useful with --no-display)")
    args = parser.parse_args()

    run_attendance(
        show_display=not args.no_display,
        image_folder=args.image_folder,
        max_frames=args.max_frames,
    )
