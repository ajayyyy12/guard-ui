import os
import sys
import time
import cv2

from typing import Optional, Dict

try:
    from ultralytics import YOLO
except Exception as e:
    print(f"❌ Failed to import ultralytics.YOLO: {e}")
    raise

# Firebase (admin SDK)
_FIREBASE_READY = False
_FIREBASE_DB = None


def init_firebase_if_needed():
    global _FIREBASE_READY, _FIREBASE_DB
    if _FIREBASE_READY and _FIREBASE_DB is not None:
        return True
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore

        # Initialize once per process
        if not firebase_admin._apps:
            cred_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "serviceAccountKey.json")
            if not os.path.exists(cred_path):
                print(f"❌ serviceAccountKey.json not found at: {cred_path}")
                return False
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)

        _FIREBASE_DB = firestore.client()
        _FIREBASE_READY = True
        print("✅ Firebase initialized")
        return True
    except Exception as e:
        print(f"❌ Firebase init error: {e}")
        return False


def fetch_student_by_rfid(rfid: str) -> Optional[Dict[str, str]]:
    if not init_firebase_if_needed():
        return None
    try:
        doc = _FIREBASE_DB.collection("students").document(rfid).get()
        if not doc.exists:
            return None
        data = doc.to_dict()

        # Determine course for SHS (supports multiple possible field names)
        senior_high = None
        for key in ("Senior High School", "senior_high_school", "senior_high", "Strand", "strand"):
            if key in data and data.get(key):
                senior_high = str(data.get(key)).strip()
                break

        if senior_high:
            course_val = f"SHS {senior_high}"
        else:
            course_val = data.get("Course", data.get("Department", "Unknown Course"))

        gender_val = data.get("Gender", data.get("gender", "Unknown"))

        return {
            "student_id": data.get("Student Number", rfid),
            "name": data.get("Name", f"Student {rfid}"),
            "course": course_val,
            "gender": gender_val,
            "rfid": rfid,
        }
    except Exception as e:
        print(f"❌ Firebase read error: {e}")
        return None


def model_path_for_student(student: Dict[str, str]) -> Optional[str]:
    course = (student.get("course") or "").upper()
    gender = (student.get("gender") or "").lower()
    rfid = str(student.get("rfid", "")).strip()

    # Special RFID overrides (optional — extend as needed)
    # NOTE: 0095129433 is Arts and Science - should use arts and science.pt, not ict and eng.pt
    overrides = {
        "0095095703": "ict and eng.pt",
        # "0095129433": "ict and eng.pt",  # REMOVED - This is Arts and Science, should use course-based detection
        "0095272249": "bshm.pt",  # BSHM student
        "0095339862": "tourism male.pt",  # Tourism student
    }
    if rfid in overrides:
        return overrides[rfid]

    # SHS detection
    if any(k in course for k in ("SHS", "SENIOR HIGH", "SENIOR_HIGH", "STEM", "ABM", "HUMSS", "GAS", "TVL")):
        return "shs.pt"

    # ICT / ENG combined
    if any(k in course for k in ("ICT", "BSCPE", "ENG", "ICT AND ENG", "ICT/ENG", "COMPUTER ENGINEERING")):
        return "ict and eng.pt"

    # Arts & Science
    if any(k in course for k in ("ARTS", "SCIENCE", "ARTS AND SCIENCE", "ARTS/SCIENCE")):
        return "arts and science.pt"

    # Tourism
    if "TOURISM" in course:
        return "tourism male.pt"

    # BSHM / Hospitality Management
    if "BSHM" in course or "HOSPITALITY" in course or "HOSPITALITY MANAGEMENT" in course:
        return "bshm.pt"

    # HM
    if "HM" in course or "HUMANITIES" in course:
        return "hm.pt"

    # BSBA fallback by gender
    if "BSBA" in course or "BUSINESS" in course:
        return "bsba_female.pt" if gender.startswith("f") else "bsba male.pt"

    # Generic fallback
    return "bsba male.pt"


def open_camera() -> Optional[cv2.VideoCapture]:
    # Try DirectShow first on Windows
    backend_prefs = []
    if hasattr(cv2, "CAP_DSHOW"):
        backend_prefs.append(cv2.CAP_DSHOW)
    backend_prefs.append(cv2.CAP_ANY if hasattr(cv2, "CAP_ANY") else 0)

    for backend in backend_prefs:
        try:
            cap = cv2.VideoCapture(0, backend)
            if cap and cap.isOpened():
                try:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                return cap
            if cap:
                cap.release()
        except Exception:
            continue
    return None


def run_detection(model_path: str):
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return

    print(f"🧠 Loading model: {model_path}")
    model = YOLO(model_path)

    cap = open_camera()
    if not cap:
        print("❌ Cannot open camera.")
        return

    print("✅ Camera opened. Press 'q' to stop.")
    last_info = ""
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.02)
            continue

        results = model(frame, conf=0.35)

        annotated = frame
        if results and len(results) > 0 and hasattr(results[0], "boxes") and results[0].boxes is not None:
            boxes = results[0].boxes
            names = results[0].names
            for box in boxes:
                conf = float(box.conf[0].cpu().numpy())
                if conf < 0.35:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                name = names.get(cls, f"class_{cls}")
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(annotated, f"{name}", (int(x1), int(y1) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Small HUD
            try:
                total = len(boxes)
                info = f"Detections: {total}"
                if info != last_info:
                    print(info)
                    last_info = info
                cv2.putText(annotated, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            except Exception:
                pass

        cv2.imshow("Camera Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_detection_stream(model_path: str, frame_callback, stop_event=None, conf: float = 0.35, detected_callback=None):
    """Run detection and stream annotated frames via callback, without creating a window.
    - frame_callback(frame_bgr): called for each annotated frame
    - detected_callback(list_of_dicts|list_of_str): optional; called each processed frame with
      the list of detected class names (and confidences if available)
    - stop_event: threading.Event to allow caller to stop the loop
    - conf: confidence threshold
    """
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return

    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    cap = open_camera()
    if not cap:
        print("❌ Cannot open camera.")
        return

    try:
        while True:
            if stop_event is not None and getattr(stop_event, 'is_set', lambda: False)():
                break

            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            try:
                results = model(frame, conf=conf)
            except Exception:
                results = None

            annotated = frame
            detected_list = []
            try:
                if results and len(results) > 0 and hasattr(results[0], "boxes") and results[0].boxes is not None:
                    boxes = results[0].boxes
                    names = results[0].names
                    for box in boxes:
                        bconf = float(box.conf[0].cpu().numpy())
                        if bconf < conf:
                            continue
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        name = names.get(cls, f"class_{cls}")
                        detected_list.append({"class_name": name, "confidence": bconf, "class_id": cls})
                        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(annotated, f"{name}", (int(x1), int(y1) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            except Exception:
                detected_list = []

            try:
                frame_callback(annotated)
            except Exception:
                pass

            # Send detected classes to optional callback
            try:
                if detected_callback is not None:
                    detected_callback(detected_list)
            except Exception:
                pass

            # modest pacing to avoid starving UI thread
            time.sleep(0.01)
    finally:
        cap.release()

def main():
    print("\n🚀 Standalone Camera Detection (RFID-driven)")
    print("=================================================")
    print("Enter an RFID, the script will fetch student info from Firebase,")
    print("pick the correct .pt model, open the camera, and run detection.\n")

    while True:
        try:
            rfid = input("Tap RFID (or type 'exit'): ").strip()
        except EOFError:
            return
        if rfid.lower() == "exit":
            return
        if not rfid:
            continue

        student = fetch_student_by_rfid(rfid)
        if not student:
            print("⚠️ No student found for that RFID. Try again.")
            continue

        print(f"\n👤 Student: {student.get('name', 'Unknown')}")
        print(f"🏫 Course: {student.get('course', 'Unknown')}")
        print(f"🧬 Gender: {student.get('gender', 'Unknown')}")

        model_path = model_path_for_student(student)
        print(f"🎯 Using model: {model_path}")
        run_detection(model_path)
        print("\n📸 Ready for next RFID.\n")


if __name__ == "__main__":
    main()


