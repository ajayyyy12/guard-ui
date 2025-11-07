import cv2
import csv
from collections import defaultdict, deque
from ultralytics import YOLO
import os

# ---------------- CONFIG ----------------
STAY_DISPLAY_SECONDS = 8     # time to show result
MIN_FRAME_COUNT = 8          # frames required to confirm a class

# --- Confidence thresholds ---
CONF_THRESHOLD_BSBA = 0.87
CONF_THRESHOLD_DEFAULT = 0.75  # fallback confidence
# ----------------------------------------

# ================= UNIFORM REQUIREMENTS =================
REQUIRED_PARTS = {
    "BSBA_MALE": [
        "black shoes",
        "blue long sleeve polo",
        "gray blazer",       # required
        "gray pants",
        "red necktie"
    ],
    "BSBA_FEMALE": [
        "close shoes",
        "blue long sleeve polo",
        "gray blazer",       # required
        "gray skirt",
        "red scarf"
    ]
}

# ================= MODEL PATHS =================
MODELS = {
    "BSBA_MALE": "bsba_male.pt",
    "BSBA_FEMALE": "bsba_female.pt"
}

# ================= CONFIDENCE MAP =================
CONF_THRESHOLDS = {
    "BSBA_MALE": CONF_THRESHOLD_BSBA,
    "BSBA_FEMALE": CONF_THRESHOLD_BSBA
}


def read_student_info(rfid_uid):
    """Read student info from students.csv"""
    if not os.path.exists("students.csv"):
        return None
    with open("students.csv", "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("uid") == rfid_uid:
                return row
    return None


def get_detected_classes(results, conf_threshold, frame_shape, min_box_px=40, min_box_area_ratio=0.002):
    """Extract valid class names from YOLO results"""
    try:
        if not results or len(results) == 0:
            return []

        res = results[0]
        boxes = getattr(res, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy().tolist()
        cls_indices = boxes.cls.cpu().numpy().astype(int).tolist()
        name_map = getattr(res, "names", {})

        frame_h, frame_w = frame_shape[0], frame_shape[1]
        frame_area = max(1, frame_w * frame_h)

        accepted = []
        for cidx, conf, box in zip(cls_indices, confs, xyxy):
            if conf < conf_threshold:
                continue

            x1, y1, x2, y2 = box
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            area = w * h

            if w < min_box_px or h < min_box_px:
                continue
            if (area / frame_area) < min_box_area_ratio:
                continue

            label = name_map.get(int(cidx), str(int(cidx)))
            accepted.append(label)

        return list(dict.fromkeys(accepted))
    except Exception as e:
        print("Detection filter error:", e)
        return []


def detect_uniform(model, required_parts, cap, conf_threshold):
    """Continuously detect until all required uniform parts are seen"""
    detection_history = defaultdict(lambda: deque(maxlen=MIN_FRAME_COUNT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf_threshold)
        names = get_detected_classes(results, conf_threshold, frame.shape)

        for n in names:
            detection_history[n].append(True)

        confirmed_parts = {p for p, times in detection_history.items() if len(times) >= MIN_FRAME_COUNT}

        annotated = results[0].plot() if results and hasattr(results[0], "plot") else frame
        display = annotated.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX

        y = 40
        for p in required_parts:
            color = (0, 255, 0) if p in confirmed_parts else (0, 0, 255)
            mark = "✔" if p in confirmed_parts else "✘"
            cv2.putText(display, f"{mark} {p}", (10, y), font, 0.6, color, 2)
            y += 25

        cv2.imshow("Uniform Detection", display)

        # ✅ Complete if all required parts are confirmed
        if all(p in confirmed_parts for p in required_parts):
            return "complete"

        # ❌ Press 'q' to manually stop and count as violation
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return "violation"

    return None


def show_result_message(cap, text, color):
    """Display final message for few seconds"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    start_time = time.time()
    while time.time() - start_time < STAY_DISPLAY_SECONDS:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(frame, text, (40, 100), font, 1.0, color, 3, cv2.LINE_AA)
        cv2.imshow("Uniform Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera.")
        return

    print("📸 Camera live. Waiting for RFID tap...")

    while True:
        rfid_uid = input("\nTap RFID UID (or type 'exit' to quit): ").strip()
        if rfid_uid.lower() == "exit":
            break

        student = read_student_info(rfid_uid)
        if not student:
            print("⚠️ No student found for that RFID. Please enroll first.")
            continue

        name = student.get("name", "Unknown")
        gender = student.get("gender", "Unknown").upper()
        course = student.get("course", "Unknown").upper()

        print(f"\n🎓 Student: {name}")
        print(f"🧬 Gender: {gender}")
        print(f"🏫 Course: {course}")

        key = f"{course}_{gender}"
        if key not in MODELS:
            print("⚠️ Unsupported course. Only BSBA students are allowed.")
            continue

        model_path = MODELS[key]
        conf_threshold = CONF_THRESHOLDS.get(key, CONF_THRESHOLD_DEFAULT)
        required_parts = REQUIRED_PARTS.get(key, [])

        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            continue

        model = YOLO(model_path)
        print(f"\n🕵️ Detecting uniform for {key}... (Press 'q' to stop)")

        result = detect_uniform(model, required_parts, cap, conf_threshold)

        if result == "complete":
            print("✅ COMPLETE UNIFORM")
            show_result_message(cap, "✅ COMPLETE UNIFORM", (0, 200, 0))
        elif result == "violation":
            print("❌ INCOMPLETE UNIFORM — Missing blazer or required parts.")
            show_result_message(cap, "❌ INCOMPLETE UNIFORM", (0, 0, 255))
        else:
            print("⚠️ Detection interrupted.")

        print("\n📸 Ready for next student RFID tap...\n")

    cap.release()
    cv2.destroyAllWindows()
    print("👋 System closed.")


if __name__ == "__main__":
    main()
