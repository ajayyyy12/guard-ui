import cv2
import time
import os
from collections import defaultdict
from ultralytics import YOLO
import threading
import random

# ---------- CONFIG ----------
MODEL_PATHS = {
    "BSBA_MALE": "bsba_male.pt",
    "BSBA_FEMALE": "bsba_female.pt"
}
CONF_THRESHOLD = 0.35
WINDOW_SECONDS = 5

# ✅ Uniform Parts
BSBA_MALE_WITH_BLAZER = {"black shoes", "blue long sleeve polo", "gray blazer", "gray pants", "red necktie"}
BSBA_MALE_NO_BLAZER = {"black shoes", "blue long sleeve polo", "gray pants", "red necktie"}

BSBA_FEMALE_SET = {"blue long sleeve polo", "close shoes", "gray blazer", "gray skirt", "red scarf"}
# ----------------------------

def get_detected_classes(results):
    names = []
    try:
        cls_indices = results[0].boxes.cls.cpu().numpy().astype(int).tolist()
        name_map = results[0].names
        names = [name_map[i] for i in cls_indices]
    except Exception as e:
        print("⚠️ Error reading detected classes:", e)
    return names

def simulate_rfid_tap():
    """Simulate an RFID tap (replace this later with actual RFID input)"""
    time.sleep(random.randint(5, 10))  # simulate waiting for a tap
    fake_users = [
        {"course": "BSBA", "gender": "Male"},
        {"course": "BSBA", "gender": "Female"},
    ]
    return random.choice(fake_users)

def detection_loop(shared_data):
    """Camera detection loop that switches models when RFID changes"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera.")
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    current_model = None
    current_user = None
    detected_parts = set()
    checking_active = False
    start_time = 0
    last_message_time = 0
    last_message_type = ""
    total_uniforms_seen = 0

    print("📸 System ready. Waiting for RFID tap...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame.")
            break

        # ✅ Check if RFID triggered a new user
        if shared_data["new_user"]:
            current_user = shared_data["user"]
            shared_data["new_user"] = False

            # Select model based on user
            key = f"{current_user['course'].upper()}_{current_user['gender'].upper()}"
            model_path = MODEL_PATHS.get(key)

            if model_path and os.path.exists(model_path):
                current_model = YOLO(model_path)
                print(f"\n👤 New user detected: {current_user['course']} - {current_user['gender']}")
                print(f"🎯 Model switched to: {model_path}")
            else:
                print(f"⚠️ Model not found for {key}")
                current_model = None

            detected_parts.clear()
            checking_active = False
            continue  # skip one frame to stabilize

        if current_model is None:
            cv2.putText(frame, "Waiting for RFID tap...", (10, 40), font, 0.8, (200,200,200), 2)
            cv2.imshow("Uniform Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Run YOLO detection
        results = current_model(frame, conf=CONF_THRESHOLD)
        detected_now = set(get_detected_classes(results))
        now = time.time()

        # Start a new detection cycle
        if detected_now:
            if not checking_active:
                checking_active = True
                start_time = now
                detected_parts.clear()
                print("\n➡️ New 5-second cycle started.")
            detected_parts.update(detected_now)

        # Detection cycle logic
        if checking_active:
            elapsed = now - start_time
            remaining = max(0, WINDOW_SECONDS - elapsed)
            seen_count = len(detected_parts)

            cv2.putText(frame, f"Detected: {seen_count} parts | Time left: {int(remaining)}s",
                        (10, 30), font, 0.7, (0,255,0), 2)

            # Determine uniform type
            if current_user["gender"] == "Male":
                has_blazer = BSBA_MALE_WITH_BLAZER.issubset(detected_parts)
                no_blazer = BSBA_MALE_NO_BLAZER.issubset(detected_parts)
                complete = has_blazer or no_blazer
            else:
                complete = BSBA_FEMALE_SET.issubset(detected_parts)

            if complete:
                total_uniforms_seen += 1
                print(f"\n✅ COMPLETE UNIFORM DETECTED for {current_user['gender']}! ({total_uniforms_seen} total)")
                cv2.putText(frame, "✅ COMPLETE UNIFORM", (10, frame.shape[0]-50), font, 1.0, (0,255,0), 2)
                checking_active = False
                detected_parts.clear()
                last_message_time = now
                last_message_type = "complete"

            elif elapsed >= WINDOW_SECONDS:
                print(f"\n⛔ INCOMPLETE UNIFORM for {current_user['gender']}")
                cv2.putText(frame, "⛔ INCOMPLETE UNIFORM", (10, frame.shape[0]-50), font, 1.0, (0,0,255), 2)
                checking_active = False
                detected_parts.clear()
                last_message_time = now
                last_message_type = "incomplete"

        elif now - last_message_time <= 3 and last_message_type:
            msg = "✅ COMPLETE UNIFORM" if last_message_type == "complete" else "⛔ INCOMPLETE UNIFORM"
            color = (0,255,0) if last_message_type == "complete" else (0,0,255)
            cv2.putText(frame, msg, (10, frame.shape[0]-50), font, 1.0, color, 2)
        else:
            cv2.putText(frame, f"Waiting... ({current_user['gender']})", (10, 30), font, 0.7, (200,200,200), 2)

        annotated_frame = results[0].plot()
        cv2.imshow("Uniform Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ============ MAIN CONTROLLER ============
def main():
    shared_data = {"user": None, "new_user": False}

    # Start camera loop in another thread
    camera_thread = threading.Thread(target=detection_loop, args=(shared_data,), daemon=True)
    camera_thread.start()

    # Simulate RFID reading loop
    while True:
        user = simulate_rfid_tap()  # Replace this with your actual RFID read
        shared_data["user"] = user
        shared_data["new_user"] = True
        print(f"\n🎟️ RFID tapped: {user}")
        time.sleep(1)

if __name__ == "__main__":
    main()
