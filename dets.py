# detection_system_pi_optimized.py
# YOLOv8 + OpenCV optimized for Raspberry Pi 4 Model B

from ultralytics import YOLO
import cv2
import time
import sys

class DetectionSystem:
    def __init__(self, model_path="male ict.pt", conf_threshold=0.5, iou_threshold=0.45, cam_index=0):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.cam_index = cam_index

        print(f"📦 Loading YOLO model: {self.model_path}")
        try:
            # Don't fuse for Pi — uses less memory
            self.model = YOLO(self.model_path)
            print("✅ Model loaded successfully.")
        except Exception as e:
            print(f"❌ Failed to load YOLO model: {e}")
            sys.exit(1)

        # Initialize webcam (Pi camera or USB)
        try:
            self.cap = cv2.VideoCapture(self.cam_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            if not self.cap.isOpened():
                raise RuntimeError("Camera not detected or cannot be opened.")
            print("🎥 Camera started successfully.")
        except Exception as e:
            print(f"❌ Failed to initialize camera: {e}")
            sys.exit(1)

        self.prev_time = time.time()
        self.fps = 0
        self.frame_skip = 2  # process every 2nd frame for speed

    def draw_fps(self, frame):
        """Display FPS on screen."""
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    def run(self):
        print("🚀 Starting detection... Press 'q' to quit.")
        time.sleep(1)
        frame_count = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("⚠️ Failed to grab frame from camera.")
                continue

            # Resize to smaller resolution for faster detection
            small_frame = cv2.resize(frame, (320, 240))

            frame_count += 1
            if frame_count % self.frame_skip != 0:
                # Skip frames to save CPU
                continue

            start_time = time.time()

            # Run YOLO detection (low resolution for speed)
            results = self.model.predict(
                small_frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,
                imgsz=320  # reduce model input size
            )

            # Draw detections (rescale boxes to original frame)
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = f"{self.model.names[cls]} {conf:.2f}"

                    # Scale boxes from 320x240 back to 640x480
                    scale_x = frame.shape[1] / 320
                    scale_y = frame.shape[0] / 240
                    x1 = int(x1 * scale_x)
                    y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x)
                    y2 = int(y2 * scale_y)

                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # FPS calculation
            end_time = time.time()
            self.fps = 1 / (end_time - self.prev_time)
            self.prev_time = end_time
            self.draw_fps(frame)

            cv2.imshow("Raspberry Pi YOLO Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("🛑 Exiting detection loop.")
                break

        self.cap.release()
        cv2.destroyAllWindows()
        print("📷 Camera stopped. Resources released.")


if __name__ == "__main__":
    detector = DetectionSystem(model_path="male ict.pt", conf_threshold=0.5)
    detector.run()
