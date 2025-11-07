import cv2
from ultralytics import YOLO

def main():
    # Load your trained YOLO model
    model = YOLO("shs.pt")

    # Open the default camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Cannot open camera.")
        return

    print("✅ Camera opened. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame.")
            break 

        # Perform detection
        results = model(frame, conf=0.65)

        # Draw bounding boxes
        annotated_frame = results[0].plot()

        # Show output
        cv2.imshow("AI-Niform Detection", annotated_frame)

        # Quit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 