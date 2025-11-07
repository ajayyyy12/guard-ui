# ...existing code...
import cv2, time
import detection1

print("Closing other camera-using apps (make sure Windows Camera/Teams/Chrome are closed).")
svc = detection1.get_detection_service()

print("Opening camera index 0 with DirectShow...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print("cap.isOpened():", cap.isOpened())

# warm up
for _ in range(30):
    try:
        cap.grab()
    except Exception:
        pass
    time.sleep(0.03)

ret, _ = cap.read()
print("first read ok:", bool(ret))

ok = svc.set_existing_camera(cap)
print("set_existing_camera returned:", ok)

# wait briefly for capture thread to populate last_frame
time.sleep(1.0)
with svc.capture_lock:
    print("svc.last_frame available:", svc.last_frame is not None)

# keep attached for 2s so you can open the GUI afterwards if needed
time.sleep(2.0)
# ...existing code...