import cv2, time, os

BACKENDS = [
    ("ANY", cv2.CAP_ANY),
    ("MSMF", getattr(cv2, "CAP_MSMF", None)),
    ("DSHOW", getattr(cv2, "CAP_DSHOW", None)),
    ("VFW", getattr(cv2, "CAP_VFW", None)),
    ("WINRT", getattr(cv2, "CAP_WINRT", None)),
]

MAX_INDEX = 15
RESOLUTIONS = [(1920,1080),(1280,720),(640,480),(320,240)]
out_dir = os.getcwd()

print("Camera probe starting...\n")
found_any = False

for backend_name, backend_flag in BACKENDS:
    print(f"--- Testing backend: {backend_name} ---")
    for idx in range(0, MAX_INDEX + 1):
        try:
            cap = cv2.VideoCapture(idx) if backend_flag is None else cv2.VideoCapture(idx, backend_flag)
        except Exception as e:
            print(f"Index {idx}: ERROR opening with backend {backend_name}: {e}")
            continue

        time.sleep(0.25)
        try:
            if cap is None:
                print(f"Index {idx}: cap is None")
            else:
                opened = cap.isOpened()
                if not opened:
                    print(f"Index {idx}: not opened (isOpened()=False)")
                else:
                    # try default read
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        h, w = frame.shape[:2]
                        fname = os.path.join(out_dir, f"probe_idx{idx}_{backend_name}_{w}x{h}.jpg")
                        cv2.imwrite(fname, frame)
                        print(f"Index {idx}: OPENED and read frame ({w}x{h}) with backend {backend_name} -> saved {fname}")
                        found_any = True
                    else:
                        # try common resolutions
                        success = False
                        for w,h in RESOLUTIONS:
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(w))
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(h))
                            time.sleep(0.08)
                            for _ in range(12):
                                try: cap.grab()
                                except: pass
                                time.sleep(0.02)
                            try:
                                ret2, frame2 = cap.read()
                            except Exception:
                                ret2, frame2 = False, None
                            if ret2 and frame2 is not None:
                                hh, ww = frame2.shape[:2]
                                fname = os.path.join(out_dir, f"probe_idx{idx}_{backend_name}_{ww}x{hh}.jpg")
                                cv2.imwrite(fname, frame2)
                                print(f"Index {idx}: OPENED at {ww}x{hh} with backend {backend_name} -> saved {fname}")
                                found_any = True
                                success = True
                                break
                        if not success:
                            print(f"Index {idx}: opened but failed to read a usable frame with backend {backend_name}")
        except Exception as e:
            print(f"Index {idx}: exception during read: {e}")
        finally:
            try: cap.release()
            except: pass
        time.sleep(0.08)
    print("")

if not found_any:
    print("No working camera index/backend combination found in probe (tried indices 0..15).")
else:
    print("At least one camera index/backend combination was able to open and read a frame.")

print("Camera probe finished.")