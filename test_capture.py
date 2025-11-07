import cv2, time, os
backends = [(cv2.CAP_DSHOW,'DSHOW'), (cv2.CAP_MSMF,'MSMF'), (cv2.CAP_ANY,'ANY')]
for backend, name in backends:
    print('---', name, '---')
    cap = cv2.VideoCapture(0, backend)
    print('isOpened:', cap.isOpened())
    if not cap.isOpened():
        try:
            cap.release()
        except Exception:
            pass
        continue
    # warm-up grabs
    for _ in range(30):
        try:
            cap.grab()
        except Exception:
            pass
        time.sleep(0.03)
    try:
        ret, frame = cap.read()
    except Exception as e:
        ret, frame = False, None
        print('read exception:', e)
    print('read_ok:', bool(ret))
    if ret and frame is not None:
        fname = os.path.join(os.getcwd(), f'test_frame_{name}.jpg')
        cv2.imwrite(fname, frame)
        print('saved:', fname)
    cap.release()