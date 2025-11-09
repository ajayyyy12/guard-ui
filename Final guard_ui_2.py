                                                                                                                                                                                                                    # guard_ui_2.py
# Simplified and robust Guard Control Center

import time
import traceback
import threading
import tkinter as tk
import platform
from tkinter import ttk
from tkinter import messagebox
from tkinter import simpledialog
import os
from collections import defaultdict, deque
import datetime
from datetime import timedelta
# Ensure `datetime.now()` convenience exists (some code calls datetime.now())
try:
    datetime.now = datetime.datetime.now
except Exception:
    pass

# CV2 availability detection
try:
    import cv2
    CV2_AVAILABLE = True
    CV2_MODULE = cv2
except Exception:
    CV2_AVAILABLE = False
    CV2_MODULE = None

# Local defaults
MIN_FRAME_COUNT = 3  # number of frames required to confirm a detected uniform part
# Detect Firebase availability: try importing firebase_admin and locating service account
FIREBASE_AVAILABLE = False
FIREBASE_CRED_PATH = None
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    # look for common credential filenames in script dir or cwd
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    potential = [
        os.path.join(script_dir, 'serviceAccountKey.json'),
        os.path.join(script_dir, 'firebase_service_account.json'),
        os.path.join(os.getcwd(), 'serviceAccountKey.json'),
        os.path.join(os.getcwd(), 'firebase_service_account.json'),
    ]
    for p in potential:
        if os.path.exists(p):
            FIREBASE_AVAILABLE = True
            FIREBASE_CRED_PATH = p
            break
except Exception:
    FIREBASE_AVAILABLE = False
    FIREBASE_CRED_PATH = None

# PIL/Pillow availability detection for image loading
PIL_AVAILABLE = False
try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️ PIL/Pillow not available. Install with: pip install Pillow")

CONF_THRESHOLD_BSBA = 0.27
CONF_THRESHOLD_ICT = 0.2
CONF_THRESHOLD_DEFAULT = 0.35  # fallback confidence

REQUIRED_PARTS = {
    # ==== BSBA ====
    "BSBA_MALE": [
        "black shoes",
        "blue long sleeve polo",
        "gray blazer",
        "gray pants",
        "red necktie"
    ],
    "BSBA_MALE_NO_BLAZER": [
        "black shoes",
        "blue long sleeve polo",
        "gray pants",
        "red necktie"
    ],
    "BSBA_FEMALE": [
        "close shoes",
        "blue long sleeve polo",
        "gray blazer",
        "gray skirt",
        "red scarf"
    ],
    "BSBA_FEMALE_NO_BLAZER": [
        "close shoes",
        "blue long sleeve polo",
        "gray skirt",
        "red scarf"
    ],

    # ==== ICT ====
    "ICT_MALE": [
        "black shoes",
        "ict gray pants",
        "ict polo"
    ],

    # ==== BSCPE (fallback uses ICT model currently) ====
    "BSCPE_MALE": [
        "black shoes",
        "ict gray pants",
        "ict polo"
    ]
}

# --- Map course type to model ---
MODELS = {
    "BSBA_MALE": "bsba male.pt",
    "BSBA_FEMALE": "bsba_female.pt",
    "ICT_MALE": "bsba male.pt",  # Use BSBA male model for ICT
    "BSCPE_MALE": "bsba male.pt",  # Use BSBA male model for BSCPE
    "ICT_FEMALE": "bsba_female.pt",  # Use BSBA female model for ICT
    "BSCPE_FEMALE": "bsba_female.pt",  # Use BSBA female model for BSCPE
}

# --- Map course type to confidence ---
CONF_THRESHOLDS = {
    "BSBA_MALE": CONF_THRESHOLD_BSBA,
    "BSBA_FEMALE": CONF_THRESHOLD_BSBA,
    "ICT_MALE": CONF_THRESHOLD_ICT,
    "BSCPE_MALE": CONF_THRESHOLD_ICT,
}

def get_detected_classes(results, conf_threshold, frame_shape, min_box_px=40, min_box_area_ratio=0.002):
    """
    Return list of class names that pass confidence and size filters.
    - conf_threshold: confidence cutoff
    - frame_shape: (height, width, channels)
    - min_box_px: minimum width/height in pixels to accept
    - min_box_area_ratio: min box area ratio of frame area (very small boxes ignored)
    """
    try:
        if not results or len(results) == 0:
            return []

        res = results[0]
        boxes = getattr(res, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        # get arrays
        xyxy = boxes.xyxy.cpu().numpy()        # shape (N,4)
        confs = boxes.conf.cpu().numpy().tolist()
        cls_indices = boxes.cls.cpu().numpy().astype(int).tolist()
        name_map = getattr(res, "names", {})

        frame_h, frame_w = frame_shape[0], frame_shape[1]
        frame_area = max(1, frame_w * frame_h)

        accepted = []
        for idx, (cidx, conf, box) in enumerate(zip(cls_indices, confs, xyxy)):
            if conf < conf_threshold:
                continue

            x1, y1, x2, y2 = box
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            area = w * h

            # filter tiny boxes
            if w < min_box_px or h < min_box_px:
                continue
            if (area / frame_area) < min_box_area_ratio:
                continue

            # map class index to name (safely)
            label = name_map.get(int(cidx), str(int(cidx)))
            accepted.append(label)

        # return unique labels (set) but keep as list
        return list(dict.fromkeys(accepted))
    except Exception as e:
        # don't crash on unexpected result layout
        print("Detection filter error:", e)
        return []

# Detection System Class (integrated from detected.py)
class ImprovedUniformTracker:
    """Improved uniform tracking with course-specific requirements and frame counting - no timeout"""
    
    def __init__(self, course_type="BSBA_MALE"):
        self.course_type = course_type
        self.required_parts = REQUIRED_PARTS.get(course_type, [])
        self.detection_history = defaultdict(lambda: deque(maxlen=MIN_FRAME_COUNT))
        self.confirmed_parts = set()
        self.start_time = time.time()
        self.last_update_time = time.time()
        self.stable_detection_threshold = 0.15  # Minimum time between updates for stability
    
    def reset(self, course_type="BSBA_MALE"):
        """Reset tracking for new detection session"""
        self.course_type = course_type
        self.required_parts = REQUIRED_PARTS.get(course_type, [])
        self.detection_history = defaultdict(lambda: deque(maxlen=MIN_FRAME_COUNT))
        self.confirmed_parts = set()
        self.start_time = time.time()
        self.last_update_time = time.time()
    
    def update_detections(self, detected_classes, current_time=None):
        """Update tracking based on detected class names with frame counting"""
        if current_time is None:
            current_time = time.time()
        
        # Only update if enough time has passed for stability
        if current_time - self.last_update_time < self.stable_detection_threshold:
            return self.is_complete(), self.get_status_text()
        
        self.last_update_time = current_time
        
        # Add current detections to history
        for class_name in detected_classes:
            self.detection_history[class_name].append(current_time)
        
        # Update confirmed parts (seen in MIN_FRAME_COUNT frames)
        self.confirmed_parts = {
            part for part, times in self.detection_history.items() 
            if len(times) >= MIN_FRAME_COUNT
        }
        
        return self.is_complete(), self.get_status_text()
    
    def is_complete(self):
        """Check if all required parts are confirmed"""
        return all(part in self.confirmed_parts for part in self.required_parts)
    
    def get_status_text(self):
        """Get current status text"""
        if self.is_complete():
            return "COMPLETE UNIFORM"
        else:
            missing_parts = [part for part in self.required_parts if part not in self.confirmed_parts]
            return f"Missing: {', '.join(missing_parts)}"
    
    def get_missing_components(self):
        """Get list of components not yet confirmed"""
        return [part for part in self.required_parts if part not in self.confirmed_parts]
    
    def get_elapsed_time(self):
        """Get elapsed time since detection started"""
        return time.time() - self.start_time

# Legacy class for backward compatibility
class BSBAUniformTracker(ImprovedUniformTracker):
    """Legacy BSBA uniform tracker - now uses improved system"""
    
    def __init__(self):
        super().__init__("BSBA_MALE")

# Note: All detection functions and variables are defined in this file

# Note: DetectionSystem class is defined below in this file

# Import YOLO if available
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("✅ YOLO imported successfully")
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ YOLO not available - using simulation mode")
    # Create a dummy YOLO class for compatibility
    class YOLO:
        def __init__(self, model_path):
            self.model_path = model_path
            print(f"🔧 Dummy YOLO model loaded: {model_path}")
        
        def __call__(self, frame, conf=0.5):
            # Return dummy results
            class DummyResults:
                def __init__(self):
                    self.boxes = None
            return [DummyResults()]

class DetectionSystem:
    def __init__(self, model_path="bsba_female.pt", conf_threshold=0.65, iou_threshold=0.15, cam_index=0):
        print(f"🔧 Initializing DetectionSystem with model: {model_path}")
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.cam_index = cam_index
        
        # Detection state
        self.detection_active = False
        self.detection_thread = None
        self.current_person_id = None
        self.current_person_name = None
        self.current_person_type = None
        
        # UI callback for updating camera feed
        self.ui_callback = None
        
        # Initialize detection service
        try:
            import sys
            import os
            # Add current directory to Python path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.append(current_dir)
            
            from detection1 import UniformDetectionService
            self.detection_service = UniformDetectionService(
                model_path=self.model_path, 
                conf_threshold=self.conf_threshold
            )
            # remember which model path is loaded to avoid re-loading unnecessarily
            try:
                self.detection_service._model_path = self.model_path
            except Exception:
                pass
            print(f"✅ DetectionSystem initialized with detection service")
        except Exception as e:
            print(f"❌ Failed to initialize detection service: {e}")
            self.detection_service = None

    def load_model(self):
        """Load detection service with model"""
        try:
            import sys
            import os
            # Add current directory to Python path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.append(current_dir)
            
            # If detection_service already exists try to switch model in-place
            from detection1 import UniformDetectionService
            if hasattr(self, 'detection_service') and self.detection_service is not None:
                try:
                    loaded = getattr(self.detection_service, '_model_path', None)
                    if loaded == self.model_path:
                        print(f"🔁 Model already loaded: {self.model_path} - skipping reload")
                        return True
                    # Switch model without rebuilding or touching camera
                    if hasattr(self.detection_service, 'switch_model'):
                        if self.detection_service.switch_model(self.model_path):
                            print(f"✅ Switched model in-place: {self.model_path}")
                            return True
                except Exception as e:
                    print(f"⚠️ In-place model switch failed, rebuilding service: {e}")

            # Create a new detection service instance with the specified model
            self.detection_service = UniformDetectionService(
                model_path=self.model_path, 
                conf_threshold=self.conf_threshold
            )
            try:
                self.detection_service._model_path = self.model_path
            except Exception:
                pass
            print(f"✅ Detection service loaded with model: {self.model_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to load detection service: {e}")
            return False

    def start_live_feed(self, ui_callback=None):
        """Start live camera feed using detection service"""
        self.ui_callback = ui_callback
        # If detection_service isn't yet created, try to load/create it now.
        if not getattr(self, 'detection_service', None):
            print("🔧 Detection service not initialized - attempting to create/load it now")
            try:
                if not self.load_model():
                    print("❌ Detection service not available after load attempt")
                    return False
            except Exception as e:
                print(f"❌ Exception while attempting to initialize detection service: {e}")
                return False
        # Load model
        try:
            if not self.load_model():
                print("❌ Failed to load detection model")
                return False
        except Exception as e:
            print(f"❌ Exception while loading model: {e}")
            import traceback; traceback.print_exc()
            return False

        # Start camera - try multiple camera indices to be resilient
        camera_started = False
        try:
            for cam_idx in range(0, 4):
                try:
                    if self.detection_service.start_camera(cam_idx):
                        camera_started = True
                        print(f"✅ Detection camera started on index {cam_idx}")
                        break
                except Exception as e:
                    print(f"⚠️ start_camera failed on index {cam_idx}: {e}")
            if not camera_started:
                print("❌ Failed to start camera on any tested index")
                return False
        except Exception as e:
            print(f"❌ Exception while starting camera: {e}")
            import traceback; traceback.print_exc()
            return False
            
        # Start detection loop
        self.detection_active = True
        self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.detection_thread.start()
        
        print(f"✅ Live camera feed started with clean detection")
        return True

    def start_detection(self, person_id, person_name, person_type, ui_callback=None):
        """Start detection for a person using detection service"""
        self.current_person_id = person_id
        self.current_person_name = person_name
        self.current_person_type = person_type
        self.ui_callback = ui_callback
        
        # Ensure detection service is ready; try to create/load if missing
        if not getattr(self, 'detection_service', None):
            print("🔧 Detection service not initialized - attempting to create/load it now")
            try:
                if not self.load_model():
                    print("❌ Detection service not available after load attempt")
                    return False
            except Exception as e:
                print(f"❌ Exception while attempting to initialize detection service: {e}")
                return False
        
        # Determine the correct model path based on person type and gender
        if hasattr(self, 'main_ui') and self.main_ui:
            model_path = self.main_ui._get_model_path_for_person(person_id, person_type)
            if model_path != self.model_path:
                self.model_path = model_path
                print(f"🔄 Switching to model: {model_path}")
            
        # Load model
        try:
            if not self.load_model():
                print("❌ Failed to load detection model")
                return False
        except Exception as e:
            print(f"❌ Exception while loading model: {e}")
            import traceback; traceback.print_exc()
            return False

        # Prefer reusing an existing guard camera to avoid reopen latency
        try:
            if hasattr(self, 'main_ui') and self.main_ui:
                guard_cap = getattr(self.main_ui, 'guard_camera_cap', None)
                if guard_cap is not None:
                    try:
                        if hasattr(self.detection_service, 'set_existing_camera'):
                            if self.detection_service.set_existing_camera(guard_cap):
                                try:
                                    self.main_ui._cap_handed_to_detection = True
                                except Exception:
                                    pass
                                self.main_ui.guard_camera_cap = None
                                print("🔁 Handed guard camera to detection (zero reopen)")
                    except Exception as e:
                        print(f"⚠️ Failed to handover guard camera: {e}")
        except Exception:
            pass

        # Start camera if not already using an existing one
        camera_started = False
        try:
            cap_obj = getattr(self.detection_service, 'cap', None)
            if cap_obj is not None:
                try:
                    if cap_obj.isOpened():
                        camera_started = True
                        print("✅ Using existing opened camera for detection")
                except Exception:
                    camera_started = False
            if not camera_started:
                for cam_idx in range(0, 2):
                    try:
                        if self.detection_service.start_camera(cam_idx):
                            camera_started = True
                            print(f"✅ Detection camera started on index {cam_idx}")
                            break
                    except Exception as e:
                        print(f"⚠️ start_camera failed on index {cam_idx}: {e}")
            if not camera_started:
                print("❌ Failed to start camera")
                return False
        except Exception as e:
            print(f"❌ Exception while starting camera: {e}")
            import traceback; traceback.print_exc()
            return False
            
        # Start detection loop
        self.detection_active = True
        self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.detection_thread.start()
        
        # Attach current RFID/person id into service for popups/workflows
        try:
            if hasattr(self.detection_service, 'current_rfid'):
                self.detection_service.current_rfid = person_id
        except Exception:
            pass

        print(f"✅ Detection started for {person_name} with clean detection")
        return True

    def stop_detection(self):
        """Stop detection and release camera control"""
        self.detection_active = False

        # Attempt to stop detection loop first
        try:
            if self.detection_service:
                try:
                    self.detection_service.stop_detection()
                except Exception as e:
                    print(f"⚠️ Error stopping detection service: {e}")

                # Try to release camera control from detection service
                returned_cap = None
                try:
                    if hasattr(self.detection_service, 'release_camera_control'):
                        returned_cap = self.detection_service.release_camera_control()
                        print("✅ Detection service released camera control")
                except Exception as e:
                    print(f"⚠️ Error releasing camera control from detection service: {e}")

                # If we received a VideoCapture back, hand it to main UI guard feed under lock
                if returned_cap is not None:
                    try:
                        if hasattr(self, 'main_ui') and self.main_ui:
                            with getattr(self.main_ui, '_guard_cap_lock', threading.Lock()):
                                self.main_ui.guard_camera_cap = returned_cap
                                try:
                                    self.main_ui._cap_handed_to_detection = False
                                except Exception:
                                    pass
                            print("🔁 Camera VideoCapture returned to main UI guard feed")
                    except Exception as e:
                        print(f"⚠️ Error assigning returned cap to main UI: {e}")
                else:
                    # If detection did not return a cap, ensure camera is stopped to free resources
                    try:
                        if hasattr(self.detection_service, 'stop_camera'):
                            self.detection_service.stop_camera()
                    except Exception as e:
                        print(f"⚠️ Error stopping detection camera fallback: {e}")
        except Exception as e:
            print(f"⚠️ Unexpected error while stopping detection: {e}")

    # Detection stopped (silent) - no noisy message
    
    def reset_detection_history(self):
        """Reset detection history for new detection session"""
        pass
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            self.stop_detection()
            print("✅ DetectionSystem cleanup completed")
        except Exception as e:
            print(f"❌ Error during DetectionSystem cleanup: {e}")

    def _detection_loop(self):
        """Clean detection loop using detection service - only bounding boxes and class names"""
        try:
            print("🔍 Starting clean detection loop - only bounding boxes and class names")
            frame_count = 0
            
            for detection_result in self.detection_service.get_detection_loop():
                if not self.detection_active:
                    break
                    
                frame_count += 1
                if frame_count % 30 == 0:
                    try:
                        print(f"🔍 Frame {frame_count}: processing...")
                    except Exception:
                        pass
                
                # Get clean annotated frame (only bounding boxes and class names)
                annotated_frame = detection_result['annotated_frame']
                detected_classes = detection_result['detected_classes']
                
                # Check if BSCPE verification is complete or incomplete
                if (hasattr(self.detection_service, 'bscpe_tracker') and 
                    self.detection_service.bscpe_tracker):
                    print(f"🔍 BSCPE tracker exists: {self.detection_service.bscpe_tracker}")
                    verification_status, verification_message = self.detection_service.bscpe_tracker.get_verification_status()
                    print(f"🔍 Checking BSCPE status: {verification_status} - {verification_message}")
                    
                    # Check elapsed time for debugging
                    elapsed_time = time.time() - self.detection_service.bscpe_tracker.start_time
                    print(f"🔍 Elapsed time: {elapsed_time:.1f}s")
                    
                    if verification_status in ["COMPLETE"]:
                        print(f"🎯 BSCPE verification {verification_status} - recording entry and stopping detection")
                        # Record entry when complete
                        if hasattr(self, 'main_ui') and self.main_ui:
                            # Get student info for entry recording
                            student_info = self.main_ui.get_student_info_by_rfid(getattr(self.detection_service, 'current_rfid', None))
                            if student_info:
                                self.main_ui.record_complete_uniform_entry(getattr(self.detection_service, 'current_rfid', None), student_info)
                                # Show complete uniform popup on main screen
                                self.main_ui.show_complete_uniform_popup(student_info)
                        # Stop detection
                        self.stop_detection()
                        if hasattr(self, 'main_ui') and self.main_ui:
                            self.main_ui.initialize_guard_camera_feed()
                        break
                    elif verification_status == "INCOMPLETE":
                        print(f"🎯 BSCPE verification {verification_status} - marking INCOMPLETE (will remain in detection mode)")
                        print("💡 Student can tap their ID again to retry uniform detection")
                        # Show missing uniform popup on main screen immediately (do not stop detection)
                        if hasattr(self, 'main_ui') and self.main_ui:
                            print(f"🔍 Main UI available, getting student info...")
                            # Get student info for popup
                            student_info = self.main_ui.get_student_info_by_rfid(getattr(self.detection_service, 'current_rfid', None))
                            print(f"🔍 Student info retrieved: {student_info}")
                            if student_info:
                                print(f"🔍 Calling show_incomplete_uniform_popup...")
                                try:
                                    self.main_ui.show_incomplete_uniform_popup(student_info)
                                except Exception:
                                    pass
                            else:
                                print(f"⚠️ No student info found for RFID: {getattr(self.detection_service, 'current_rfid', None)}")
                        else:
                            print(f"⚠️ Main UI not available")

                        # Instead of stopping detection, reset the BSCPE tracker so the detection
                        # loop can continue and allow the student to retry without restarting camera.
                        try:
                            if hasattr(self.detection_service, 'bscpe_tracker') and self.detection_service.bscpe_tracker:
                                try:
                                    self.detection_service.bscpe_tracker.reset(self.detection_service.bscpe_tracker.course_type)
                                    # reset the start time so watchdog/window timers start fresh
                                    self.detection_service.bscpe_tracker.start_time = time.time()
                                    print("🔄 BSCPE tracker reset - detection remains active for retry")
                                except Exception as e:
                                    print(f"⚠️ Failed to reset BSCPE tracker: {e}")
                        except Exception:
                            pass
                        # Continue processing (do not break the loop)
                else:
                    print(f"🔍 No BSCPE tracker found - detection_service: {hasattr(self.detection_service, 'bscpe_tracker')}, tracker: {getattr(self.detection_service, 'bscpe_tracker', None)}")
                
                # Update UI with clean frame
                if self.ui_callback and annotated_frame is not None:
                    self.ui_callback(annotated_frame)
                
                # Also update detected classes list in the UI if available.
                # Only update when this frame was actually processed (not skipped)
                # and when we have at least one detection to show.
                try:
                    if (
                        hasattr(self, 'main_ui') and self.main_ui and 
                        hasattr(self.main_ui, 'update_detected_classes_list')
                    ):
                        if not detection_result.get('skipped') and detected_classes:
                            self.main_ui.update_detected_classes_list(detected_classes)
                except Exception:
                    pass
                
                # Print detected classes to console (not on camera)
                if detected_classes and frame_count % 60 == 0:
                    try:
                        class_names = [d['class_name'] for d in detected_classes]
                        confidences = [f"{d['confidence']:.2f}" for d in detected_classes]
                        print(f"🔍 Detected: {class_names} (conf: {confidences})")
                    except Exception:
                        pass
                    
        except Exception as e:
            print(f"❌ Error in detection loop: {e}")
        finally:
            print("🛑 Clean detection loop ended")
            # Release camera control back to guard camera
            if hasattr(self.detection_service, 'release_camera_control'):
                self.detection_service.release_camera_control()
                print("✅ Camera control released back to guard camera")

class GuardMainControl:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🛡️ AI-niform - Guard Control Center")
        # Expose firebase and cv2 modules (if loaded at module-level autodetect)
        self.firebase_admin = globals().get('firebase_admin', None)
        self.FIREBASE_CRED_PATH = globals().get('FIREBASE_CRED_PATH', None)
        self.cv2_available = globals().get('CV2_AVAILABLE', False)
        self.cv2 = globals().get('CV2_MODULE', None)
        
        # Configure Guard window for primary monitor (fullscreen)
        self.root.geometry("1200x900")
        self.root.configure(bg='#ffffff')
        self.root.minsize(1000, 800)  # Set minimum window size
        
        # Position on primary monitor and make fullscreen
        # Detected-classes UI state
        self._last_detected_classes = []
        self._last_detect_time = 0.0
        try:
            # periodic staleness check (clears list if no detections for a while)
            self.root.after(700, self._refresh_detected_classes_view)
        except Exception:
            pass

        # Uniform requirements configuration (normalized to lowercase)
        self.uniform_requirements_presets = {
            'SHS': [
                'black shoes',
                'necktie',
                'pants',
                'polo with vest',
            ],
            'BSHM_MALE': [
                'black shoes',
                'blazer',
                'white long sleeve',
                'rtw pants',
            ],
            'BSHM_FEMALE': [
                'closed shoes',
                'blazer',
                'white long sleeve',
                'rtw pants',
                'rtw skirt',
            ],
            'BSCPE_MALE': [
                'black shoes',
                'gray polo',
                'rtw pants',
            ],
            'ARTS_AND_SCIENCE': [
                'Blazer',
                'Blue Long Sleeve',
                'Blue Scarf',
                'Close shoes',
                'Skirt',
            ],
            'TOURISM': [
                'black shoes',
                'Blazer',
                'pants',
                'tourism pin',
                'white polo',
                'yellow necktie',
            ]
        }
        self.current_uniform_key = 'SHS'
        self.current_uniform_requirements = list(self.uniform_requirements_presets.get(self.current_uniform_key, []))
        self._requirements_present = set()
        # Track detection counts and permanently checked requirements
        self._requirement_detection_counts = {}  # Maps requirement (lowercase) -> count
        self._permanently_checked_requirements = set()  # Requirements detected twice
        try:
            # Get screen dimensions for fullscreen
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            
            print(f"🖥️ Screen dimensions: {screen_width}x{screen_height}")
            
            # Debug execution environment
            import os
            import sys
            print(f"DEBUG: Python executable: {sys.executable}")
            print(f"DEBUG: Python version: {sys.version}")
            print(f"DEBUG: Current working directory: {os.getcwd()}")
            print(f"DEBUG: Script directory: {os.path.dirname(os.path.abspath(__file__))}")
            print(f"DEBUG: Command line arguments: {sys.argv}")
            
            # Set geometry to fullscreen on primary monitor
            self.root.geometry(f"{screen_width}x{screen_height}+0+0")
            
            # Try fullscreen methods in order of preference
            try:
                # Method 1: Try fullscreen attribute (most reliable on Linux)
                self.root.attributes('-fullscreen', True)
                print("SUCCESS: Fullscreen set using attributes('-fullscreen', True)")
            except Exception as e1:
                print(f"ERROR: Fullscreen attribute failed: {e1}")
                try:
                    # Method 2: Try overrideredirect (removes window decorations)
                    self.root.overrideredirect(True)
                    self.root.geometry(f"{screen_width}x{screen_height}+0+0")
                    print("SUCCESS: Fullscreen set using overrideredirect")
                except Exception as e2:
                    print(f"ERROR: Overrideredirect failed: {e2}")
                    try:
                        # Method 3: Try zoomed attribute
                        self.root.attributes('-zoomed', True)
                        print("SUCCESS: Fullscreen set using attributes('-zoomed', True)")
                    except Exception as e3:
                        print(f"ERROR: Zoomed attribute failed: {e3}")
                        print("SUCCESS: Using geometry-based fullscreen")
                        
        except Exception as e:
            print(f"WARNING: Error setting fullscreen: {e}")
            # Fallback: just set geometry
            try:
                screen_width = self.root.winfo_screenwidth()
                screen_height = self.root.winfo_screenheight()
                self.root.geometry(f"{screen_width}x{screen_height}+0+0")
                print("SUCCESS: Fallback: Set geometry to fullscreen")
            except:
                print("WARNING: Could not set fullscreen mode - using normal state")
        
        self.total_detections = 0
        # Guard camera synchronization lock to avoid concurrent release/open
        self._guard_cap_lock = threading.Lock()
        # When True, the guard camera loop will keep the cap open on stop so
        # it can be handed to the detection service instead of releasing it.
        self._guard_keep_cap_on_stop = False
        # Flag to indicate the guard camera cap has been handed off to detection
        self._cap_handed_to_detection = False

        # Valid guard IDs (default set) - will be refreshed from Firebase if available
        # NOTE: This prevents AttributeError in validate_guard_id if Firebase load hasn't run yet
        self.valid_guard_ids = ['0095081841', '0095339862']

        # Model configuration - use default that exists or will be overridden
        self.current_model_path = "bsba_female.pt"  # Default, will be overridden when student taps ID
        # Detection parameters - Optimized for visibility
        self.conf_threshold = 0.1  # Very low threshold to show all detections
        self.iou_threshold = 0.1   # Lower threshold for better overlap detection
        self.frame_skip = 1        # Process every frame for better detection
        self.prev_time = time.time()
        self.fps = 0
        
        # Complete uniform detection tracking
        self.detected_components = {}  # Track detected uniform components
        
        # Temporal tracking for improved accuracy
        self.previous_detections = []  # Store previous frame detections
        
        # Multi-monitor setup
        self.screen_info = self.get_screen_info()
        self.required_components = {}  # Required components per course
        self.uniform_complete = False
        self.last_complete_check = 0  # Timestamp of last complete uniform check
        
        # UI variables
        self.guard_id_var = tk.StringVar()
        self.login_status_var = tk.StringVar(value="Not Logged In")
        self.person_id_var = tk.StringVar()
        self.person_type_var = tk.StringVar(value="student")
        
        # Tab control
        self.notebook = None
        self.login_tab = None
        self.dashboard_tab = None
        self.visitor_tab = None
        self.rfid_history_tab = None
        self.student_forgot_tab = None
        self.main_screen_window = None
        
        # Initialize uniform components requirements
        self.initialize_uniform_requirements()
        
        # Initialize Firebase if available
        self.db = None
        self.firebase_initialized = False
        
        # Initialize Arduino connection for gate control
        self.arduino_connected = False
        self.arduino_serial = None
        self.cleanup_done = False
        
        # Visitor RFID management
        self.visitor_rfid_registry = {}  # RFID -> visitor_info
        self.active_visitors = {}  # RFID -> visitor_info with time_in
        self.available_rfids = ["0095272249", "0095520658"]  # Available RFID cards
        
        # Student forgot ID RFID management
        self.student_forgot_rfids = ["0095272825", "0095277892"]  # Empty RFID cards for student forgot ID
        self.student_rfid_assignments = {}  # RFID -> student_info for temporary assignments
        # Maps to convert displayed combobox strings back to raw RFID IDs
        self.rfid_display_map = {}  # for visitor combobox: display_string -> rfid_id
        self.student_rfid_display_map = {}  # for student forgot combobox
        # Availability tracking maps: raw_rfid -> bool
        self.rfid_availability_map = {}
        self.student_rfid_availability_map = {}
        # Flag to track when RFID assignment mode is active (waiting for empty RFID tap)
        self.rfid_assignment_active = False
        # Flag to track when visitor RFID assignment mode is active
        self.visitor_rfid_assignment_active = False
        # Flag to track when RFID return mode is active
        self.visitor_rfid_return_mode = False
        # References to return RFID dialog
        self.return_rfid_dialog = None
        self.return_rfid_entry_var = None
        self.return_rfid_entry = None
        
        # Main screen display timer (for auto-clearing after 15 seconds)
        self.main_screen_clear_timer = None
        
        # Violation session tracking (for retry logic)
        self.active_session_violations = {}  # rfid -> list of pending violation dicts
        self.active_detection_sessions = {}  # rfid -> session_start_time
        
        # Event Mode state (bypasses uniform detection)
        self.event_mode_active = False
        
        # Activity logs
        self.activity_logs = []
        self.max_logs = 100
        # Recent entries dedupe tracker (store tuple (id, action) for last entries)
        from collections import deque as _dq
        self.recent_entries_keys = _dq(maxlen=200)
        
        # Setup UI first
        self.setup_ui()
        
        # Initialize detection system
        self.init_detection_system()
        
        # Validate startup requirements
        self.validate_startup_requirements()
        
        # Initialize Firebase immediately during startup
        if FIREBASE_AVAILABLE:
            print("DEBUG: FIREBASE_AVAILABLE is True, initializing Firebase...")
            # Try synchronous initialization first
            success = self.init_firebase()
            if not success:
                print("INFO: Synchronous Firebase initialization failed, trying async...")
                self.init_firebase_async()
                # Wait a moment for async initialization
                time.sleep(2)
            
            # Clear approved violations on app startup (reset 24-hour timer)
            # This ensures that approved violations don't persist across app restarts
            if success and self.firebase_initialized and self.db:
                try:
                    self.clear_all_approved_violations_on_startup()
                except Exception as e:
                    print(f"⚠️ Warning: Could not clear approved violations on startup: {e}")
        else:
            print("DEBUG: FIREBASE_AVAILABLE is False, skipping Firebase initialization")
        
        # Set a flag to detect if Firebase consistently fails
        self.firebase_consistently_failed = False
        self.firebase_init_attempts = 0
        
        # Initialize BSBA uniform tracker
        self.bsba_uniform_tracker = BSBAUniformTracker()
        
        # 15-second uniform detection timer state
        self.uniform_detection_timer_start = None  # Timestamp when detection started
        self.uniform_detection_timer_duration = 15.0  # 15 second duration
        self.uniform_detection_complete = False  # Flag for early completion
        self.uniform_detection_timer_thread = None  # Background timer thread
        self.current_student_info_for_timer = None  # Store student_info for timer callback
        self.current_rfid_for_timer = None  # Store RFID separately to ensure it's always available
        self.requirements_hide_scheduled = False  # Flag to track if hide is already scheduled
        self.incomplete_student_info_for_retry = None  # Store student info when incomplete uniform timeout occurs
        
        # State tracking for DENIED button toggle functionality
        self.denied_button_clicked = False  # Flag to track if DENIED button was clicked
        self.original_detection_state = None  # Store original uniform detection state (complete/incomplete)
        self.original_uniform_status = None  # Store if student was originally complete or incomplete
        self.complete_uniform_auto_entry_timer = None  # Timer for auto-entry after complete uniform (8 seconds)
        self.complete_uniform_student_info = None  # Store student info for complete uniform auto-entry
        
        # Local cache to track recent entries (prevents race condition with Firebase)
        # Maps RFID to timestamp when entry was saved
        self.recent_entry_cache = {}  # {rfid: timestamp}
        
        # Initialize Arduino connection after UI is ready
        self.root.after(300, self.init_arduino_connection)
        
        # Start listening for Arduino button presses
        self.root.after(500, self.listen_for_arduino_buttons)
        
        # Ensure fullscreen on primary monitor after UI is ready
        self.root.after(200, self.ensure_fullscreen_on_primary_monitor)
        
        print("SUCCESS: Guard Control Center initialized successfully")
    
    def get_screen_info(self):
        """Get information about available screens"""
        try:
            # Get screen information
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            
            # Try to get multiple monitor information
            try:
                # For Windows
                if platform.system() == "Windows":
                    try:
                        import win32api  # type: ignore
                        monitors = win32api.EnumDisplayMonitors()
                    except ImportError:
                        print("WARNING: win32api not available - using basic screen info")
                        monitors = []
                    screen_info = {
                        'primary': {'width': screen_width, 'height': screen_height, 'x': 0, 'y': 0},
                        'secondary': None,
                        'count': len(monitors)
                    }
                    
                    if len(monitors) > 1:
                        # Get secondary monitor info
                        for monitor in monitors[1:]:
                            screen_info['secondary'] = {
                                'width': monitor[2][2] - monitor[2][0],
                                'height': monitor[2][3] - monitor[2][1],
                                'x': monitor[2][0],
                                'y': monitor[2][1]
                            }
                            break
                else:
                    # For Linux/Mac - use tkinter's screen info
                    screen_info = {
                        'primary': {'width': screen_width, 'height': screen_height, 'x': 0, 'y': 0},
                        'secondary': None,
                        'count': 1  # Default to single monitor
                    }
                    
                    # Try to detect secondary monitor using xrandr (Linux)
                    if platform.system() == "Linux":
                        try:
                            import subprocess
                            result = subprocess.run(['xrandr'], capture_output=True, text=True)
                            if 'connected' in result.stdout:
                                lines = result.stdout.split('\n')
                                connected_monitors = [line for line in lines if 'connected' in line and 'disconnected' not in line]
                                if len(connected_monitors) > 1:
                                    screen_info['count'] = len(connected_monitors)
                                    # Parse secondary monitor info
                                    for line in connected_monitors[1:]:
                                        if 'connected' in line:
                                            parts = line.split()
                                            for i, part in enumerate(parts):
                                                if '+' in part and 'x' in part:
                                                    try:
                                                        resolution = part.split('+')[0]
                                                        width, height = map(int, resolution.split('x'))
                                                        x_offset = int(parts[i].split('+')[1]) if '+' in parts[i] else 0
                                                        y_offset = int(parts[i].split('+')[2]) if '+' in parts[i] and len(parts[i].split('+')) > 2 else 0
                                                        screen_info['secondary'] = {
                                                            'width': width,
                                                            'height': height,
                                                            'x': x_offset,
                                                            'y': y_offset
                                                        }
                                                        break
                                                    except:
                                                        continue
                                            break
                        except:
                            pass
                            
            except ImportError:
                # Fallback for systems without win32api
                screen_info = {
                    'primary': {'width': screen_width, 'height': screen_height, 'x': 0, 'y': 0},
                    'secondary': None,
                    'count': 1
                }
            
            print(f"🖥️ Screen Info: {screen_info['count']} monitor(s) detected")
            if screen_info['secondary']:
                print(f"🖥️ Secondary monitor: {screen_info['secondary']['width']}x{screen_info['secondary']['height']} at ({screen_info['secondary']['x']}, {screen_info['secondary']['y']})")
            
            return screen_info
            
        except Exception as e:
            print(f"WARNING: Error getting screen info: {e}")
            return {
                'primary': {'width': 1920, 'height': 1080, 'x': 0, 'y': 0},
                'secondary': None,
                'count': 1
            }
    
    def initialize_uniform_requirements(self):
        """Initialize required uniform components for each course"""
        # Define required uniform components for each course
        self.required_components = {
            'ict': {
                'male': ['shirt', 'pants', 'shoes', 'belt', 'id_card'],
                'female': ['blouse', 'skirt', 'shoes', 'belt', 'id_card']
            },
            'tourism': {
                'male': ['polo_shirt', 'pants', 'shoes', 'belt', 'id_card'],
                'female': ['polo_shirt', 'pants', 'shoes', 'belt', 'id_card']
            },
            'teacher': {
                'male': ['dress_shirt', 'pants', 'shoes', 'belt', 'id_card'],
                'female': ['blouse', 'pants', 'shoes', 'belt', 'id_card']
            },
            'visitor': {
                'male': ['shirt', 'pants', 'shoes'],
                'female': ['shirt', 'pants', 'shoes']
            }
        }
        
        # Initialize detected components tracking
        self.detected_components = {}
        self.uniform_complete = False
        self.last_complete_check = 0
        
        print("SUCCESS: Uniform requirements initialized")
    
    def toggle_fullscreen(self, event=None):
        """Toggle fullscreen mode"""
        try:
            if self.root.attributes('-fullscreen'):
                self.root.attributes('-fullscreen', False)
                print("🖥️ Exited fullscreen mode")
            else:
                self.root.attributes('-fullscreen', True)
                print("🖥️ Entered fullscreen mode")
        except Exception as e:
            print(f"WARNING: Error toggling fullscreen: {e}")
    
    def exit_fullscreen(self, event=None):
        """Exit fullscreen mode"""
        try:
            self.root.attributes('-fullscreen', False)
            self.root.overrideredirect(False)
            print("🖥️ Exited fullscreen mode")
        except Exception as e:
            print(f"WARNING: Error exiting fullscreen: {e}")
    
    def toggle_main_screen_fullscreen(self, event=None):
        """Toggle Main Screen fullscreen mode"""
        try:
            if hasattr(self, 'main_screen_window') and self.main_screen_window:
                if self.main_screen_window.attributes('-fullscreen'):
                    self.main_screen_window.attributes('-fullscreen', False)
                    self.main_screen_window.overrideredirect(False)
                    print("🖥️ Main Screen exited fullscreen mode")
                else:
                    # Get secondary monitor dimensions
                    if self.screen_info['secondary']:
                        secondary = self.screen_info['secondary']
                        self.main_screen_window.geometry(f"{secondary['width']}x{secondary['height']}+{secondary['x']}+{secondary['y']}")
                        self.main_screen_window.attributes('-fullscreen', True)
                        print(f"🖥️ Main Screen entered fullscreen mode: {secondary['width']}x{secondary['height']}")
                    else:
                        print("WARNING: No secondary monitor available for Main Screen fullscreen")
        except Exception as e:
            print(f"WARNING: Error toggling Main Screen fullscreen: {e}")
    
    def exit_main_screen_fullscreen(self, event=None):
        """Exit Main Screen fullscreen mode"""
        try:
            if hasattr(self, 'main_screen_window') and self.main_screen_window:
                self.main_screen_window.attributes('-fullscreen', False)
                self.main_screen_window.overrideredirect(False)
                print("🖥️ Main Screen exited fullscreen mode")
        except Exception as e:
            print(f"WARNING: Error exiting Main Screen fullscreen: {e}")
    
    def ensure_fullscreen_on_primary_monitor(self):
        """Ensure window is fullscreen on primary monitor"""
        try:
            # Get primary monitor dimensions
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            
            # Set to fullscreen on primary monitor (position 0,0)
            self.root.geometry(f"{screen_width}x{screen_height}+0+0")
            
            # Try to set fullscreen
            try:
                self.root.attributes('-fullscreen', True)
                print(f"SUCCESS: Fullscreen set on primary monitor: {screen_width}x{screen_height}")
            except:
                try:
                    self.root.attributes('-zoomed', True)
                    print(f"SUCCESS: Maximized on primary monitor: {screen_width}x{screen_height}")
                except:
                    print(f"SUCCESS: Positioned on primary monitor: {screen_width}x{screen_height}")
                    
        except Exception as e:
            print(f"WARNING: Error ensuring fullscreen on primary monitor: {e}")
    
    def validate_startup_requirements(self):
        """Validate all startup requirements"""
        try:
            print("🔍 Validating startup requirements...")
            
            # Check for required model files
            model_files = ["bsba male.pt", "bsba_female.pt"]
            missing_models = []
            
            for model_file in model_files:
                if not os.path.exists(model_file):
                    missing_models.append(model_file)
            
            if missing_models:
                print(f"WARNING: Missing model files: {missing_models}")
            else:
                print("SUCCESS: All model files found")
            
            # Check Firebase configuration
            if FIREBASE_AVAILABLE:
                if os.path.exists("firebase_service_account.json") or os.path.exists("serviceAccountKey.json"):
                    print("SUCCESS: Firebase configuration file found")
                else:
                    print("WARNING: Firebase configuration file not found")
            else:
                print("WARNING: Firebase not available - running in offline mode")
            
            # Check camera availability
            if CV2_AVAILABLE:
                try:
                    # Try different camera indices - prioritize built-in camera first
                    camera_found = False
                    for camera_index in [0, 1, 2]:  # Try built-in camera (0) first, then external (1), then others
                        try:
                            # Prefer DirectShow backend on Windows for more reliable access
                            if platform.system().lower().startswith('win') and hasattr(cv2, 'CAP_DSHOW'):
                                test_cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
                            else:
                                test_cap = cv2.VideoCapture(camera_index)
                            if test_cap.isOpened():
                                # Use normal camera settings for testing
                                ret, frame = test_cap.read()
                                if ret and frame is not None:
                                    camera_type = "Built-in" if camera_index == 0 else "External" if camera_index == 1 else f"Camera {camera_index}"
                                    print(f"SUCCESS: {camera_type} camera available on index {camera_index}")
                                    camera_found = True
                                    test_cap.release()
                                    break
                                test_cap.release()
                        except Exception as e:
                            print(f"WARNING: Camera {camera_index} test failed: {e}")
                            continue
                    
                    if not camera_found:
                        print("WARNING: Camera not accessible - detection features will be limited")
                        print("💡 Try connecting a USB camera or check camera permissions")
                except Exception as e:
                    print(f"WARNING: Camera test failed: {e}")
            else:
                print("WARNING: OpenCV not available - camera features disabled")
            
            # Check YOLO availability
            if YOLO_AVAILABLE:
                print("SUCCESS: YOLO available")
            else:
                print("WARNING: YOLO not available - AI detection disabled")
            
            print("SUCCESS: Startup validation completed")
            
        except Exception as e:
            print(f"ERROR: Error during startup validation: {e}")
    
    def debug_firebase_student(self, student_id):
        """Debug function to check student data in Firebase"""
        try:
            if not self.firebase_initialized or not self.db:
                print("ERROR: Firebase not initialized")
                return None
            
            print(f"🔍 Checking Firebase for student ID: {student_id}")
            
            # Query Firebase students collection
            doc_ref = self.db.collection('students').document(student_id)
            doc = doc_ref.get()
            
            if doc.exists:
                data = doc.to_dict()
                print(f"SUCCESS: Document found in Firebase:")
                print(f"   Document ID: {student_id}")
                print(f"   Data: {data}")
                print(f"   Name: {data.get('name', 'Not found')}")
                print(f"   Course: {data.get('course', 'Not found')}")
                print(f"   Gender: {data.get('gender', 'Not found')}")
                return data
            else:
                print(f"ERROR: Document '{student_id}' not found in Firebase students collection")
                return None
                
        except Exception as e:
            print(f"ERROR: Error checking Firebase student: {e}")
            return None
    
    def test_firebase_connection(self):
        """Test Firebase connection and students collection access"""
        try:
            if not self.firebase_initialized or not self.db:
                print("ERROR: Firebase not initialized")
                return False
            
            # Test connection by trying to read from students collection
            test_doc = self.db.collection('students').limit(1).get()
            print("SUCCESS: Firebase connection test successful")
            return True
            
        except Exception as e:
            print(f"ERROR: Firebase connection test failed: {e}")
            return False
    
    def test_firebase_connection_with_timeout(self):
        """Test Firebase connection with timeout handling"""
        try:
            import threading
            
            def test_connection():
                try:
                    if not self.firebase_initialized or not self.db:
                        print("ERROR: Firebase not initialized")
                        return False
                    
                    # Test connection by trying to read from students collection
                    test_doc = self.db.collection('students').limit(1).get()
                    print("SUCCESS: Firebase connection test successful")
                    return True
                    
                except Exception as e:
                    print(f"ERROR: Firebase connection test failed: {e}")
                    return False
            
            # Run connection test in a separate thread with timeout
            result = [False]
            def run_test():
                result[0] = test_connection()
            
            thread = threading.Thread(target=run_test)
            thread.daemon = True
            thread.start()
            thread.join(timeout=10)  # 10 second timeout
            
            if thread.is_alive():
                print("WARNING: Firebase connection test timed out - continuing in offline mode")
            else:
                print("SUCCESS: Firebase connection test completed")
                
        except Exception as e:
            print(f"WARNING: Firebase connection test error: {e}")
    
    def init_firebase_async(self):
        """Initialize Firebase connection asynchronously with robust path handling"""
        print("DEBUG: Starting Firebase async initialization...")
        try:
            # Get the current working directory and script directory
            import os
            current_dir = os.getcwd()
            script_dir = os.path.dirname(os.path.abspath(__file__))
            
            print(f"DEBUG: Current working directory: {current_dir}")
            print(f"DEBUG: Script directory: {script_dir}")
            
            # Check for Firebase credentials in multiple locations
            cred_files = [
                'serviceAccountKey.json',
                'firebase_service_account.json',
                os.path.join(script_dir, 'serviceAccountKey.json'),
                os.path.join(script_dir, 'firebase_service_account.json'),
                os.path.join(current_dir, 'serviceAccountKey.json'),
                os.path.join(current_dir, 'firebase_service_account.json')
            ]
            
            cred_file = None
            for file_path in cred_files:
                if os.path.exists(file_path):
                    cred_file = file_path
                    print(f"SUCCESS: Found Firebase credentials at: {file_path}")
                    break
            
            if not cred_file:
                print("WARNING: Firebase service account not found - running in offline mode")
                print(f"DEBUG: Searched locations: {cred_files}")
                return
            
            # Check if Firebase is already initialized
            if hasattr(firebase_admin, '_apps') and firebase_admin._apps:
                print("INFO: Firebase already initialized")
                self.firebase_initialized = True
                self.db = firestore.client()
                return
            
            print(f"INFO: Initializing Firebase with credentials: {cred_file}")
            cred = credentials.Certificate(cred_file)
            firebase_admin.initialize_app(cred)
            self.db = firestore.client()
            self.firebase_initialized = True
            print("SUCCESS: Firebase initialized successfully")
            print(f"SUCCESS: Connected to project: ainiform-system-c42de")
            
            # Save guards to Firebase
            self.save_guards_to_firebase()
            
            # Save permanent students to Firebase
            self.save_permanent_students_to_firebase()
            
            # Test Firebase connection with timeout
            if hasattr(self, 'root') and self.root.winfo_exists():
                self.root.after(1000, self.test_firebase_connection_with_timeout)
            else:
                print("INFO: Firebase initialization completed")
                
        except Exception as e:
            print(f"ERROR: Firebase initialization failed: {e}")
            print(f"DEBUG: Error type: {type(e).__name__}")
            print(f"DEBUG: Error details: {str(e)}")
            print("INFO: Switching to offline mode")
            self.firebase_initialized = False
    
    def init_firebase(self):
        """Initialize Firebase connection (synchronous) with robust path handling"""
        try:
            print("INFO: Starting Firebase initialization...")
            
            # Check if Firebase is already initialized
            if hasattr(firebase_admin, '_apps') and firebase_admin._apps:
                print("INFO: Firebase already initialized")
                self.firebase_initialized = True
                self.db = firestore.client()
                return True
            
            # Look for credentials file
            cred_file = 'serviceAccountKey.json'
            if not os.path.exists(cred_file):
                print(f"ERROR: Firebase credentials not found: {cred_file}")
                print("INFO: Make sure serviceAccountKey.json is in the project directory")
                self.firebase_initialized = False
                return False
            
            print(f"INFO: Found Firebase credentials: {cred_file}")
            print(f"INFO: Initializing Firebase connection...")
            
            # Initialize Firebase
            cred = credentials.Certificate(cred_file)
            firebase_admin.initialize_app(cred)
            self.db = firestore.client()
            self.firebase_initialized = True
            
            print("SUCCESS: Firebase initialized successfully")
            print(f"SUCCESS: Connected to project: {self.db.project}")
            
            # Test the connection
            try:
                # Try to read from Firestore to test connection
                test_ref = self.db.collection('test').limit(1)
                list(test_ref.stream())
                print("SUCCESS: Firebase connection test passed")
            except Exception as e:
                print(f"WARNING: Firebase connection test failed: {e}")
            
            # Save initial data to Firebase
            try:
                self.save_guards_to_firebase()
                self.save_permanent_students_to_firebase()
                print("SUCCESS: Initial data saved to Firebase")
            except Exception as e:
                print(f"WARNING: Failed to save initial data to Firebase: {e}")
            
            return True
                
        except Exception as e:
            print(f"ERROR: Firebase initialization failed: {e}")
            print(f"DEBUG: Error type: {type(e).__name__}")
            print(f"DEBUG: Error details: {str(e)}")
            print("INFO: Switching to offline mode")
            self.firebase_initialized = False
            return False
    
    def center_window(self):
        """Center the window on screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
    
    def setup_ui(self):
        """Setup the main UI"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create login tab
        self.create_login_tab()
        
        # Initially hide dashboard tab
        self.dashboard_tab = None
    
    def create_login_tab(self):
        """Create the login tab with improved visibility"""
        self.login_tab = tk.Frame(self.notebook, bg='#ffffff')
        self.notebook.add(self.login_tab, text="🔐 Guard Login")
        
        # Set tab as active
        self.notebook.select(self.login_tab)
        
        # Header
        self.create_header(self.login_tab)
        
        # Login content
        self.create_login_content(self.login_tab)
    
    def create_header(self, parent):
        """Create header section with improved visibility"""
        header_frame = tk.Frame(parent, bg='#1e3a8a', height=100)
        header_frame.pack(fill=tk.X, pady=(0, 25))
        header_frame.pack_propagate(False)
        
        # Left side - Title
        title_frame = tk.Frame(header_frame, bg='#1e3a8a')
        title_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        title_label = tk.Label(
            title_frame,
            text="🛡️ AI-niform Guard Control Center",
            font=('Arial', 28, 'bold'),
            fg='white',
            bg='#1e3a8a'
        )
        title_label.pack(expand=True)
        
        subtitle_label = tk.Label(
            title_frame,
            text="Advanced Security Management System",
            font=('Arial', 14),
            fg='#e5e7eb',
            bg='#1e3a8a'
        )
        subtitle_label.pack()
        
        # Right side - Logout button (only on dashboard)
        if self.dashboard_tab is not None:
            logout_frame = tk.Frame(header_frame, bg='#1e3a8a')
            logout_frame.pack(side=tk.RIGHT, padx=25, pady=15)
            
            self.logout_btn = tk.Button(
                logout_frame,
                text="🚪 Logout",
                font=('Arial', 12, 'bold'),
                fg='white',
                bg='#dc2626',
                activebackground='#b91c1c',
                activeforeground='white',
                relief='raised',
                bd=3,
                padx=15,
                pady=8,
                cursor='hand2',
                command=self.handle_logout
            )
            self.logout_btn.pack(fill=tk.X)
    
    def create_login_content(self, parent):
        """Create login content with perfectly positioned buttons"""
        # Main login container with centered layout
        login_frame = tk.Frame(parent, bg='#ffffff', relief='solid', bd=2)
        login_frame.pack(fill=tk.BOTH, expand=True, padx=40, pady=40)
        
        # Title section - centered
        title_frame = tk.Frame(login_frame, bg='#ffffff')
        title_frame.pack(fill=tk.X, pady=(30, 25))
        
        login_title = tk.Label(
            title_frame,
            text="🔐 Guard Authentication",
            font=('Arial', 24, 'bold'),
            fg='#1e3a8a',
            bg='#ffffff'
        )
        login_title.pack()
        
        # Subtitle
        subtitle = tk.Label(
            title_frame,
            text="Enter your Guard ID to access the system",
            font=('Arial', 13),
            fg='#6b7280',
            bg='#ffffff'
        )
        subtitle.pack(pady=(8, 0))
        
        # Main form container - centered
        form_container = tk.Frame(login_frame, bg='#ffffff')
        form_container.pack(expand=True, pady=15)
        
        # Guard ID input section - centered
        input_section = tk.Frame(form_container, bg='#ffffff')
        input_section.pack(expand=True)
        
        # Guard ID label
        id_label = tk.Label(
            input_section,
            text="Guard ID:",
            font=('Arial', 16, 'bold'),
            fg='#1e3a8a',
            bg='#ffffff'
        )
        id_label.pack(pady=(0, 15))
        
        # Input field - properly sized
        self.id_entry = tk.Entry(
            input_section,
            textvariable=self.guard_id_var,
            font=('Arial', 18, 'bold'),
            width=12,
            justify=tk.CENTER,
            relief='solid',
            bd=3,
            bg='#f0f9ff',
            fg='#1e3a8a',
            insertbackground='#1e3a8a'
        )
        self.id_entry.pack(pady=(0, 30))
        self.id_entry.bind('<Return>', lambda e: self.login_manual())
        self.id_entry.focus()
        
        # Button container for proper alignment
        button_container = tk.Frame(input_section, bg='#ffffff')
        button_container.pack(expand=True, pady=10)
        
        # LOGIN button - properly sized and positioned
        self.manual_login_btn = tk.Button(
            button_container,
            text="🚀 LOGIN",
            command=self.login_manual,
            font=('Arial', 18, 'bold'),
            bg='#3b82f6',
            fg='white',
            relief='raised',
            bd=4,
            padx=50,
            pady=15,
            cursor='hand2',
            activebackground='#2563eb',
            activeforeground='white'
        )
        self.manual_login_btn.pack(pady=(0, 20))
        
        # Status display section
        status_frame = tk.Frame(login_frame, bg='#ffffff')
        status_frame.pack(fill=tk.X, pady=(20, 15))
        
        self.status_label = tk.Label(
            status_frame,
            textvariable=self.login_status_var,
            font=('Arial', 14, 'bold'),
            fg='#dc2626',
            bg='#ffffff'
        )
        self.status_label.pack()
        
        # Instructions
        instructions = tk.Label(
            status_frame,
            text="Enter your Guard ID to login",
            font=('Arial', 11),
            fg='#6b7280',
            bg='#ffffff'
        )
        instructions.pack(pady=(8, 0))
        
        # QUIT button section - highly visible
        quit_section = tk.Frame(login_frame, bg='#ffffff')
        quit_section.pack(fill=tk.X, pady=(30, 20))
        
        # Add a separator line
        separator = tk.Frame(quit_section, height=2, bg='#e5e7eb')
        separator.pack(fill=tk.X, pady=(0, 20))
        
        # QUIT button - much larger and more visible
        quit_btn_main = tk.Button(
            quit_section,
            text="QUIT APPLICATION",
            command=self.quit_application,
            font=('Arial', 20, 'bold'),
            bg='#dc2626',
            fg='white',
            relief='raised',
            bd=5,
            padx=60,
            pady=18,
            cursor='hand2',
            activebackground='#b91c1c',
            activeforeground='white'
        )
        quit_btn_main.pack(pady=10)
        
        # Add instruction text
        quit_instruction = tk.Label(
            quit_section,
            text="Click the button above to exit the application",
            font=('Arial', 12),
            fg='#6b7280',
            bg='#ffffff'
        )
        quit_instruction.pack(pady=(5, 0))
    
    def create_footer(self, parent):
        """Create footer section"""
        footer_frame = tk.Frame(parent, bg='#e5e7eb', relief='solid', bd=2)
        footer_frame.pack(fill=tk.X, pady=(25, 0))
        
        # Footer content container
        footer_content = tk.Frame(footer_frame, bg='#e5e7eb')
        footer_content.pack(expand=True, pady=20)
    
    def login_manual(self):
        """Handle manual login"""
        guard_id = self.guard_id_var.get().strip().upper()
        
        if not guard_id:
            messagebox.showwarning("Warning", "Please enter a Guard ID")
            return
        
        if self.validate_guard_id(guard_id):
            self.authenticate_guard(guard_id)
        else:
            messagebox.showerror("Error", f"Invalid Guard ID: {guard_id}")
    
    def validate_guard_id(self, guard_id):
        """Validate guard ID"""
        return guard_id in self.valid_guard_ids
    
    def authenticate_guard(self, guard_id):
        """Authenticate the guard"""
        self.current_guard_id = guard_id
        self.login_status_var.set("Logged In")
        
        # Generate session ID
        self.session_id = f"{guard_id}_{int(time.time())}"
        
        # Save guard login to Firebase
        self.save_guard_login_to_firebase(guard_id)
        
        # Update guard login info in guards collection
        self.update_guard_login_info(guard_id)
        
        # Disable login button
        self.manual_login_btn.config(state=tk.DISABLED)
        
        # Disable and unbind guard login Entry field to prevent RFID input from going to it
        # This prevents RFID taps from being incorrectly routed to guard login validation
        if hasattr(self, 'id_entry'):
            self.id_entry.config(state=tk.DISABLED)
            self.id_entry.unbind('<Return>')
            self.id_entry.config(takefocus=False)
            # Clear any existing value and remove focus
            self.guard_id_var.set("")
            # Remove focus from the Entry field to prevent RFID input from going to it
            try:
                self.root.focus_set()  # Move focus to root window instead of Entry field
            except Exception:
                pass
        
        # Camera will open when student taps their ID
        self.update_camera_label_for_guard()
        
        # Enable logout button
        if hasattr(self, 'logout_btn'):
            self.logout_btn.config(state=tk.NORMAL)
        
        # Update guard name display
        self.update_guard_name_display(guard_id)
        
        self.show_green_success_message("Login Successful", f"Welcome, Guard {guard_id}!")
        
        # Start security system
        self.start_security_system()
    
    def get_guard_rfid(self, guard_id):
        """Get guard's RFID from Firebase guards collection"""
        try:
            if not self.firebase_initialized or not self.db:
                print(f"⚠️ Firebase not initialized - cannot get guard RFID for {guard_id}")
                return None
            
            # Query Firebase guards collection using guard_id as document ID
            guard_ref = self.db.collection('guards').document(guard_id)
            guard_doc = guard_ref.get()
            
            if guard_doc.exists:
                guard_data = guard_doc.to_dict()
                # Try different field name variations
                rfid = guard_data.get('rfid') or guard_data.get('RFID') or guard_data.get('guard_rfid')
                if rfid:
                    return str(rfid).strip()
                else:
                    print(f"⚠️ Guard {guard_id} document exists but has no RFID field")
                    return None
            else:
                print(f"⚠️ Guard {guard_id} not found in Firebase guards collection")
                return None
                
        except Exception as e:
            print(f"❌ Error getting guard RFID from Firebase: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_guard_name(self, guard_id):
        """Get guard's name from Firebase guards collection"""
        try:
            if not self.firebase_initialized or not self.db:
                print(f"⚠️ Firebase not initialized - cannot get guard name for {guard_id}")
                return None
            
            # Query Firebase guards collection using guard_id as document ID
            guard_ref = self.db.collection('guards').document(guard_id)
            guard_doc = guard_ref.get()
            
            if guard_doc.exists:
                guard_data = guard_doc.to_dict()
                # Try different field name variations for name
                name = guard_data.get('name') or guard_data.get('Name') or guard_data.get('guard_name') or guard_data.get('full_name')
                if name:
                    return str(name).strip()
                else:
                    print(f"⚠️ Guard {guard_id} document exists but has no name field")
                    return None
            else:
                print(f"⚠️ Guard {guard_id} not found in Firebase guards collection")
                return None
                
        except Exception as e:
            print(f"❌ Error getting guard name from Firebase: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def update_guard_name_display(self, guard_id=None):
        """Update the guard name display in the Person ID Entry section"""
        try:
            if hasattr(self, 'guard_name_label'):
                if guard_id:
                    guard_name = self.get_guard_name(guard_id)
                    if guard_name:
                        self.guard_name_label.config(text=guard_name, fg='#1e3a8a')
                    else:
                        # If name not found, show guard ID
                        self.guard_name_label.config(text=f"Guard {guard_id}", fg='#1e3a8a')
                else:
                    self.guard_name_label.config(text="No guard logged in", fg='#6b7280')
        except Exception as e:
            print(f"❌ Error updating guard name display: {e}")
    
    def handle_logout(self):
        """Handle guard logout with RFID confirmation"""
        try:
            # Check if guard is logged in
            if not self.current_guard_id:
                messagebox.showwarning("Not Logged In", "No guard is currently logged in.")
                return
            
            # Get guard's RFID from Firebase
            guard_rfid = self.get_guard_rfid(self.current_guard_id)
            if not guard_rfid:
                messagebox.showerror("Error", f"Could not retrieve RFID for Guard {self.current_guard_id}.\n\nPlease contact administrator.")
                return
            
            # Show popup asking for RFID confirmation
            import tkinter.simpledialog as simpledialog
            rfid_input = simpledialog.askstring(
                "Logout Confirmation",
                f"Please tap your Guard RFID to confirm logout.\n\nGuard ID: {self.current_guard_id}",
                parent=self.root
            )
            
            # Check if user cancelled
            if rfid_input is None:
                print("Logout cancelled by user")
                return
            
            # Validate RFID (case-insensitive, strip whitespace)
            rfid_input = str(rfid_input).strip()
            guard_rfid_clean = str(guard_rfid).strip()
            
            if rfid_input.upper() != guard_rfid_clean.upper():
                messagebox.showerror(
                    "Invalid RFID",
                    f"RFID does not match Guard {self.current_guard_id}.\n\nLogout cancelled."
                )
                return
            
            # RFID matches - proceed with logout
            guard_id_to_logout = self.current_guard_id
            
            # Stop any active detection and close camera
            if hasattr(self, 'detection_active') and self.detection_active:
                self.stop_detection()
            
            # Reset camera to standby
            if hasattr(self, 'reset_camera_to_standby'):
                self.reset_camera_to_standby()
            
            # Close main screen
            if hasattr(self, 'close_main_screen'):
                self.close_main_screen()
            
            # Save guard logout to Firebase before clearing session data
            if hasattr(self, 'save_guard_logout_to_firebase') and self.current_guard_id:
                self.save_guard_logout_to_firebase(self.current_guard_id)
            
            # Clear session
            self.current_guard_id = None
            self.session_id = None
            
            # Reset login status
            self.login_status_var.set("Not Logged In")
            
            # Re-enable login button
            if hasattr(self, 'manual_login_btn'):
                self.manual_login_btn.config(state=tk.NORMAL)
            
            # Re-enable guard ID entry field
            if hasattr(self, 'id_entry'):
                self.id_entry.config(state=tk.NORMAL)
                self.id_entry.config(takefocus=True)
                self.guard_id_var.set("")
                # Rebind Return key for login
                self.id_entry.bind('<Return>', lambda e: self.login_manual())
            
            # Disable logout button
            if hasattr(self, 'logout_btn'):
                self.logout_btn.config(state=tk.DISABLED)
            
            # Switch back to login tab
            if hasattr(self, 'notebook') and hasattr(self, 'login_tab') and hasattr(self, 'dashboard_tab'):
                if self.dashboard_tab is not None:
                    self.notebook.select(self.login_tab)
                    self.notebook.hide(self.dashboard_tab)
            
            # Update guard name display to show no guard logged in
            self.update_guard_name_display(None)
            
            # Show success message
            messagebox.showinfo("Logout Successful", f"Guard {guard_id_to_logout} has been logged out successfully.")
            
            # Add to activity log
            if hasattr(self, 'add_activity_log'):
                self.add_activity_log(f"Guard {guard_id_to_logout} logged out")
            
            print(f"✅ Guard {guard_id_to_logout} logged out successfully")
            
        except Exception as e:
            print(f"❌ Error handling logout: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"An error occurred during logout: {e}")
    
# Camera method removed - will be replaced with user's detection file

    def update_camera_feed(self, frame):
        """Update camera feed with detection frame"""
        try:
            if hasattr(self, 'camera_label') and self.camera_label and self.root.winfo_exists():
                # Get the camera label dimensions
                label_width = self.camera_label.winfo_width()
                label_height = self.camera_label.winfo_height()
                
                # If label dimensions are not available yet, use default
                if label_width <= 1 or label_height <= 1:
                    label_width = 400
                    label_height = 300
                
                # Calculate aspect ratio preserving resize
                frame_height, frame_width = frame.shape[:2]
                aspect_ratio = frame_width / frame_height
                
                # Calculate new dimensions that fit within the label
                if label_width / label_height > aspect_ratio:
                    # Label is wider than needed, fit to height
                    new_height = label_height - 20  # Leave some padding
                    new_width = int(new_height * aspect_ratio)
                else:
                    # Label is taller than needed, fit to width
                    new_width = label_width - 20  # Leave some padding
                    new_height = int(new_width / aspect_ratio)
                
                # Ensure minimum size
                new_width = max(new_width, 200)
                new_height = max(new_height, 150)
                
                # Resize frame to fit camera label properly
                frame_resized = cv2.resize(frame, (new_width, new_height))
                
                # Convert BGR to RGB for Tkinter
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                
                # Convert to PhotoImage
                from PIL import Image, ImageTk
                pil_image = Image.fromarray(frame_rgb)
                photo = ImageTk.PhotoImage(pil_image)
                
                # Update camera label in main thread
                self.root.after(0, self._update_camera_label_with_frame, photo)
                
        except Exception as e:
            print(f"ERROR: Failed to update camera feed: {e}")
            import traceback
            traceback.print_exc()

    def _add_bscpe_guard_overlay(self, frame):
        """Add BSCPE uniform checklist overlay to guard camera feed - DISABLED"""
        # BSCPE overlay disabled - requirements shown in UI panel instead
        return frame
        try:
            if not (hasattr(self, 'detection_system') and self.detection_system and 
                    hasattr(self.detection_system, 'detection_service') and 
                    self.detection_system.detection_service and
                    hasattr(self.detection_system.detection_service, 'bscpe_tracker') and
                    self.detection_system.detection_service.bscpe_tracker):
                return frame
            
            bscpe_tracker = self.detection_system.detection_service.bscpe_tracker
            person_name = self.detection_system.detection_service.current_person_name or "Unknown"
            
            # Get checklist status
            checklist = bscpe_tracker.get_checklist_status()
            verification_status, verification_message = bscpe_tracker.get_verification_status()
            
            # Set overlay colors based on status
            if verification_status == "COMPLETE":
                color = (0, 255, 0)  # Green
                bg_color = (0, 100, 0)  # Dark green background
            elif verification_status == "TIMEOUT":
                color = (0, 0, 255)  # Red
                bg_color = (100, 0, 0)  # Dark red background
            else:  # PENDING
                color = (0, 165, 255)  # Orange
                bg_color = (100, 50, 0)  # Dark orange background
            
            # Add background rectangle
            cv2.rectangle(frame, (10, 10), (450, 200), bg_color, -1)
            cv2.rectangle(frame, (10, 10), (450, 200), color, 2)
            
            # Add title
            cv2.putText(frame, "BSCPE UNIFORM CHECKLIST", (20, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Add student info
            cv2.putText(frame, f"Student: {person_name}", (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add status
            cv2.putText(frame, f"Status: {verification_status}", (20, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add checklist items
            y_pos = 105
            for part, status_info in checklist.items():
                if status_info['status'] == 'complete':
                    # Green checkmark for completed items
                    item_text = f"✅ {part}"
                    text_color = (0, 255, 0)  # Green
                elif status_info['status'] == 'partial':
                    # Yellow for partially detected items
                    item_text = f"🟡 {part}: {status_info['count']}/{status_info['required']}"
                    text_color = (0, 255, 255)  # Yellow
                else:
                    # White for missing items
                    item_text = f"⚪ {part}: 0/{status_info['required']}"
                    text_color = (255, 255, 255)  # White
                
                cv2.putText(frame, item_text, (20, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
                y_pos += 20
            
            # Add timer if pending
            if verification_status == "PENDING":
                elapsed_time = time.time() - bscpe_tracker.start_time
                remaining_time = max(0, bscpe_tracker.timeout_duration - elapsed_time)
                timer_text = f"Time remaining: {remaining_time:.1f}s"
                cv2.putText(frame, timer_text, (20, y_pos + 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            return frame
            
        except Exception as e:
            print(f"ERROR: Failed to add BSCPE guard overlay: {e}")
            return frame

    def _update_camera_label_with_frame(self, photo):
        """Update camera label with frame in main thread"""
        try:
            if hasattr(self, 'camera_label') and self.camera_label:
                self.camera_label.config(
                    image=photo,
                    text="",  # Remove text when showing image
                    # Remove fixed width/height to allow proper scaling
                    relief="sunken",
                    bd=2
                )
                self.camera_label.image = photo  # Keep a reference
                print("📹 Camera frame updated successfully")
        except Exception as e:
            print(f"ERROR: Failed to update camera label with frame: {e}")
            import traceback
            traceback.print_exc()

    def start_person_detection_integrated(self, person_id, person_name, person_type):
        """Start detection for a person using the integrated detection system.
        To avoid freezing the UI when an RFID tap occurs, run the detection
        handoff in a background thread. The background worker performs the
        same steps as before (stop guard feed, initialize detection system,
        start detection) but won't block the main thread.
        """
        def _worker():
            try:
                print(f"🔍 Starting integrated detection for {person_name} ({person_type})")
                self.add_activity_log(f"🔍 Starting integrated detection for {person_name} ({person_type})")

                # Get the correct model path for this person BEFORE initializing DetectionSystem
                model_path = self._get_model_path_for_person(person_id, person_type)
                print(f"🎯 Using model path for DetectionSystem: {model_path}")
                
                # Set uniform requirements and show requirements section for students
                if person_type.lower() == 'student':
                    try:
                        student_info = self.get_student_info(person_id)
                        if student_info:
                            course = student_info.get('course', '')
                            gender = student_info.get('gender', '')
                            self.set_uniform_requirements_by_course(course, gender)
                            self.show_requirements_section(student_info)
                            
                            # Reset denied button state flags when new student ID is tapped (but keep button enabled)
                            # Only reset flags, don't disable button - it will be enabled when detection starts
                            self.denied_button_clicked = False
                            self.original_detection_state = None
                            self.original_uniform_status = None
                            
                            # Reset DENIED button text and command to original (but keep it enabled)
                            if hasattr(self, 'deny_button') and self.deny_button:
                                self.deny_button.config(
                                    text="DENIED",
                                    command=self.handle_interface_deny
                                )
                            
                            # Make checkboxes read-only initially
                            self._make_requirement_checkboxes_editable(False)
                            
                            print(f"✅ Requirements section shown for: {course} ({gender})")
                    except Exception as e:
                        print(f"⚠️ Error setting requirements: {e}")
                
                # Prepare detection system first so we can transfer the open camera
                try:
                    if not hasattr(self, 'detection_system') or self.detection_system is None:
                        conf_threshold = getattr(self, 'conf_threshold', 0.65)
                        self.detection_system = DetectionSystem(model_path=model_path, conf_threshold=conf_threshold)
                        try:
                            self.detection_system.main_ui = self
                        except Exception:
                            pass
                        print("✅ DetectionSystem initialized successfully")
                        self.add_activity_log("✅ DetectionSystem initialized successfully")
                except Exception as e:
                    print(f"❌ Failed to initialize DetectionSystem before camera transfer: {e}")
                    import traceback; traceback.print_exc()

                # Attempt to transfer the currently-open guard camera to the detection service
                try:
                    # If guard camera exists, hand it over without releasing the device hardware
                    svc_present = getattr(self.detection_system, 'detection_service', None)
                    print(f"DEBUG: detection_service present before handoff: {bool(svc_present)}")
                    if getattr(self, 'guard_camera_cap', None) is not None and getattr(self.detection_system, 'detection_service', None):
                        try:
                            # Signal guard loop to stop but keep the cap for transfer
                            self._guard_keep_cap_on_stop = True
                            self.guard_camera_active = False

                            # Wait briefly for guard loop to expose the cap under lock
                            transfer_wait = time.time()
                            while getattr(self, 'guard_camera_cap', None) is None and time.time() - transfer_wait < 1.0:
                                time.sleep(0.02)

                            # Acquire cap under lock and hand to detection service
                            with self._guard_cap_lock:
                                cap_to_transfer = getattr(self, 'guard_camera_cap', None)
                                if cap_to_transfer is not None:
                                    try:
                                        # Use detection service API to accept existing camera
                                        self.detection_system.detection_service.set_existing_camera(cap_to_transfer)
                                        # Mark that cap is now handed to detection and clear main reference
                                        try:
                                            self._cap_handed_to_detection = True
                                        except Exception:
                                            pass
                                        self.guard_camera_cap = None
                                        print("🔁 Guard camera transferred to detection service")
                                    except Exception as e:
                                        print(f"⚠️ Failed to transfer guard cap to detection service: {e}")
                        except Exception:
                            pass
                except Exception:
                    pass

                # DetectionSystem already initialized above (or earlier). If still missing, initialize.
                if not hasattr(self, 'detection_system') or self.detection_system is None:
                    try:
                        # Get model path (should already be set above, but get it again to be safe)
                        model_path = self._get_model_path_for_person(person_id, person_type)
                        conf_threshold = getattr(self, 'conf_threshold', 0.65)
                        self.detection_system = DetectionSystem(model_path=model_path, conf_threshold=conf_threshold)
                        self.detection_system.main_ui = self
                        print("✅ DetectionSystem initialized successfully")
                        self.add_activity_log("✅ DetectionSystem initialized successfully")
                    except Exception as e:
                        print(f"❌ Failed to initialize DetectionSystem: {e}")
                        import traceback; traceback.print_exc()
                        self.add_activity_log(f"❌ Failed to initialize DetectionSystem: {e}")
                        return

                # Start detection with UI callback
                success = False
                try:
                    # Tune detection service for fast startup if available
                    try:
                        svc = getattr(self.detection_system, 'detection_service', None)
                        if svc:
                            svc.frame_skip = 1
                            svc.process_interval = 0.05
                            print("🔧 Tuned detection service for fast startup (frame_skip=1, process_interval=0.05)")
                    except Exception:
                        pass

                    success = self.detection_system.start_detection(
                        person_id, person_name, person_type, ui_callback=self.update_camera_feed
                    )
                except Exception as e:
                    print(f"⚠️ start_detection raised: {e}")

                if success:
                    # Mark detection active in main UI state
                    try:
                        self.detection_active = True
                        self.current_person_id = person_id
                        self.current_person_type = person_type
                        self.current_person_name = person_name
                    except Exception:
                        pass

                    print(f"✅ Integrated detection started successfully for {person_name}")
                    self.add_activity_log(f"✅ Integrated detection started successfully for {person_name}")
                    # Start watchdog to ensure completion (already present elsewhere)
                    try:
                        self._start_uniform_watchdog(30)
                    except Exception:
                        pass
                else:
                    print(f"❌ Failed to start integrated detection for {person_name}")
                    self.add_activity_log(f"❌ Failed to start integrated detection for {person_name}")
                    # Hide requirements section if detection fails
                    try:
                        self.hide_requirements_section()
                    except Exception:
                        pass

            except Exception as e:
                print(f"❌ Error starting integrated detection: {e}")
                self.add_activity_log(f"❌ Error starting integrated detection: {e}")

        threading.Thread(target=_worker, daemon=True).start()
        return

    def start_person_detection(self, person_id, person_name, person_type):
        """Start detection for a person using the integrated detection system"""
        try:
            print(f"🔍 Starting detection for {person_name} ({person_type})")
            self.add_activity_log(f"🔍 Starting detection for {person_name} ({person_type})")
            
            # Prefer zero-downtime handover: give detection the existing guard camera if running
            try:
                if not hasattr(self, 'detection_system'):
                    self.detection_system = DetectionSystem()
                    self.detection_system.main_ui = self
                svc = getattr(self.detection_system, 'detection_service', None)
                cap = getattr(self, 'guard_camera_cap', None)
                if svc and cap is not None and hasattr(svc, 'set_existing_camera'):
                    # Signal guard loop to stop but keep the cap for transfer
                    try:
                        self._guard_keep_cap_on_stop = True
                        self.guard_camera_active = False
                    except Exception:
                        pass

                    # Briefly wait for guard loop to quiesce
                    t0 = time.time()
                    while time.time() - t0 < 0.3:
                        try:
                            if not getattr(self, 'guard_camera_active', False):
                                break
                        except Exception:
                            break
                        time.sleep(0.01)

                    # Hand over cap to detection
                    if svc.set_existing_camera(cap):
                        try:
                            self._cap_handed_to_detection = True
                        except Exception:
                            pass
                        self.guard_camera_cap = None
                        print("🔁 Guard camera handed over to detection (no reopen)")
                else:
                    # fallback: stop guard preview if we cannot handover
                    self.stop_guard_camera_feed()
            except Exception:
                try:
                    self.stop_guard_camera_feed()
                except Exception:
                    pass

            # Initialize detection system
            if not hasattr(self, 'detection_system'):
                self.detection_system = DetectionSystem()
                self.detection_system.main_ui = self  # Pass main UI reference
            
            # Tune detection service for fast startup if available
            try:
                svc = getattr(self.detection_system, 'detection_service', None)
                if svc:
                    svc.frame_skip = 1
                    svc.process_interval = 0.015
                    # Preload most used models in background to remove first-use latency
                    try:
                        if hasattr(svc, 'preload_models') and not getattr(self, '_models_preloaded', False):
                            svc.preload_models([
                                'shs.pt',
                                'ict and eng.pt',
                                'bsba male.pt',
                                'bsba_female.pt',
                                'arts and science.pt',
                                'tourism male.pt',
                                'hm.pt'
                            ])
                            self._models_preloaded = True
                    except Exception:
                        pass
                    print("🔧 Tuned detection service for fast startup (frame_skip=1, process_interval=0.015)")
            except Exception:
                pass

            # Start detection with UI callback
            success = self.detection_system.start_detection(
                person_id, person_name, person_type, 
                ui_callback=self.update_camera_feed
            )
            
            if success:
                print(f"✅ Detection started successfully for {person_name}")
                self.add_activity_log(f"✅ Detection started successfully for {person_name}")
                # Start a 30-second watchdog to check if uniform detection completes
                try:
                    self._start_uniform_watchdog(30)
                except Exception:
                    pass
            else:
                print(f"❌ Failed to start detection for {person_name}")
                self.add_activity_log(f"❌ Failed to start detection for {person_name}")
                
        except Exception as e:
            print(f"❌ Error starting detection: {e}")
            self.add_activity_log(f"❌ Error starting detection: {e}")

    def stop_detection(self):
        """Stop detection"""
        try:
            if hasattr(self, 'detection_system') and self.detection_system:
                self.detection_system.stop_detection()
            # Detection stopped (logging suppressed to keep detection persistent unless explicitly requested)
            # Clear main UI detection context so subsequent taps are evaluated correctly
            try:
                self.detection_active = False
                self.current_person_id = None
                self.current_person_type = None
                self.current_person_name = None
            except Exception:
                pass
            # Restart guard camera feed (no detection) so system returns to standby
            try:
                self.initialize_guard_camera_feed()
            except Exception:
                pass
        except Exception as e:
            print(f"❌ Error stopping detection: {e}")
            self.add_activity_log(f"❌ Error stopping detection: {e}")

    def _start_uniform_watchdog(self, timeout_seconds=30):
        """Start a background watchdog that waits timeout_seconds and checks
        whether the BSCPE/uniform detection has completed. If not complete
        after the timeout and detection is still active, treat as INCOMPLETE
        and stop detection (minimal behavior per user request).
        """
        try:
            def _watchdog():
                try:
                    start_t = time.time()
                    while time.time() - start_t < float(timeout_seconds):
                        # If detection no longer active, exit early
                        if not getattr(self, 'detection_active', False):
                            return

                        # Inspect detection service for a bscpe tracker
                        svc = getattr(self, 'detection_system', None)
                        svc = getattr(svc, 'detection_service', None) if svc else None
                        tracker = getattr(svc, 'bscpe_tracker', None) if svc else None

                        if tracker is not None:
                            try:
                                # Prefer boolean is_complete() if available
                                if hasattr(tracker, 'is_complete') and callable(getattr(tracker, 'is_complete')):
                                    if tracker.is_complete():
                                        return
                                # Or fall back to get_verification_status() if present
                                if hasattr(tracker, 'get_verification_status') and callable(getattr(tracker, 'get_verification_status')):
                                    status = tracker.get_verification_status()
                                    # handle either tuple or single return
                                    if isinstance(status, tuple):
                                        if status and status[0] == 'COMPLETE':
                                            return
                                    else:
                                        if status == 'COMPLETE':
                                            return
                            except Exception:
                                # ignore tracker errors and continue waiting
                                pass

                        time.sleep(0.5)

                    # Timeout expired - act only if detection still active
                    if getattr(self, 'detection_active', False):
                        print(f"⏱️ Uniform detection watchdog: no completion after {timeout_seconds}s - marking INCOMPLETE (keeping detection active)")
                        # Attempt to show incomplete popup via main_ui if available
                        try:
                            current_rfid = None
                            svc_full = getattr(self, 'detection_system', None)
                            if svc_full and getattr(svc_full, 'detection_service', None):
                                current_rfid = getattr(svc_full.detection_service, 'current_rfid', None)

                            student_info = None
                            if hasattr(self, 'main_ui') and self.main_ui:
                                try:
                                    student_info = self.main_ui.get_student_info_by_rfid(current_rfid)
                                except Exception:
                                    student_info = None

                            if student_info and hasattr(self.main_ui, 'show_incomplete_uniform_popup'):
                                try:
                                    self.main_ui.show_incomplete_uniform_popup(student_info)
                                except Exception:
                                    pass
                        except Exception:
                            pass

                        # Do NOT stop detection automatically. Instead, show popup (done above)
                        # and reset the BSCPE tracker so detection remains active and the
                        # student can retry without camera reopen.
                        try:
                            svc_full = getattr(self, 'detection_system', None)
                            svc = getattr(svc_full, 'detection_service', None) if svc_full else None
                            if svc and getattr(svc, 'bscpe_tracker', None):
                                try:
                                    svc.bscpe_tracker.reset(svc.bscpe_tracker.course_type)
                                    svc.bscpe_tracker.start_time = time.time()
                                    print("🔄 Watchdog: BSCPE tracker reset, detection remains active")
                                except Exception as e:
                                    print(f"⚠️ Watchdog: failed to reset tracker: {e}")
                        except Exception:
                            pass

                except Exception as e:
                    print(f"⚠️ Watchdog internal error: {e}")

            t = threading.Thread(target=_watchdog, daemon=True)
            t.start()
        except Exception as e:
            print(f"⚠️ Failed to start uniform watchdog: {e}")

    def stop_camera_for_guard(self):
        """Stop camera for guard"""
        pass
    
    def update_camera_label_for_guard(self):
        """Update camera label to show guard camera status"""
        pass
    
    def ensure_camera_closed(self):
        """Ensure camera is closed and in standby mode"""
        pass
    
    def initialize_guard_camera_feed(self):
        """Initialize guard camera feed - hide requirements section since no student is active."""
        try:
            # Ensure any running guard camera loop is stopped
            try:
                self.stop_guard_camera_feed()
            except Exception:
                pass

            # Keep requirements section visible - only hide when gate control button is clicked
            # Don't hide when detection stops - requirements stay visible until guard clicks a button
            # if not self.requirements_hide_scheduled:
            #     self.hide_requirements_section()

            # Update the UI label to show that preview is disabled (external window is used)
            if hasattr(self, 'camera_label') and self.camera_label:
                standby_text = (
                    "📷 CAMERA PREVIEW DISABLED\n\n"
                    "Detection runs in the external Camera Detection window.\n"
                    "Tap a student RFID to start detection.\n\n"
                    "Press 'q' in the detection window to close it."
                )
                try:
                    self.camera_label.config(text=standby_text, bg='#dbeafe', fg='#374151')
                    self.camera_label.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
                except Exception:
                    pass
            print("✅ UI set to standby (no in-UI camera)")
            self.add_activity_log("✅ UI set to standby (no in-UI camera)")
        except Exception as e:
            print(f"❌ Error setting UI standby: {e}")

    def prepare_detection_for_quick_start(self, model_path=None):
        """Prepare/load detection model in background so detection starts faster on RFID tap.
        This loads the model into memory but does not open the camera (avoids device conflicts).
        """
        try:
            if model_path is None:
                model_path = getattr(self, 'current_model_path', 'bsba_female.pt')
            
            # Verify model file exists before attempting to load
            import os
            if not os.path.exists(model_path):
                print(f"⚠️ Preload: Model file not found: {model_path}, skipping preload")
                return

            def _warmup():
                try:
                    print(f"🔧 Preloading detection model for quick start: {model_path}")
                    # Initialize detection system if missing
                    if not hasattr(self, 'detection_system') or self.detection_system is None:
                        self.detection_system = DetectionSystem(model_path=model_path, conf_threshold=self.conf_threshold)
                        self.detection_system.main_ui = self

                    # Load model into detection service (this calls DetectionSystem.load_model)
                    try:
                        self.detection_system.load_model()
                    except Exception as e:
                        print(f"⚠️ Preload: failed to load model: {e}")

                    # Tune detection service parameters for faster first-frame detection
                    try:
                        svc = getattr(self.detection_system, 'detection_service', None)
                        if svc:
                            svc.frame_skip = 1
                            svc.process_interval = 0.05
                            print("🔧 Preload: detection service tuned for fast startup (frame_skip=1, process_interval=0.05)")
                    except Exception:
                        pass

                    # Optionally perform a tiny dummy inference to warm model caches (CPU/GPU)
                    try:
                        if YOLO_AVAILABLE and hasattr(self.detection_system, 'detection_service') and self.detection_system.detection_service.model:
                            import numpy as _np
                            dummy = _np.zeros((480, 640, 3), dtype=_np.uint8)
                            try:
                                _ = self.detection_system.detection_service.model(dummy, conf=self.detection_system.conf_threshold)
                                print("🔧 Preload: dummy inference completed")
                            except Exception:
                                pass
                    except Exception:
                        pass

                except Exception as e:
                    print(f"⚠️ Error during detection warmup: {e}")

            t = threading.Thread(target=_warmup, daemon=True)
            t.start()
        except Exception as e:
            print(f"⚠️ prepare_detection_for_quick_start error: {e}")

    def initialize_live_camera_feed(self):
        """Initialize live camera feed for continuous monitoring"""
        try:
            # Initialize detection system for live feed
            if not hasattr(self, 'detection_system') or self.detection_system is None:
                self.detection_system = DetectionSystem()
                self.detection_system.main_ui = self  # Pass main UI reference
                print("✅ DetectionSystem initialized for live feed")
            
            # Start live camera feed (without specific person detection)
            success = self.detection_system.start_live_feed(ui_callback=self.update_camera_feed)
            if success:
                print("✅ Live camera feed started")
                self.add_activity_log("✅ Live camera feed started")
            else:
                print("❌ Failed to start live camera feed")
                self.add_activity_log("❌ Failed to start live camera feed")
                # Try fallback method
                self.start_fallback_camera_feed()
            
        except Exception as e:
            print(f"❌ Error initializing live camera feed: {e}")
            self.add_activity_log(f"❌ Error initializing live camera feed: {e}")
            # Try fallback method
            self.start_fallback_camera_feed()
    
    def start_guard_camera_feed(self):
        """Start camera feed for guard monitoring (no detection)"""
        try:
            print("🔧 Starting guard camera feed...")
            self.add_activity_log("🔧 Starting guard camera feed...")
            
            # Simple camera feed without detection
            import cv2
            import threading
            
            def guard_camera_loop():
                cap = None
                try:
                    self.guard_camera_active = True
                    camera_index = 0  # start with camera 0
                    # If a VideoCapture was already returned from detection, reuse it to avoid toggling device
                    existing_cap = getattr(self, 'guard_camera_cap', None)
                    if existing_cap is not None and hasattr(existing_cap, 'isOpened') and existing_cap.isOpened():
                        cap = existing_cap
                        print("🔁 Reusing returned VideoCapture for guard feed")
                    else:
                        # Try to open camera (prefer DirectShow on Windows)
                        try:
                            if platform.system().lower().startswith('win') and hasattr(cv2, 'CAP_DSHOW'):
                                cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
                            else:
                                cap = cv2.VideoCapture(camera_index)
                            # reduce internal buffer where supported to make reads responsive
                            try:
                                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            except Exception:
                                pass
                        except Exception:
                            cap = None

                    if not cap or not cap.isOpened():
                        print(f"❌ Camera index {camera_index} failed to open for guard feed, trying fallback index 0")
                        try:
                            if platform.system().lower().startswith('win') and hasattr(cv2, 'CAP_DSHOW'):
                                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                            else:
                                cap = cv2.VideoCapture(0)
                        except Exception:
                            cap = None

                    if not cap or not cap.isOpened():
                        print("❌ Guard camera failed to open")
                        return

                    # Expose cap for other parts of the app to inspect/release
                    try:
                        with self._guard_cap_lock:
                            self.guard_camera_cap = cap
                    except Exception:
                        # Fallback: set without lock if lock missing
                        self.guard_camera_cap = cap
                    print(f"✅ Guard camera opened (index {camera_index}) for monitoring")

                    frame_count = 0
                    while getattr(self, 'guard_camera_active', True):
                        ret, frame = False, None
                        try:
                            ret, frame = cap.read()
                        except Exception:
                            # reading failed (possibly due to a release) — treat as temporary failure
                            ret = False

                        if not ret or frame is None:
                            # small sleep to avoid busy loop if camera temporarily fails
                            time.sleep(0.03)
                            continue

                        frame_count += 1
                        
                        # Always show plain guard monitoring status in UI (no detection overlay)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(frame, "GUARD MONITORING", (10, 30), font, 0.8, (0, 255, 255), 2)
                        cv2.putText(frame, "Camera Active - No Detection", (10, 60), font, 0.6, (255, 255, 255), 2)
                        cv2.putText(frame, "Waiting for student RFID...", (10, 90), font, 0.5, (0, 255, 0), 2)

                                # Update UI
                        try:
                            self.update_camera_feed(frame)
                        except Exception:
                            pass

                        # Debug: Print every 60 frames
                        if frame_count % 60 == 0:
                            try:
                                print(f"📹 Guard camera frame {frame_count} - Camera active: {self.guard_camera_cap is not None and self.guard_camera_cap.isOpened()}")
                            except Exception:
                                pass

                        time.sleep(0.033)

                except Exception as e:
                    print(f"❌ Guard camera error: {e}")
                finally:
                    try:
                        if cap is not None:
                            # If this cap was handed to detection, do not release it here;
                            # detection now owns the handle and will return it when done.
                            if getattr(self, '_cap_handed_to_detection', False):
                                try:
                                    print("ℹ️ Guard camera cap handed to detection - not releasing here")
                                except Exception:
                                    pass
                            else:
                                try:
                                    cap.release()
                                except Exception:
                                    pass
                    finally:
                        # Clear shared cap reference and active flag
                        try:
                            # If cap was handed to detection, detection will manage the cap and
                            # main_ui.guard_camera_cap is set when reclaimed; clear local ref.
                            self.guard_camera_cap = None
                        except Exception:
                            pass
                        try:
                            self.guard_camera_active = False
                        except Exception:
                            pass
                    print("🛑 Guard camera loop ended")
            
            # Start guard camera in separate thread
            self.guard_camera_thread = threading.Thread(target=guard_camera_loop, daemon=True)
            self.guard_camera_thread.start()
            print("✅ Guard camera feed started")
            
        except Exception as e:
            print(f"❌ Guard camera feed failed: {e}")
            self.add_activity_log(f"❌ Guard camera feed failed: {e}")

    def start_fallback_camera_feed(self):
        """Fallback camera feed when main detection system fails"""
        try:
            print("🔄 Starting fallback camera feed...")
            self.add_activity_log("🔄 Starting fallback camera feed...")
            
            # Simple camera feed without detection
            import cv2
            import threading
            
            def fallback_camera_loop():
                try:
                    # Use working camera configuration
                    camera_index = 0  # Camera 0 works with DirectShow
                    cap = cv2.VideoCapture(camera_index)
                    if not cap.isOpened():
                        print(f"❌ External camera (index {camera_index}) not found, trying built-in camera")
                        camera_index = 0  # Fallback to built-in camera
                        cap = cv2.VideoCapture(camera_index)
                        if not cap.isOpened():
                            print("❌ Fallback camera failed to open")
                            return
                    else:
                        print(f"✅ Using external camera (index {camera_index}) for fallback")
                    
                    # Use camera with default settings - no filters or adjustments
                    
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            continue
                        
                        # Add fallback status
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(frame, "FALLBACK CAMERA FEED", (10, 30), font, 0.8, (0, 255, 255), 2)
                        cv2.putText(frame, "Basic camera only", (10, 60), font, 0.6, (255, 255, 255), 2)
                        
                        # Update UI
                        self.update_camera_feed(frame)
                        time.sleep(0.033)
                        
                except Exception as e:
                    print(f"❌ Fallback camera error: {e}")
                finally:
                    if 'cap' in locals():
                        cap.release()
            
            # Start fallback camera in separate thread
            fallback_thread = threading.Thread(target=fallback_camera_loop, daemon=True)
            fallback_thread.start()
            print("✅ Fallback camera feed started")
            
        except Exception as e:
            print(f"❌ Fallback camera feed failed: {e}")
            self.add_activity_log(f"❌ Fallback camera feed failed: {e}")
    
    def stop_guard_camera_feed(self):
        """Stop guard camera feed with robust cleanup"""
        try:
            # Quietly stop the guard camera feed (no noisy stop message)
            
            # Signal the guard camera loop to stop
            try:
                self.guard_camera_active = False
                print("✅ Guard camera active flag set to False")
            except Exception as e:
                print(f"⚠️ Error setting guard camera active flag: {e}")

            # If there is an exposed cap, try to release it with multiple attempts
            try:
                if getattr(self, 'guard_camera_cap', None) is not None:
                    # If a handoff to detection was requested, leave the cap for transfer
                    if getattr(self, '_guard_keep_cap_on_stop', False) or getattr(self, '_cap_handed_to_detection', False):
                        print("ℹ️ Guard camera cap handoff requested - not releasing cap here")
                    else:
                        print("🔄 Releasing guard camera cap...")
                        try:
                            with self._guard_cap_lock:
                                try:
                                    self.guard_camera_cap.release()
                                    print(f"✅ Guard camera released")
                                except Exception as e:
                                    print(f"⚠️ Guard camera release failed: {e}")
                                # Clear the cap reference under lock
                                self.guard_camera_cap = None
                        except Exception:
                            # Fallback: attempt release without lock
                            try:
                                self.guard_camera_cap.release()
                            except Exception:
                                pass
                            self.guard_camera_cap = None
                        # Force garbage collection
                        import gc
                        gc.collect()
                        print("✅ Guard camera cap cleared")
                else:
                    print("ℹ️ No guard camera cap to release")
            except Exception as e:
                print(f"⚠️ Error releasing guard camera cap: {e}")

            # Do not block waiting for the thread; signal it to stop and release resources
            try:
                if hasattr(self, 'guard_camera_thread') and self.guard_camera_thread:
                    # let thread finish asynchronously; clear reference so future starts create a fresh thread
                    if not self.guard_camera_thread.is_alive():
                        self.guard_camera_thread = None
            except Exception as e:
                print(f"⚠️ Error handling guard camera thread cleanup: {e}")

            print("✅ Guard camera feed stopped")
            # Record activity without noisy emoji
            self.add_activity_log("Guard camera feed stopped")
            
        except Exception as e:
            print(f"❌ Error stopping guard camera feed: {e}")

    def switch_to_bsba_student_detection(self, rfid, person_name, course, gender):
        """Switch to BSBA student detection with appropriate model"""
        try:
            print(f"🔍 Switching to BSBA detection for {person_name} ({course}, {gender})")
            self.add_activity_log(f"🔍 Switching to BSBA detection for {person_name} ({course}, {gender})")
            
            # Determine the correct model based on gender
            if gender.upper() in ["MALE", "M"]:
                model_name = "bsba male.pt"
                model_key = "BSBA_MALE"
            elif gender.upper() in ["FEMALE", "F"]:
                model_name = "bsba_female.pt"
                model_key = "BSBA_FEMALE"
            else:
                # Default to male if gender is unclear
                model_name = "bsba male.pt"
                model_key = "BSBA_MALE"
                print(f"⚠️ Unknown gender '{gender}', defaulting to male model")
            
            print(f"📦 Selected model: {model_name} for {gender} student")
            self.add_activity_log(f"📦 Selected model: {model_name} for {gender} student")
            
            # Initialize detection system with specific model
            if not hasattr(self, 'detection_system') or self.detection_system is None:
                self.detection_system = DetectionSystem(model_path=model_name)
                self.detection_system.main_ui = self
            else:
                # Update model if different
                if self.detection_system.model_path != model_name:
                    self.detection_system.model_path = model_name
                    self.detection_system.model = YOLO(model_name)
            
            # Set current person info
            self.detection_system.current_person_id = rfid
            self.detection_system.current_person_name = person_name
            
            # Tune detection service for fast startup if available
            try:
                svc = getattr(self.detection_system, 'detection_service', None)
                if svc:
                    svc.frame_skip = 1
                    svc.process_interval = 0.05
                    print("🔧 Tuned detection service for fast startup (frame_skip=1, process_interval=0.05)")
            except Exception:
                pass

            # Start detection with BSBA model
            success = self.detection_system.start_detection(rfid, person_name, "student", ui_callback=self.update_camera_feed)
            
            if success:
                print(f"✅ BSBA {gender} detection started for {person_name}")
                self.add_activity_log(f"✅ BSBA {gender} detection started for {person_name}")
            else:
                print(f"❌ Failed to start BSBA detection for {person_name}")
                self.add_activity_log(f"❌ Failed to start BSBA detection for {person_name}")
                
        except Exception as e:
            print(f"❌ Error switching to BSBA student detection: {e}")
            self.add_activity_log(f"❌ Error switching to BSBA student detection: {e}")

    def switch_to_student_detection(self, rfid, person_name, course, gender):
        """Switch from live feed to specific student detection"""
        try:
            if hasattr(self, 'detection_system') and self.detection_system:
                # Stop current live feed
                self.detection_system.stop_detection()
                
                # Set current person info for model selection
                self.detection_system.current_person_id = rfid
                self.detection_system.current_person_name = person_name
                
                # Tune detection service for fast startup if available
                try:
                    svc = getattr(self.detection_system, 'detection_service', None)
                    if svc:
                        svc.frame_skip = 1
                        svc.process_interval = 0.05
                        print("🔧 Tuned detection service for fast startup (frame_skip=1, process_interval=0.05)")
                except Exception:
                    pass

                # Start detection with appropriate model
                self.detection_system.start_detection(rfid, person_name, "student", ui_callback=self.update_camera_feed)
                print(f"✅ Switched to {course} {gender} detection for {person_name}")
                self.add_activity_log(f"✅ Switched to {course} {gender} detection for {person_name}")
            else:
                print("❌ Detection system not available")
                self.add_activity_log("❌ Detection system not available")
                
        except Exception as e:
            print(f"❌ Error switching to student detection: {e}")
            self.add_activity_log(f"❌ Error switching to student detection: {e}")

    def start_bscpe_detection(self, rfid, person_name, course, gender):
        """Start BSCPE detection with uniform verification.
        Run the camera teardown/setup and detection start in a background
        thread to avoid blocking the main UI on RFID taps.
        """
        def _worker():
            try:
                print(f"🎓 Starting BSCPE detection for {person_name} ({course}, {gender})")
                self.add_activity_log(f"🎓 Starting BSCPE detection for {person_name} ({course}, {gender})")

                # Gentle stop of existing detection (non-blocking)
                try:
                    if hasattr(self, 'detection_system') and self.detection_system:
                        print("🛑 Stopping existing detection system...")
                        self.detection_system.stop_detection()
                        time.sleep(0.05)
                except Exception:
                    pass

                # Signal guard camera loop to stop and release resources (respect handoff flags)
                try:
                    self.guard_camera_active = False
                    if getattr(self, 'guard_camera_cap', None):
                        # If a handoff to detection was requested, do not release here
                        if getattr(self, '_guard_keep_cap_on_stop', False) or getattr(self, '_cap_handed_to_detection', False):
                            print("ℹ️ Guard camera cap handoff requested in start_bscpe_detection - not releasing cap here")
                        else:
                            try:
                                self.guard_camera_cap.release()
                            except Exception:
                                pass
                            self.guard_camera_cap = None
                except Exception:
                    pass

                # Try to join the guard camera thread briefly
                try:
                    if getattr(self, 'guard_camera_thread', None):
                        self.guard_camera_thread.join(timeout=0.3)
                        self.guard_camera_thread = None
                except Exception:
                    pass

                # Run GC to free handles
                try:
                    import gc
                    gc.collect()
                except Exception:
                    pass

                # Initialize detection system if missing
                try:
                    if not hasattr(self, 'detection_system') or self.detection_system is None:
                        self.detection_system = DetectionSystem()
                        self.detection_system.main_ui = self
                except Exception as e:
                    print(f"❌ Failed to create DetectionSystem: {e}")
                    self.initialize_guard_camera_feed()
                    return

                # Reset detection service
                try:
                    if hasattr(self.detection_system, 'detection_service') and self.detection_system.detection_service:
                        self.detection_system.detection_service.reset_for_new_detection()
                except Exception:
                    pass

                # Start detection camera
                try:
                    ds = getattr(self.detection_system, 'detection_service', None)
                    if not ds or not ds.start_camera():
                        print("❌ Failed to start detection camera")
                        self.initialize_guard_camera_feed()
                        return
                except Exception as e:
                    print(f"❌ Exception starting detection camera: {e}")
                    self.initialize_guard_camera_feed()
                    return

                # Configure BSCPE verification
                try:
                    svc = self.detection_system.detection_service
                    svc.set_bscpe_verification(course, gender, person_name)
                    svc.current_rfid = rfid
                except Exception:
                    print("❌ Detection service not available or failed to configure BSCPE")
                    self.initialize_guard_camera_feed()
                    return

                # Start detection (tune service for fast startup first)
                try:
                    try:
                        svc = getattr(self.detection_system, 'detection_service', None)
                        if svc:
                            svc.frame_skip = 1
                            svc.process_interval = 0.05
                            print("🔧 Tuned detection service for fast startup (frame_skip=1, process_interval=0.05)")
                    except Exception:
                        pass

                    success = self.detection_system.start_detection(rfid, person_name, "student", ui_callback=self.update_camera_feed)
                except Exception as e:
                    print(f"⚠️ start_detection exception: {e}")
                    success = False

                if success:
                    print(f"✅ BSCPE detection started for {person_name}")
                    self.add_activity_log(f"✅ BSCPE detection started for {person_name}")
                else:
                    print(f"❌ Failed to start BSCPE detection for {person_name}")
                    self.add_activity_log(f"❌ Failed to start BSCPE detection for {person_name}")
                    self.initialize_guard_camera_feed()

            except Exception as e:
                print(f"❌ Error starting BSCPE detection: {e}")
                self.add_activity_log(f"❌ Error starting BSCPE detection: {e}")
                try:
                    self.initialize_guard_camera_feed()
                except Exception:
                    pass

        threading.Thread(target=_worker, daemon=True).start()
        return
    
    def start_security_system(self):
        """Start the security dashboard"""
        if not self.current_guard_id:
            messagebox.showerror("Error", "Please login first")
            return
        
        try:
            print(f"🔍 Starting security system for guard: {self.current_guard_id}")
            
            # Create dashboard tab if it doesn't exist
            if self.dashboard_tab is None:
                print("🔍 Creating dashboard tab...")
                self.create_dashboard_tab()
                print("SUCCESS: Dashboard tab created")
            else:
                print("SUCCESS: Dashboard tab already exists")
            
            # Update guard name display after dashboard is created
            if hasattr(self, 'current_guard_id') and self.current_guard_id:
                self.update_guard_name_display(self.current_guard_id)
            
            # Switch to dashboard tab
            print("🔍 Switching to dashboard tab...")
            self.notebook.select(self.dashboard_tab)
            self.notebook.hide(self.login_tab)
            print("SUCCESS: Switched to dashboard tab")
            
            # Initialize camera for guard (no detection, just live feed)
            print("🔍 Initializing camera for guard monitoring...")
            self.initialize_guard_camera_feed()
            print("SUCCESS: Guard camera feed initialized")
            
            # Open main screen for entry/exit monitoring
            print("🔍 Opening main screen...")
            self.open_main_screen()
            print("SUCCESS: Main screen opened")
            
            # Set focus on hidden RFID Entry field to capture RFID input globally
            # Use after() to ensure UI is fully ready before setting focus
            self.root.after(500, self.set_rfid_entry_focus)
            
            # Set up global focus management - refocus hidden field periodically
            # This ensures RFID input always goes to the hidden field
            self.setup_rfid_focus_management()
            
            print("SUCCESS: Security dashboard started successfully")
            
        except Exception as e:
            print(f"ERROR: Error starting security dashboard: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to start security dashboard: {e}")
    
    def open_main_screen(self):
        """Open the main screen for entry/exit monitoring"""
        try:
            if self.main_screen_window is None or not self.main_screen_window.winfo_exists():
                self.create_main_screen()
            else:
                # Bring existing window to front
                self.main_screen_window.lift()
                self.main_screen_window.focus_force()
        except Exception as e:
            print(f"Error opening main screen: {e}")
            # Try to create a simple fallback window
            try:
                self.create_simple_main_screen()
            except Exception as e2:
                print(f"Failed to create fallback main screen: {e2}")
    
    def create_main_screen(self):
        """Create the main screen window for entry/exit monitoring"""
        try:
            # Create new window
            self.main_screen_window = tk.Toplevel(self.root)
            self.main_screen_window.title("AI-niform - Main Screen")
            self.main_screen_window.geometry("1200x800")
            self.main_screen_window.configure(bg='#f0f8ff')
            
            # Position Main Screen on secondary monitor if available
            if self.screen_info['secondary']:
                secondary = self.screen_info['secondary']
                # Set fullscreen on secondary monitor
                self.main_screen_window.geometry(f"{secondary['width']}x{secondary['height']}+{secondary['x']}+{secondary['y']}")
                
                # Try to make fullscreen on secondary monitor
                try:
                    # Method 1: Try fullscreen attribute (most reliable on Linux)
                    self.main_screen_window.attributes('-fullscreen', True)
                    print(f"SUCCESS: Main Screen fullscreen on secondary monitor: {secondary['width']}x{secondary['height']} at ({secondary['x']}, {secondary['y']})")
                except Exception as e1:
                    print(f"ERROR: Main Screen fullscreen attribute failed: {e1}")
                    try:
                        # Method 2: Try overrideredirect (removes window decorations)
                        self.main_screen_window.overrideredirect(True)
                        self.main_screen_window.geometry(f"{secondary['width']}x{secondary['height']}+{secondary['x']}+{secondary['y']}")
                        print(f"SUCCESS: Main Screen fullscreen on secondary monitor using overrideredirect: {secondary['width']}x{secondary['height']}")
                    except Exception as e2:
                        print(f"ERROR: Main Screen overrideredirect failed: {e2}")
                        try:
                            # Method 3: Try zoomed attribute
                            self.main_screen_window.attributes('-zoomed', True)
                            print(f"SUCCESS: Main Screen maximized on secondary monitor: {secondary['width']}x{secondary['height']}")
                        except Exception as e3:
                            print(f"ERROR: Main Screen zoomed failed: {e3}")
                            print(f"🖥️ Main Screen positioned on secondary monitor at ({secondary['x']}, {secondary['y']})")
            else:
                # Fallback: position on primary monitor but offset
                primary = self.screen_info['primary']
                offset_x = primary['width'] // 4
                offset_y = primary['height'] // 4
                self.main_screen_window.geometry(f"1200x800+{offset_x}+{offset_y}")
                print(f"🖥️ Main Screen positioned on primary monitor at ({offset_x}, {offset_y})")
                
        except Exception as e:
            print(f"ERROR: Failed to create main screen window: {e}")
            # Try fallback approach
            try:
                self.create_simple_main_screen()
                return
            except Exception as e2:
                print(f"ERROR: Failed to create fallback main screen: {e2}")
                return
        
        # Add keyboard controls for Main Screen fullscreen
        self.main_screen_window.bind('<F11>', self.toggle_main_screen_fullscreen)
        self.main_screen_window.bind('<Escape>', self.exit_main_screen_fullscreen)
        
        # Prevent window from being closed accidentally
        self.main_screen_window.protocol("WM_DELETE_WINDOW", self.minimize_main_screen)
        
        # Prevent window from minimizing when switching to it (Alt+Tab)
        self.main_screen_window.bind('<FocusIn>', self.on_main_screen_focus_in)
        self.main_screen_window.bind('<Map>', self.on_main_screen_map)
        
        # Create main screen content
        self.create_main_screen_content()
        
        # Start monitoring loop
        self.start_main_screen_monitoring()
    
    def create_simple_main_screen(self):
        """Create a simple fallback main screen"""
        try:
            self.main_screen_window = tk.Toplevel(self.root)
            self.main_screen_window.title("AI-niform - Main Screen (Simple)")
            self.main_screen_window.geometry("800x600")
            self.main_screen_window.configure(bg='#f0f8ff')
            
            # Simple content
            label = tk.Label(
                self.main_screen_window,
                text="AI-niform Main Screen\n\nSimple fallback mode\n\nSystem is running normally",
                font=('Arial', 16),
                bg='#f0f8ff',
                fg='#1e3a8a'
            )
            label.pack(expand=True)
            
            print("SUCCESS: Simple main screen created")
            
        except Exception as e:
            print(f"ERROR: Failed to create simple main screen: {e}")
    
    def create_main_screen_content(self):
        """Create the main screen content"""
        # Header
        header_frame = tk.Frame(self.main_screen_window, bg='#1e3a8a', height=80)
        header_frame.pack(fill=tk.X, padx=0, pady=0)
        header_frame.pack_propagate(False)
        
        # Title
        title_label = tk.Label(
            header_frame,
            text="AI-niform Security System",
            font=('Arial', 24, 'bold'),
            fg='#ffffff',
            bg='#1e3a8a'
        )
        title_label.pack(side=tk.LEFT, padx=20, pady=20)
        
        # Current time
        self.time_label = tk.Label(
            header_frame,
            text="",
            font=('Arial', 16, 'bold'),
            fg='#ffffff',
            bg='#1e3a8a'
        )
        self.time_label.pack(side=tk.RIGHT, padx=20, pady=20)
        
        # Main content area
        main_content = tk.Frame(self.main_screen_window, bg='#f0f8ff')
        main_content.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Current person display (large center area)
        current_person_frame = tk.LabelFrame(
            main_content,
            text="Current Person",
            font=('Arial', 18, 'bold'),
            fg='#1e3a8a',
            bg='#ffffff',
            relief='solid',
            bd=2
        )
        current_person_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # Person info display
        self.person_display_frame = tk.Frame(current_person_frame, bg='#ffffff')
        self.person_display_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(20, 10))
        
        # Initialize with standby message
        self.show_standby_message()
        
        # Create persistent Activity Log frame (outside person_display_frame so it won't be destroyed)
        self.persistent_activity_log_frame = tk.LabelFrame(
            current_person_frame,
            text="ACTIVITY LOG",
            font=('Arial', 14, 'bold'),
            fg='#1e3a8a',
            bg='#ffffff',
            relief='solid',
            bd=2
        )
        self.persistent_activity_log_frame.pack(fill=tk.BOTH, expand=False, padx=20, pady=(0, 20))
        
        # Create persistent activity log listbox
        self.recent_entries_listbox = tk.Listbox(
            self.persistent_activity_log_frame,
            font=('Arial', 12),
            bg='#ffffff',
            fg='#1e3a8a',
            selectbackground='#3b82f6',
            height=8  # Fixed height
        )
        self.recent_entries_listbox.pack(fill=tk.BOTH, expand=False, padx=10, pady=10)
        
        # Add scrollbar for activity log
        try:
            scrollbar = tk.Scrollbar(self.persistent_activity_log_frame, orient=tk.VERTICAL, command=self.recent_entries_listbox.yview)
            self.recent_entries_listbox.config(yscrollcommand=scrollbar.set)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        except Exception:
            pass
    
    def load_person_image(self, person_info, max_width=800, max_height=800):
        """Load and resize person image from local folder"""
        try:
            if not PIL_AVAILABLE:
                print("⚠️ PIL/Pillow not available - cannot load images")
                return None
                
            person_type = person_info.get('type', '').lower()
            rfid = person_info.get('rfid')
            
            # Determine folder based on person type
            if person_type == 'student':
                image_folder = 'student_images'
            elif person_type == 'teacher':
                image_folder = 'teacher_images'
            else:
                return None  # No images for visitors/other types
            
            if not rfid:
                print(f"⚠️ No RFID provided for {person_type} image loading")
                return None
            
            # Try different image formats
            image_formats = ['.jpg', '.jpeg', '.png']
            image_path = None
            
            for ext in image_formats:
                potential_path = os.path.join(image_folder, f"{rfid}{ext}")
                if os.path.exists(potential_path):
                    image_path = potential_path
                    break
            
            if not image_path or not os.path.exists(image_path):
                print(f"⚠️ Image not found for {person_type} {rfid} in {image_folder}/")
                return None
            
            # Load and resize image to fill the box
            try:
                img = Image.open(image_path)
                # Resize to maximum dimensions to fill the picture frame
                # Use larger dimensions to ensure image fills the available space
                img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                print(f"✅ Loaded image for {person_type} {rfid}: {image_path} (size: {img.size})")
                return photo
            except Exception as e:
                print(f"⚠️ Error loading image {image_path}: {e}")
                return None
                
        except Exception as e:
            print(f"⚠️ Error in load_person_image: {e}")
            return None
    
    def show_standby_message(self):
        """Show standby message when no one is detected"""
        # Cancel any pending clear timer when going to standby mode
        if hasattr(self, 'main_screen_clear_timer') and self.main_screen_clear_timer is not None:
            try:
                self.root.after_cancel(self.main_screen_clear_timer)
                self.main_screen_clear_timer = None
            except Exception:
                pass
        
        # Clear previous content
        for widget in self.person_display_frame.winfo_children():
            widget.destroy()
        
        standby_label = tk.Label(
            self.person_display_frame,
            text="🔒 SYSTEM STANDBY\n\nWaiting for person detection...",
            font=('Arial', 32, 'bold'),
            fg='#1e3a8a',
            bg='#ffffff',
            justify=tk.CENTER
        )
        standby_label.pack(expand=True)
    
    def display_person_info(self, person_data):
        """Display person information on main screen"""
        # Clear previous content
        for widget in self.person_display_frame.winfo_children():
            widget.destroy()
        
        # Person type and status
        status_frame = tk.Frame(self.person_display_frame, bg='#ffffff')
        status_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Person type
        person_type_label = tk.Label(
            status_frame,
            text=f"Type: {person_data.get('type', 'Unknown').upper()}",
            font=('Arial', 20, 'bold'),
            fg='#1e3a8a',
            bg='#ffffff'
        )
        person_type_label.pack(side=tk.LEFT)
        
        # Determine entry/exit/action consistently from either 'action' or 'status'
        # For students, check detection_active flag to show DETECTING instead of ENTRY
        action = person_data.get('action')
        status_val = str(person_data.get('status', '')).upper()
        
        if not action:
            if status_val in ('TIME-IN', 'ENTRY', 'COMPLETE UNIFORM'):
                action = 'ENTRY'
            elif status_val in ('TIME-OUT', 'EXIT', 'EXITED'):
                action = 'EXIT'
            elif status_val in ('DETECTING', 'INCOMPLETE UNIFORM'):
                # Don't convert DETECTING to ENTRY - keep it as DETECTING
                action = 'DETECTING'
            else:
                action = person_data.get('status', 'UNKNOWN')
        
        # For students, check if detection is active and uniform is not complete
        if person_data.get('type', '').lower() == 'student':
            detection_active = getattr(self, 'detection_active', False)
            uniform_complete = getattr(self, 'uniform_detection_complete', False)
            
            # CRITICAL: If status is TIME-IN or ENTRY, don't show DETECTING even if detection_active is True
            # This happens when entry is saved but detection_active flag wasn't reset yet
            status_from_data = str(person_data.get('status', '')).upper()
            is_entry_status = status_from_data in ('TIME-IN', 'ENTRY', 'TIME-IN (WITH VIOLATION)', 'TIME-IN (EVENT MODE)')
            
            if detection_active and not uniform_complete and not is_entry_status:
                # Detection in progress - show DETECTING (but not if entry was already saved)
                status_text = "DETECTING"
                status_color = '#f59e0b'  # Orange/amber
            elif person_data.get('status') == 'COMPLETE UNIFORM' or uniform_complete:
                # Complete uniform - show ENTRY
                status_text = "ENTRY"
                status_color = '#10b981'  # Green
            elif str(action).upper() == 'ENTRY' or is_entry_status:
                # Entry - prioritize entry status over DETECTING
                # If status is TIME-IN, always show ENTRY, not DETECTING
                if is_entry_status:
                    status_text = "ENTRY"
                    status_color = '#10b981'  # Green for entry
                elif status_val == 'DETECTING':
                    status_text = "DETECTING"
                    status_color = '#f59e0b'  # Orange/amber
                else:
                    status_text = f"Status: {action}"
                    status_color = '#10b981'  # Green for entry
            elif str(action).upper() == 'DETECTING':
                status_text = "DETECTING"
                status_color = '#f59e0b'  # Orange/amber
            else:
                status_text = f"Status: {action}"
                status_color = '#f59e0b' if str(action).upper() != 'EXIT' else '#ef4444'
        else:
            # For non-students, use original logic
            if person_data.get('status') == 'COMPLETE UNIFORM' or str(action).upper() == 'COMPLETE UNIFORM':
                status_color = '#10b981'  # Green for complete uniform
                status_text = "COMPLETE UNIFORM SUCCESS:"
            elif str(action).upper() == 'ENTRY':
                status_color = '#10b981'  # Green for entry
                status_text = f"Status: {action}"
            else:
                status_color = '#f59e0b'  # Orange for exit/other
                status_text = f"Status: {action}"
        
        status_label = tk.Label(
            status_frame,
            text=status_text,
            font=('Arial', 20, 'bold'),
            fg=status_color,
            bg='#ffffff'
        )
        status_label.pack(side=tk.RIGHT)
        
        # Person details
        details_frame = tk.Frame(self.person_display_frame, bg='#ffffff')
        details_frame.pack(fill=tk.BOTH, expand=True)
        
        # Name
        name_label = tk.Label(
            details_frame,
            text=f"Name: {person_data.get('name', 'Unknown')}",
            font=('Arial', 24, 'bold'),
            fg='#1e3a8a',
            bg='#ffffff'
        )
        name_label.pack(pady=(0, 10))
        
        # ID
        id_label = tk.Label(
            details_frame,
            text=f"ID: {person_data.get('id', 'Unknown')}",
            font=('Arial', 18, 'bold'),
            fg='#374151',
            bg='#ffffff'
        )
        id_label.pack(pady=(0, 10))
        
        # Additional info based on type
        if person_data.get('type') == 'student':
            # Display 'Senior High School' label when course indicates SHS (e.g., 'SHS STEM')
            course_value = str(person_data.get('course', 'Unknown'))
            if course_value.upper().startswith('SHS'):
                course_label_text = f"Senior High School: {course_value[3:].strip() or ''}".strip()
                # If there's no strand after 'SHS', just show 'Senior High School'
                if course_label_text.endswith(':') or course_label_text == 'Senior High School:':
                    course_label_text = 'Senior High School'
            else:
                course_label_text = f"Course: {course_value}"

            course_label = tk.Label(
                details_frame,
                text=course_label_text,
                font=('Arial', 16, 'bold'),
                fg='#3b82f6',
                bg='#ffffff'
            )
            course_label.pack(pady=(0, 5))
            
            gender_label = tk.Label(
                details_frame,
                text=f"Gender: {person_data.get('gender', 'Unknown')}",
                font=('Arial', 16, 'bold'),
                fg='#3b82f6',
                bg='#ffffff'
            )
            gender_label.pack(pady=(0, 5))
        
        elif person_data.get('type') == 'visitor':
            company_label = tk.Label(
                details_frame,
                text=f"Company: {person_data.get('company', 'N/A')}",
                font=('Arial', 16, 'bold'),
                fg='#3b82f6',
                bg='#ffffff'
            )
            company_label.pack(pady=(0, 5))
            
            purpose_label = tk.Label(
                details_frame,
                text=f"Purpose: {person_data.get('purpose', 'General Visit')}",
                font=('Arial', 16, 'bold'),
                fg='#3b82f6',
                bg='#ffffff'
            )
            purpose_label.pack(pady=(0, 5))
        
        # Time
        time_label = tk.Label(
            details_frame,
            text=f"Time: {person_data.get('timestamp', 'Unknown')}",
            font=('Arial', 14, 'bold'),
            fg='#1e3a8a',
            bg='#ffffff'
        )
        time_label.pack(pady=(20, 0))
        
        # DO NOT add to recent entries here - entry should only be logged after complete uniform
        # Entry logging happens in record_complete_uniform_entry() for students
        # Only non-students (visitors/teachers) should have immediate entry logging
    
    def add_to_recent_entries(self, person_data):
        """Add person to recent entries list"""
        from datetime import datetime

        try:
            # Debug: Log what we're receiving
            print(f"🔍 DEBUG add_to_recent_entries called:")
            print(f"   person_data keys: {list(person_data.keys())}")
            print(f"   person_data: {person_data}")

            timestamp = person_data.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            name = person_data.get('name', 'Unknown')
            person_type = person_data.get('type', 'Unknown')

            # Support both 'action' and legacy 'status' keys so recent list never shows 'Unknown'
            action = person_data.get('action')
            if not action:
                status = str(person_data.get('status', '')).upper()
                # DETECTING and INCOMPLETE UNIFORM should NOT be logged as ENTRY
                # Handle Event Mode status
                if 'TIME-IN' in status or 'EVENT MODE' in status or status in ('TIME-IN', 'ENTRY', 'COMPLETE UNIFORM'):
                    action = 'ENTRY'
                elif 'TIME-OUT' in status or status in ('TIME-OUT', 'EXIT', 'EXITED'):
                    action = 'EXIT'
                elif status in ('DETECTING', 'INCOMPLETE UNIFORM'):
                    # Don't log detection states - only log after completion
                    print(f"⚠️ Skipping entry: status is {status} (detection state)")
                    return  # Exit early, don't add to recent entries
                else:
                    action = person_data.get('status', 'Unknown')

            print(f"🔍 DEBUG: Resolved action = '{action}'")

            # Normalize a canonical person key so different callers produce the same id
            raw_id = person_data.get('id', name)
            person_id = str(raw_id).strip()
            # If numeric ID, strip leading zeros for canonicalization
            if person_id.isdigit():
                canonical_id = person_id.lstrip('0') or '0'
            else:
                canonical_id = person_id.lower()

            key = (canonical_id, str(action).upper())
            print(f"🔍 DEBUG: Generated key = {key} (canonical_id={canonical_id}, action={action})")

            # Prepare recent entries structures
            try:
                last_key = self.recent_entries_keys[0] if len(self.recent_entries_keys) > 0 else None
                print(f"🔍 DEBUG: last_key = {last_key}")
            except Exception as e:
                print(f"⚠️ Warning: Error getting last_key: {e}")
                last_key = None

            # Per-person last-action tracking to suppress near-simultaneous duplicates
            if not hasattr(self, 'last_logged_action'):
                # map: canonical_id -> (action, datetime)
                self.last_logged_action = {}

            # Check if this is Event Mode entry - always show Event Mode entries
            is_event_mode = person_data.get('event_mode', False) or getattr(self, 'event_mode_active', False)
            
            # Suppression window: 1 second for normal entries, always show Event Mode entries
            SUPPRESSION_WINDOW = 1.0 if not is_event_mode else 0.0  # seconds
            now = datetime.now()

            # For Event Mode, always show entries (skip duplicate checks)
            if is_event_mode:
                print(f"✅ Event Mode entry - skipping duplicate suppression")
            else:
                # If duplicate of most recent global entry, skip inserting (only for normal mode)
                if last_key == key:
                    print(f"⚠️ Skipping entry: duplicate of most recent entry (key={key})")
                    return

                # If same person had the same action very recently, skip to avoid duplicates
                try:
                    last_action, last_ts = self.last_logged_action.get(canonical_id, (None, None))
                    if last_action == str(action).upper() and last_ts and (now - last_ts).total_seconds() < SUPPRESSION_WINDOW:
                        # Suppress duplicate for same person/action within suppression window
                        elapsed = (now - last_ts).total_seconds()
                        print(f"⚠️ Skipping entry: same action '{action}' within suppression window ({elapsed:.1f}s < {SUPPRESSION_WINDOW}s)")
                        return
                except Exception as e:
                    print(f"⚠️ Warning: Error checking suppression window: {e}")

            entry_text = f"[{timestamp}] {name} ({person_type.upper()}) - {action}"
            print(f"🔍 DEBUG: Entry text = '{entry_text}'")

            # Check if listbox exists
            if not hasattr(self, 'recent_entries_listbox') or not self.recent_entries_listbox:
                print(f"❌ ERROR: recent_entries_listbox not found!")
                return

            # Insert at the beginning
            try:
                self.recent_entries_listbox.insert(0, entry_text)
                print(f"✅ Entry inserted into listbox at index 0")
            except Exception as e:
                print(f"❌ ERROR: Failed to insert into listbox: {e}")
                import traceback
                traceback.print_exc()
                return

            # Track the key
            try:
                self.recent_entries_keys.appendleft(key)
                print(f"✅ Key tracked: {key}")
            except Exception as e:
                print(f"⚠️ Warning: Could not track key: {e}")

            # Update per-person last action
            try:
                self.last_logged_action[canonical_id] = (str(action).upper(), now)
                print(f"✅ Last action updated for {canonical_id}: {action} at {now}")
            except Exception as e:
                print(f"⚠️ Warning: Could not update last action: {e}")

            # Keep only last 20 entries in UI listbox
            try:
                if self.recent_entries_listbox.size() > 20:
                    self.recent_entries_listbox.delete(20, tk.END)
            except Exception as e:
                print(f"⚠️ Warning: Could not trim listbox: {e}")

            # Force UI update
            try:
                self.recent_entries_listbox.update_idletasks()
                print(f"✅ UI updated after adding entry")
            except Exception as e:
                print(f"⚠️ Warning: Could not update UI: {e}")

            print(f"✅ SUCCESS: Entry '{entry_text}' added to recent entries")
            
        except Exception as e:
            print(f"❌ ERROR in add_to_recent_entries: {e}")
            import traceback
            traceback.print_exc()
    
    def start_main_screen_monitoring(self):
        """Start monitoring for main screen updates"""
        self.update_main_screen_time()
        # Schedule periodic updates
        self.main_screen_window.after(1000, self.start_main_screen_monitoring)
    
    def update_main_screen_time(self):
        """Update the time display on main screen"""
        if self.main_screen_window and self.main_screen_window.winfo_exists():
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.time_label.config(text=current_time)
    
    def minimize_main_screen(self):
        """Minimize main screen instead of closing"""
        if self.main_screen_window:
            self.main_screen_window.iconify()
    
    def on_main_screen_focus_in(self, event=None):
        """Handle main screen getting focus - ensure it stays visible"""
        try:
            if self.main_screen_window and self.main_screen_window.winfo_exists():
                # Deiconify (restore) if minimized
                try:
                    state = self.main_screen_window.state()
                    if state == 'iconic':
                        self.main_screen_window.deiconify()
                        print("✅ Main screen restored from minimized state")
                except Exception:
                    pass
                # Ensure window is raised to top
                self.main_screen_window.lift()
                self.main_screen_window.focus_force()
        except Exception as e:
            print(f"⚠️ Error in on_main_screen_focus_in: {e}")
    
    def on_main_screen_map(self, event=None):
        """Handle main screen window being mapped (shown) - ensure it stays visible"""
        try:
            if self.main_screen_window and self.main_screen_window.winfo_exists():
                # Ensure window stays on top and visible
                self.main_screen_window.lift()
                self.main_screen_window.focus_force()
        except Exception as e:
            print(f"⚠️ Error in on_main_screen_map: {e}")
    
    def close_main_screen(self):
        """Close the main screen window"""
        if self.main_screen_window:
            self.main_screen_window.destroy()
            self.main_screen_window = None
    
    def update_main_screen_person(self, person_data):
        """Update main screen with new person data"""
        if self.main_screen_window and self.main_screen_window.winfo_exists():
            self.display_person_info(person_data)
            # DO NOT add to recent entries here - only log after complete uniform
            # Entry logging should only happen in record_complete_uniform_entry()
            # CRITICAL: Don't schedule clear timer if entry was just saved (timer already scheduled in clear_main_screen_after_gate_action)
            # Check if entry was recently saved by checking if detection_active is False and uniform_detection_complete is True
            entry_was_saved = (
                hasattr(self, 'detection_active') and not self.detection_active and
                hasattr(self, 'uniform_detection_complete') and self.uniform_detection_complete
            )
            if not entry_was_saved:
                # Schedule clearing main screen after 15 seconds (only if entry was NOT just saved)
                self._schedule_main_screen_clear()
            else:
                print(f"⏸️ Skipping _schedule_main_screen_clear - entry was just saved, timer already scheduled")
    
    def update_main_screen_with_person_info(self, person_id, person_name, person_type):
        """Update main screen when person taps their ID"""
        try:
            if not self.main_screen_window or not self.main_screen_window.winfo_exists():
                return
            
            # CRITICAL: Don't overwrite ENTRY status if entry was just saved
            # Check if entry was recently saved by checking if detection_active is False and uniform_detection_complete is True
            if person_type.lower() == 'student':
                if hasattr(self, 'detection_active') and not self.detection_active:
                    if hasattr(self, 'uniform_detection_complete') and self.uniform_detection_complete:
                        # Entry was just saved - don't overwrite with DETECTING
                        print(f"⏸️ Skipping update_main_screen_with_person_info - entry was just saved, keep ENTRY status visible")
                        return  # Exit early - don't overwrite entry status
            
            # In Event Mode, ALLOW showing student info (but no "DETECTING" status)
            # Event Mode just bypasses detection, but we still show student info
            if getattr(self, 'event_mode_active', False):
                print(f"✅ Event Mode active - Showing student info (no DETECTING status)")
                # Continue to show student info in Event Mode (don't return early)
                # But we'll show "ENTRY" status instead of "DETECTING"
            
            # Create person data
            # For students, don't set action to ENTRY - set to DETECTING until uniform is complete
            # In Event Mode, show ENTRY status instead of DETECTING
            if person_type.lower() == 'student':
                if getattr(self, 'event_mode_active', False):
                    action_val = 'ENTRY'
                    status_val = 'TIME-IN (EVENT MODE)'
                    # In Event Mode, detection is NOT active
                    self.detection_active = False
                    self.uniform_detection_complete = False
                else:
                    action_val = 'DETECTING'
                    status_val = 'DETECTING'
                    # Set detection_active before updating screen
                    self.detection_active = True
                    self.uniform_detection_complete = False
            else:
                action_val = 'ENTRY'  # Non-students can have immediate entry
                status_val = 'TIME-IN'
            
            person_data = {
                'id': person_id,
                'name': person_name,
                'type': person_type,
                'action': action_val,  # DETECTING for students, ENTRY for others
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'status': status_val,  # DETECTING for students, TIME-IN for others
                'confidence': 0.0,
                'class_name': 'initializing'
            }
            
            # Add additional info based on person type
            if person_type.lower() == 'student':
                # Get student info from credentials
                student_info = self.get_student_info(person_id)
                if student_info:
                    person_data.update(student_info)
            
            elif person_type.lower() == 'visitor':
                # Add visitor-specific info
                person_data['company'] = 'N/A'
                person_data['purpose'] = 'General Visit'
            
            # Update main screen immediately
            self.display_person_info(person_data)
            
            # Schedule clearing main screen after 15 seconds
            self._schedule_main_screen_clear()
            
            print(f"SUCCESS: Main screen updated with {person_type}: {person_name} ({person_id})")
            
        except Exception as e:
            print(f"ERROR: Error updating main screen with person info: {e}")
    
    def update_main_screen_with_exit(self, person_id, person_type):
        """Update main screen when person exits"""
        try:
            if not self.main_screen_window or not self.main_screen_window.winfo_exists():
                return
            
            # Get person name
            person_name = self.get_person_name(person_id, person_type)
            
            # Create person data for exit
            person_data = {
                'id': person_id,
                'name': person_name,
                'type': person_type,
                'action': 'EXIT',
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'status': 'EXITED',
                'confidence': 1.0,
                'class_name': 'exit'
            }
            
            # Add additional info based on person type
            if person_type.lower() == 'student':
                # Get student info from credentials
                student_info = self.get_student_info(person_id)
                if student_info:
                    person_data.update(student_info)
            
            elif person_type.lower() == 'visitor':
                # Add visitor-specific info
                person_data['company'] = 'N/A'
                person_data['purpose'] = 'General Visit'
            
            # Update main screen with exit information
            self.display_person_info(person_data)
            
            # Schedule clearing main screen after 15 seconds
            self._schedule_main_screen_clear()
            
            print(f"SUCCESS: Main screen updated with {person_type} exit: {person_name} ({person_id})")
            
        except Exception as e:
            print(f"ERROR: Error updating main screen with exit info: {e}")
    
    def test_main_screen_display(self):
        """Test function to display sample data on main screen"""
        try:
            # Sample person data for testing
            test_person_data = {
                'id': '02000289900',
                'name': 'John Doe',
                'type': 'student',
                'action': 'ENTRY',
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'course': 'ICT',
                'gender': 'Male',
                'status': 'COMPLIANT',
                'confidence': 0.55,
                'class_name': 'uniform_detected'
            }
            
            # Update main screen with test data
            self.update_main_screen_person(test_person_data)
            
            print("SUCCESS: Test data displayed on main screen")
            
        except Exception as e:
            print(f"ERROR: Error testing main screen: {e}")
    
    def debug_firebase_student_ui(self):
        """UI function to debug Firebase student data"""
        try:
            # Get student ID from user input
            student_id = tk.simpledialog.askstring(
                "Debug Firebase Student",
                "Enter Student ID to check in Firebase:",
                initialvalue="02000289900"
            )
            
            if student_id:
                # Call debug function
                result = self.debug_firebase_student(student_id)
                
                # Show result in message box
                if result:
                    # Prefer showing 'Senior High School' label for SHS courses
                    course_val = str(result.get('course', 'Not found'))
                    if course_val.upper().startswith('SHS'):
                        shs_strand = course_val[3:].strip()
                        if shs_strand:
                            course_line = f"Senior High School: {shs_strand}\n"
                        else:
                            course_line = "Senior High School\n"
                    else:
                        course_line = f"Course: {course_val}\n"

                    message = f"Student ID: {student_id}\n\n"
                    message += f"Name: {result.get('name', 'Not found')}\n"
                    message += course_line
                    message += f"Gender: {result.get('gender', 'Not found')}\n\n"
                    message += f"Full Data: {result}"
                    messagebox.showinfo("Firebase Student Data", message)
                else:
                    messagebox.showerror("Student Not Found", 
                                       f"Student ID '{student_id}' not found in Firebase students collection.")
            
        except Exception as e:
            messagebox.showerror("Debug Error", f"Error checking Firebase student: {e}")
    
    def logout_guard(self):
        """Logout the guard and close camera"""
        try:
            # Save guard logout to Firebase before clearing session data
            if hasattr(self, 'current_guard_id') and self.current_guard_id:
                self.save_guard_logout_to_firebase(self.current_guard_id)
            
            # Stop any active detection and close camera
            if self.detection_active:
                self.stop_detection()
            
            # Reset camera to standby
            self.reset_camera_to_standby()
            
            # Close main screen
            self.close_main_screen()
            
            # Clear session data
            self.current_guard_id = None
            self.session_id = None
            self.login_status_var.set("Not Logged In")
            
            # Re-enable login button
            self.manual_login_btn.config(state=tk.NORMAL)
            
            # Switch back to login tab
            self.notebook.select(self.login_tab)
            self.notebook.hide(self.dashboard_tab)
            
            # Update guard name display to show no guard logged in
            if hasattr(self, 'update_guard_name_display'):
                self.update_guard_name_display(None)
            
            print("🔓 Guard logged out - Camera closed")
            messagebox.showinfo("Logout", "Guard logged out successfully. Camera is now closed.")
            
        except Exception as e:
            print(f"ERROR: Error during logout: {e}")
            messagebox.showerror("Error", f"Logout failed: {e}")
    
    def create_dashboard_tab(self):
        """Create the dashboard tab"""
        try:
            print("🔍 Creating dashboard tab...")
            self.dashboard_tab = tk.Frame(self.notebook, bg='#ffffff')
            self.notebook.add(self.dashboard_tab, text="📊 AI-niform Dashboard")
            print("SUCCESS: Dashboard tab added to notebook")
            
            # Create dashboard content
            print("🔍 Creating dashboard content...")
            self.create_dashboard_content(self.dashboard_tab)
            print("SUCCESS: Dashboard content created")
            
        except Exception as e:
            print(f"ERROR: Error creating dashboard tab: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    def create_dashboard_content(self, parent):
        """Create dashboard content"""
        # Main container
        main_container = tk.Frame(parent, bg='#ffffff')
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Header
        self.create_header(main_container)
        
        # Content area
        content_frame = tk.Frame(main_container, bg='#ffffff')
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(20, 0))
        
        # Left column - Person ID Entry and Requirement Parts
        left_column = tk.Frame(content_frame, bg='#ffffff', width=520)
        left_column.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))
        left_column.pack_propagate(False)

        # Put person entry and gate controls on the left
        # UNIFORM REQUIREMENTS will be created inside Person ID Entry section
        self.create_person_entry_section(left_column)
        
        # Guard on duty display - at the bottom of left column
        guard_duty_frame = tk.Frame(left_column, bg='#ffffff')
        guard_duty_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=15, pady=(15, 15))
        
        # Separator line
        separator_line = tk.Frame(guard_duty_frame, height=1, bg='#e5e7eb')
        separator_line.pack(fill=tk.X, pady=(0, 10))
        
        # Guard on duty label
        guard_duty_title = tk.Label(
            guard_duty_frame,
            text="Guard on Duty:",
            font=('Arial', 10, 'bold'),
            fg='#6b7280',
            bg='#ffffff'
        )
        guard_duty_title.pack(anchor=tk.W)
        
        # Guard name display (will be updated when guard logs in)
        self.guard_name_label = tk.Label(
            guard_duty_frame,
            text="No guard logged in",
            font=('Arial', 11, 'bold'),
            fg='#1e3a8a',
            bg='#ffffff'
        )
        self.guard_name_label.pack(anchor=tk.W, pady=(5, 0))
        
        # Update guard name display if guard is already logged in
        if hasattr(self, 'current_guard_id') and self.current_guard_id:
            self.update_guard_name_display(self.current_guard_id)

        # Right column - Camera Feed only
        right_column = tk.Frame(content_frame, bg='#ffffff')
        right_column.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(12, 0))

        # Camera feed container - fills the right column
        camera_container = tk.Frame(right_column, bg='#ffffff')
        camera_container.pack(fill=tk.BOTH, expand=True, padx=(0, 0), pady=(0, 0))
        camera_container.pack_propagate(False)

        # Create camera feed section
        self.create_camera_feed_section(camera_container)
        
        # Activity Log - full width at bottom, below both columns
        logs_container = tk.Frame(content_frame, bg='#ffffff', height=250)
        logs_container.pack(side=tk.BOTTOM, fill=tk.X, expand=False, padx=(0, 0), pady=(12, 0))
        logs_container.pack_propagate(False)
        
        # Create activity logs section
        self.create_activity_logs_section(logs_container)
    
    def create_person_entry_section(self, parent):
        """Create person ID entry section"""
        entry_frame = tk.LabelFrame(
            parent,
            text="👥 Person ID Entry",
            font=('Arial', 12, 'bold'),
            fg='#1e3a8a',
            bg='#ffffff',
            relief='groove',
            bd=2
        )
        # Store entry_frame as instance variable so requirements section can pack before it
        self.entry_frame = entry_frame
        entry_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Create a container frame for content that can expand (everything except Gate Control)
        self.entry_content_frame = tk.Frame(entry_frame, bg='#ffffff')
        self.entry_content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Special action buttons - inside content frame
        type_frame = tk.Frame(self.entry_content_frame, bg='#ffffff')
        type_frame.pack(fill=tk.X, padx=15, pady=(15, 12))
        
        # Special buttons row
        special_buttons_frame = tk.Frame(type_frame, bg='#ffffff')
        special_buttons_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Manual visitor entry button - slightly bigger size
        self.manual_visitor_btn = tk.Button(
            special_buttons_frame,
            text="📝 Visitor",
            command=self.handle_manual_visitor,
            font=('Arial', 11, 'bold'),  # Slightly bigger font
            bg='#10b981',
            fg='white',
            relief='raised',
            bd=2,
            padx=12,  # Slightly bigger padding
            pady=7,  # Slightly bigger padding
            cursor='hand2',
            activebackground='#059669',
            activeforeground='white'
        )
        self.manual_visitor_btn.pack(side=tk.LEFT, padx=(0, 8), fill=tk.X, expand=True)
        
        # Event Mode toggle button - resized
        self.event_mode_btn = tk.Button(
            special_buttons_frame,
            text="Event Mode:\nOFF",
            font=('Arial', 10, 'bold'),  # Resized font
            bg='#9ca3af',  # Gray when OFF
            fg='white',
            relief='raised',
            bd=2,
            padx=4,  # Resized padding
            pady=2,  # Resized padding
            cursor='hand2',
            activebackground='#6b7280',
            activeforeground='white',
            command=self.toggle_event_mode
        )
        self.event_mode_btn.pack(side=tk.LEFT, padx=(8, 0), fill=tk.X, expand=True)
        
        # Return RFID button - inside content frame
        return_rfid_frame = tk.Frame(self.entry_content_frame, bg='#ffffff')
        return_rfid_frame.pack(fill=tk.X, padx=15, pady=(15, 15))
        
        # Add a globally accessible Return Assigned RFID button for quick access
        self.return_assigned_rfid_btn = tk.Button(
            return_rfid_frame,
            text="RETURN RFID",
            command=self.handle_rfid_history_tab,
            font=('Arial', 12, 'bold'),
            bg='#f97316',
            fg='white',
            relief='raised',
            bd=2,
            padx=12,
            pady=8,
            cursor='hand2',
            activebackground='#ea580c',
            activeforeground='white'
        )
        self.return_assigned_rfid_btn.pack(fill=tk.X, pady=(0, 15))
        
        # UNIFORM REQUIREMENTS box - always visible container inside Person ID Entry
        # The box stays visible, but requirement parts only appear when student ID is tapped
        self.requirements_box_frame = tk.LabelFrame(
            self.entry_content_frame,
            text="UNIFORM REQUIREMENTS",
            font=('Arial', 12, 'bold'),
            fg='#1f2937',
            bg='#ffffff',
            relief='groove',
            bd=2
        )
        # Pack with minimal height when empty - will expand when requirements are shown
        # Positioned above Gate Control section
        # Pack the box - it will be visible even when empty
        self.requirements_box_frame.pack(fill=tk.X, padx=15, pady=(0, 15))
        
        # Add a larger spacer to ensure box is visible and has good size when empty
        # This prevents the box from collapsing to zero height and makes it more prominent
        self.requirements_spacer = tk.Frame(self.requirements_box_frame, bg='#ffffff', height=100)
        self.requirements_spacer.pack(fill=tk.BOTH, expand=True)
        
        # Create requirements section inside the box (initially hidden)
        self.create_requirement_parts_section(self.requirements_box_frame)
        
        # Gate Control section - fixed at bottom of Person ID Entry (doesn't move when requirements appear)
        gate_control_frame = tk.Frame(entry_frame, bg='#ffffff')
        self.gate_control_frame = gate_control_frame  # Store reference
        # Pack at bottom so it stays fixed - requirements will appear above it in entry_content_frame
        gate_control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=15, pady=(0, 15))
        
        # Gate Control title
        gate_title = tk.Label(
            gate_control_frame,
            text="🚪 Gate Control",
            font=('Arial', 12, 'bold'),
            fg='#1f2937',
            bg='#ffffff'
        )
        gate_title.pack(anchor=tk.W, pady=(0, 10))
        
        # Gate status display
        status_frame = tk.Frame(gate_control_frame, bg='#ffffff')
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.gate_status_label = tk.Label(
            status_frame,
            text="🔒 Gate: Locked",
            font=('Arial', 11, 'bold'),
            fg='#dc2626',
            bg='#ffffff'
        )
        self.gate_status_label.pack(side=tk.LEFT)
        
        # Arduino connection status
        self.arduino_status_label = tk.Label(
            status_frame,
            text="🔌 Arduino: Disconnected",
            font=('Arial', 10),
            fg='#6b7280',
            bg='#ffffff'
        )
        self.arduino_status_label.pack(side=tk.RIGHT)
        
        # Update Arduino status if already connected
        if hasattr(self, 'arduino_connected') and self.arduino_connected:
            self.update_arduino_connection_status(True)
        
        # Control buttons frame - 3 buttons in one row
        buttons_frame = tk.Frame(gate_control_frame, bg='#ffffff')
        buttons_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Approve button - DISABLED by default, only enabled when incomplete uniform is detected
        # Slightly bigger size
        self.approve_button = tk.Button(
            buttons_frame,
            text="ACCESS GRANTED",
            font=('Arial', 11, 'bold'),  # Slightly bigger font
            bg='#10b981',
            fg='white',
            relief='raised',
            bd=2,  # Thicker border
            padx=10,  # Slightly bigger padding
            pady=6,  # Slightly bigger padding
            cursor='hand2',
            activebackground='#059669',
            activeforeground='white',
            command=self.handle_interface_approve,
            state=tk.DISABLED  # Disabled by default - only enabled when incomplete uniform detected
        )
        self.approve_button.pack(side=tk.LEFT, padx=(0, 2), fill=tk.X, expand=True)
        
        # Store incomplete student info for APPROVE button (similar to approve_complete_uniform)
        self.incomplete_student_info_for_approve = None
        
        # Cancel button - DISABLED by default, only enabled after incomplete uniform or DENIED clicked
        # Slightly bigger size
        self.cancel_button = tk.Button(
            buttons_frame,
            text="CANCEL",
            font=('Arial', 11, 'bold'),  # Slightly bigger font
            bg='#6b7280',
            fg='white',
            disabledforeground='#d1d5db',  # Light gray text when disabled - ensures text is visible
            relief='raised',
            bd=2,
            padx=10,  # Slightly bigger padding
            pady=6,  # Slightly bigger padding
            cursor='hand2',
            activebackground='#4b5563',
            activeforeground='white',
            command=self.handle_interface_cancel,
            state=tk.DISABLED  # Disabled by default - only enabled after incomplete uniform or DENIED clicked
        )
        self.cancel_button.pack(side=tk.LEFT, padx=(2, 2), fill=tk.X, expand=True)
        
        # Deny button - slightly bigger size
        self.deny_button = tk.Button(
            buttons_frame,
            text="DENIED",
            font=('Arial', 11, 'bold'),  # Slightly bigger font
            bg='#dc2626',
            fg='white',
            relief='raised',
            bd=2,  # Thicker border
            padx=10,  # Slightly bigger padding
            pady=6,  # Slightly bigger padding
            cursor='hand2',
            activebackground='#b91c1c',
            activeforeground='white',
            command=self.handle_interface_deny
        )
        self.deny_button.pack(side=tk.LEFT, padx=(2, 0), fill=tk.X, expand=True)
        
        # Instructions
        instructions_label = tk.Label(
            gate_control_frame,
            text="💡 Use buttons to control gate. Event Mode bypasses uniform detection.",
            font=('Arial', 9, 'italic'),
            fg='#6b7280',
            bg='#ffffff',
            wraplength=400
        )
        instructions_label.pack(pady=(5, 0))
        
        # Create hidden RFID Entry field for global RFID capture
        # This field is invisible but always ready to receive RFID input
        self.hidden_rfid_entry = tk.Entry(
            self.entry_content_frame,
            textvariable=self.person_id_var,
            width=1,
            font=('Arial', 1),
            bg='#ffffff',
            fg='#ffffff',
            relief='flat',
            bd=0,
            highlightthickness=0
        )
        # Position it off-screen but keep it in the widget tree
        self.hidden_rfid_entry.place(x=-100, y=-100)
        
        # Bind Return key to auto-process RFID input
        self.hidden_rfid_entry.bind('<Return>', lambda e: self.process_rfid_input())
        # Keep it always ready for RFID input
        self.hidden_rfid_entry.config(takefocus=True)
    
    def create_requirement_parts_section(self, parent):
        """Create Requirement Parts section inside the requirements box"""
        # Requirement Parts container (inside the box, not a separate LabelFrame)
        self.requirements_frame = tk.Frame(
            parent,
            bg='#ffffff'
        )
        # Don't pack yet - will be packed when shown
        # Initial pack is done in show_requirements_section
        
        # Student info label - Name and Course/Senior High School
        self.requirements_student_info_label = tk.Label(
            self.requirements_frame,
            text="",
            font=('Arial', 10, 'bold'),
            fg='#1f2937',
            bg='#ffffff',
            justify=tk.LEFT
        )
        self.requirements_student_info_label.pack(anchor='w', padx=15, pady=(10, 5))
        
        # Status label - IN/COMPLETE UNIFORM
        self.requirements_status_label = tk.Label(
            self.requirements_frame,
            text="IN/COMPLETE UNIFORM",
            font=('Arial', 11, 'bold'),
            fg='#dc2626',
            bg='#ffffff',
            justify=tk.LEFT
        )
        self.requirements_status_label.pack(anchor='w', padx=15, pady=(0, 5))
        
        # Requirements container frame for checkboxes
        self.requirements_container = tk.Frame(self.requirements_frame, bg='#ffffff')
        self.requirements_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 10))
        
        # Dictionary to store checkbox variables and widgets
        self.requirement_checkboxes = {}  # key: requirement_name (lowercase) -> {'var': tk.BooleanVar, 'widget': Checkbutton}
        self.requirement_checkbox_widgets = {}  # key: requirement_name (lowercase) -> Checkbutton widget
        
        # Hide requirements section by default - only show when student ID is tapped
        self.requirements_frame.pack_forget()
        
        # Populate initial requirements list (will be shown when student ID is tapped)
        try:
            self._render_uniform_requirements()
        except Exception:
            pass
    
    def setup_rfid_focus_management(self):
        """Set up periodic focus management for hidden RFID Entry field"""
        def periodic_refocus():
            try:
                # Only refocus if no Entry field is currently focused
                # This allows normal UI interactions but ensures RFID goes to hidden field
                focused_widget = self.root.focus_get()
                if focused_widget is None or not isinstance(focused_widget, tk.Entry):
                    if hasattr(self, 'hidden_rfid_entry') and self.hidden_rfid_entry:
                        self.hidden_rfid_entry.focus_set()
            except Exception:
                pass
            # Schedule next check (every 2 seconds)
            self.root.after(2000, periodic_refocus)
        
        # Start periodic focus management
        self.root.after(1000, periodic_refocus)
    
    def set_rfid_entry_focus(self):
        """Set focus on hidden RFID Entry field to capture RFID input"""
        try:
            if hasattr(self, 'hidden_rfid_entry') and self.hidden_rfid_entry:
                self.hidden_rfid_entry.focus_set()
                print("✅ Focus set on hidden RFID Entry field")
        except Exception as e:
            print(f"⚠️ Could not set focus on hidden RFID Entry: {e}")
    
    def is_empty_rfid_available(self, rfid):
        """Check if RFID is an available empty RFID card"""
        try:
            # First check if RFID is already assigned to a student (local memory)
            if rfid in getattr(self, 'student_rfid_assignments', {}):
                return False
            
            # Check if RFID is already assigned to a visitor (local memory)
            if hasattr(self, 'visitor_rfid_registry') and rfid in self.visitor_rfid_registry:
                return False
            if hasattr(self, 'visitor_rfid_assignments') and rfid in getattr(self, 'visitor_rfid_assignments', {}):
                return False
            
            # Note: student_forgot_rfids list is no longer used as a separate pool
            # Empty RFIDs are shared between visitor and student forgot ID
            # All empty RFIDs come from the "Empty RFID" collection in Firebase
            
            # Check Firebase Empty RFID collection
            if self.firebase_initialized and self.db:
                try:
                    empty_rfid_ref = self.db.collection('Empty RFID').document(rfid)
                    rfid_doc = empty_rfid_ref.get()
                    if rfid_doc.exists:
                        rfid_data = rfid_doc.to_dict() or {}
                        # Check if available and not assigned
                        if rfid_data.get('available', True) and not rfid_data.get('assigned_to'):
                            # Check Firebase student_rfid_assignments collection
                            try:
                                assignments_ref = self.db.collection('student_rfid_assignments')
                                assignments_query = assignments_ref.where('rfid', '==', rfid).where('status', '==', 'active').limit(1)
                                assignments = assignments_query.get()
                                if assignments:
                                    return False  # RFID is assigned to a student in Firebase
                            except Exception as e:
                                print(f"⚠️ Error checking student_rfid_assignments for {rfid}: {e}")
                            
                            # Check Firebase visitors collection for active assignments
                            try:
                                visitors_ref = self.db.collection('visitors')
                                visitors_query = visitors_ref.where('rfid', '==', rfid).where('status', 'in', ['registered', 'active']).limit(1)
                                visitors = visitors_query.get()
                                if visitors:
                                    return False  # RFID is assigned to a visitor in Firebase
                            except Exception as e:
                                print(f"⚠️ Error checking visitors collection for {rfid}: {e}")
                            
                            return True
                except Exception as e:
                    print(f"⚠️ Error checking Firebase for RFID {rfid}: {e}")
            
            return False
        except Exception as e:
            print(f"⚠️ Error in is_empty_rfid_available: {e}")
            return False
    
    def check_student_already_has_rfid(self, student_id):
        """Check if student already has an RFID assigned
        
        Returns:
            tuple: (has_rfid: bool, existing_rfid: str or None)
        """
        try:
            student_id = str(student_id).strip()
            
            # Check local memory first
            if hasattr(self, 'student_rfid_assignments') and isinstance(self.student_rfid_assignments, dict):
                for rfid, assignment_info in self.student_rfid_assignments.items():
                    assignment_student_id = str(assignment_info.get('student_id', '')).strip()
                    if assignment_student_id == student_id:
                        # Check if assignment is still active (not expired)
                        expiry_time_str = assignment_info.get('expiry_time', '')
                        if expiry_time_str:
                            try:
                                from datetime import datetime
                                expiry_time = datetime.strptime(expiry_time_str, "%Y-%m-%d %H:%M:%S")
                                if datetime.now() < expiry_time:
                                    return (True, rfid)
                            except Exception:
                                # If expiry parsing fails, assume active
                                return (True, rfid)
                        else:
                            # No expiry time, assume active
                            return (True, rfid)
            
            # Check Firebase student_rfid_assignments collection
            if self.firebase_initialized and self.db:
                try:
                    assignments_ref = self.db.collection('student_rfid_assignments')
                    assignments_query = assignments_ref.where('student_id', '==', student_id).where('status', '==', 'active').limit(1)
                    assignments = assignments_query.get()
                    
                    if assignments:
                        for doc in assignments:
                            assignment_data = doc.to_dict()
                            rfid = assignment_data.get('rfid', '')
                            # Check if not expired
                            expiry_time_str = assignment_data.get('expiry_time', '')
                            if expiry_time_str:
                                try:
                                    from datetime import datetime
                                    expiry_time = datetime.strptime(expiry_time_str, "%Y-%m-%d %H:%M:%S")
                                    if datetime.now() < expiry_time:
                                        return (True, rfid)
                                except Exception:
                                    # If expiry parsing fails, assume active
                                    return (True, rfid)
                            else:
                                # No expiry time, assume active
                                return (True, rfid)
                except Exception as e:
                    print(f"⚠️ Error checking Firebase student_rfid_assignments for student {student_id}: {e}")
            
            return (False, None)
        except Exception as e:
            print(f"⚠️ Error in check_student_already_has_rfid: {e}")
            return (False, None)
    
    def process_rfid_input(self):
        """Process RFID input from hidden field - automatically routes to student detection or assignment"""
        try:
            rfid = self.person_id_var.get().strip()
            if not rfid:
                return
            
            print(f"🔍 RFID detected from hidden field: {rfid}")
            
            # Clear the field immediately for next RFID
            self.person_id_var.set("")
            
            # Check if we're in RFID return mode (highest priority)
            if hasattr(self, 'visitor_rfid_return_mode') and self.visitor_rfid_return_mode:
                print(f"🔍 RFID return mode is ACTIVE - auto-filling RFID in return dialog: {rfid}")
                # Auto-fill the RFID in the return dialog if it's open
                if hasattr(self, 'return_rfid_entry_var') and self.return_rfid_entry_var:
                    self.return_rfid_entry_var.set(rfid)
                    # Focus the entry field
                    if hasattr(self, 'return_rfid_entry') and self.return_rfid_entry:
                        self.return_rfid_entry.focus_set()
                        self.return_rfid_entry.select_range(0, tk.END)
                    print(f"✅ RFID {rfid} auto-filled in return dialog")
                else:
                    # Dialog not open, process directly
                    self.process_rfid_return(rfid)
                return
            
            # Check if we're in visitor RFID assignment mode
            if hasattr(self, 'visitor_rfid_assignment_active') and self.visitor_rfid_assignment_active:
                print(f"🔍 Visitor RFID assignment mode is ACTIVE - processing RFID: {rfid}")
                # Route RFID to the visible entry field in Visitor RFID section
                if hasattr(self, 'visitor_rfid_entry_var'):
                    print(f"🔍 Routing RFID to visitor entry field: {rfid}")
                    self.visitor_rfid_entry_var.set(rfid)
                    # Automatically process it
                    self.root.after(100, self.process_visitor_rfid_tap_entry)
                else:
                    # Fallback: process directly if entry field doesn't exist
                    print(f"⚠️ visitor_rfid_entry_var not found - using fallback processing")
                    is_available = self.is_empty_rfid_available(rfid)
                    print(f"🔍 Empty RFID availability check result: {is_available}")
                    if is_available:
                        print(f"✅ Empty RFID detected in visitor assignment mode: {rfid}")
                        # Auto-register visitor with this RFID
                        self.register_visitor(rfid=rfid)
                        self.visitor_rfid_assignment_active = False
                        self.reset_visitor_rfid_tap_button()
                    else:
                        print(f"⚠️ Tapped RFID {rfid} is not an available empty RFID")
                        messagebox.showerror(
                            "RFID Not Available",
                            f"RFID {rfid} is already assigned to a student or visitor.\n\nPlease return it first or choose an available RFID."
                        )
                        self.reset_visitor_rfid_tap_button()
                # Refocus the hidden field
                self.root.after(200, self.set_rfid_entry_focus)
                return
            
            # Check if we're in student RFID assignment mode
            if hasattr(self, 'rfid_assignment_active') and self.rfid_assignment_active:
                print(f"🔍 Student RFID assignment mode is ACTIVE - processing RFID: {rfid}")
                # Route RFID to the entry variable and process it (for popup dialog)
                if hasattr(self, 'rfid_tap_entry_var'):
                    print(f"🔍 Routing RFID to student assignment entry field: {rfid}")
                    self.rfid_tap_entry_var.set(rfid)
                    # Automatically process it
                    self.root.after(100, self.process_rfid_tap_entry)
                else:
                    # Fallback: process directly if entry field doesn't exist
                    print(f"⚠️ rfid_tap_entry_var not found - using fallback processing")
                    # Check if RFID is available first
                    is_available = self.is_empty_rfid_available(rfid)
                    print(f"🔍 Empty RFID availability check result: {is_available}")
                    if not is_available:
                        print(f"⚠️ Tapped RFID {rfid} is not an available empty RFID")
                        messagebox.showerror(
                            "RFID Not Available",
                            f"RFID {rfid} is already assigned to a student or visitor.\n\nPlease return it first or choose an available RFID."
                        )
                        self.reset_rfid_tap_button()
                        self.root.after(200, self.set_rfid_entry_focus)
                        return
                    
                    # Check if student already has RFID
                    if hasattr(self, 'pending_assignment_student') and self.pending_assignment_student:
                        student_id = str(self.pending_assignment_student.get('student_id', '')).strip()
                    else:
                        student_id = self.student_id_entry.get().strip()
                    
                    if student_id:
                        has_rfid, existing_rfid = self.check_student_already_has_rfid(student_id)
                        if has_rfid:
                            print(f"⚠️ Student {student_id} already has RFID {existing_rfid}")
                            messagebox.showerror(
                                "Student Already Has RFID",
                                f"This student is already assigned to RFID {existing_rfid}.\n\nPlease return the existing RFID before assigning a new one."
                            )
                            self.reset_rfid_tap_button()
                            self.root.after(200, self.set_rfid_entry_focus)
                            return
                    
                    # All checks passed - proceed with assignment
                    print(f"✅ Empty RFID detected in assignment mode: {rfid}")
                    self.assign_rfid_to_student(rfid=rfid)
                    self.rfid_assignment_active = False
                    self.reset_rfid_tap_button()
                # Refocus the hidden field
                self.root.after(200, self.set_rfid_entry_focus)
                return
            
            # Normal mode - check for entry/exit (NOT assignment)
            # First check if RFID is assigned to a visitor for entry/exit
            visitor_info = None
            if hasattr(self, 'visitor_rfid_assignments') and rfid in getattr(self, 'visitor_rfid_assignments', {}):
                visitor_info = self.visitor_rfid_assignments[rfid]
            elif rfid in getattr(self, 'visitor_rfid_registry', {}):
                visitor_info = self.visitor_rfid_registry[rfid]
            
            # Also check Firebase visitors collection if not found locally
            if not visitor_info and self.firebase_initialized and self.db:
                try:
                    visitors_ref = self.db.collection('visitors')
                    visitors_query = visitors_ref.where('rfid', '==', rfid).limit(1)
                    visitors = visitors_query.get()
                    if visitors:
                        for doc in visitors:
                            visitor_data = doc.to_dict()
                            status = visitor_data.get('status', '')
                            time_out = visitor_data.get('time_out')
                            
                            # Skip if visitor has exited (status is 'exited' and has time_out)
                            if status == 'exited' and time_out:
                                print(f"🔍 DEBUG: RFID {rfid} visitor has exited (status: {status}, time_out: {time_out}), skipping")
                                continue
                            
                            # Only consider active visitors (registered or active status, or has time_in without time_out)
                            if status not in ('registered', 'active') and not (visitor_data.get('time_in') and not time_out):
                                print(f"🔍 DEBUG: RFID {rfid} visitor status is '{status}' (not active), skipping")
                                continue
                            
                            # Check if assignment is still valid (not expired)
                            expiry_time_str = visitor_data.get('expiry_time', '')
                            if expiry_time_str:
                                try:
                                    expiry_time = datetime.datetime.strptime(expiry_time_str, "%Y-%m-%d %H:%M:%S")
                                    if datetime.datetime.now() >= expiry_time:
                                        print(f"🔍 DEBUG: RFID {rfid} visitor assignment expired (expiry: {expiry_time_str}), skipping")
                                        continue
                                except Exception:
                                    # If expiry parsing fails, assume valid
                                    pass
                            
                            # Valid active visitor found
                            visitor_info = visitor_data
                            break
                except Exception as e:
                    print(f"⚠️ Error checking Firebase visitors for RFID {rfid}: {e}")
            
            # If RFID is assigned to a visitor, handle as visitor entry/exit
            if visitor_info:
                print(f"🔍 RFID {rfid} is assigned to visitor: {visitor_info.get('name', 'Unknown')} - processing entry/exit")
                self.handle_visitor_rfid_tap(rfid)
                # Refocus the hidden field after processing to be ready for next RFID
                self.root.after(100, self.set_rfid_entry_focus)
                return
            
            # Check if RFID belongs to a teacher
            teacher_info = None
            if self.firebase_initialized and self.db:
                try:
                    teacher_info = self.get_teacher_info_by_rfid(rfid)
                    if teacher_info:
                        print(f"🔍 RFID {rfid} is assigned to teacher: {teacher_info.get('name', 'Unknown')} - processing entry/exit")
                        self.handle_teacher_rfid_tap(rfid)
                        # Refocus the hidden field after processing to be ready for next RFID
                        self.root.after(100, self.set_rfid_entry_focus)
                        return
                except Exception as e:
                    print(f"⚠️ Error checking teacher for RFID {rfid}: {e}")
            
            # Check temporary student assignments (forgot ID) FIRST - before permanent students
            # This ensures temporary assignments take precedence even if RFID exists in students collection
            assignment_info = None
            student_rfid_assignments = getattr(self, 'student_rfid_assignments', {})
            print(f"🔍 DEBUG: Checking temporary assignments for RFID {rfid}")
            print(f"🔍 DEBUG: student_rfid_assignments keys: {list(student_rfid_assignments.keys())}")
            if rfid in student_rfid_assignments:
                assignment_info = student_rfid_assignments[rfid]
                print(f"✅ DEBUG: Found temporary assignment in local memory: {assignment_info.get('name', 'Unknown')}")
            else:
                print(f"⚠️ DEBUG: RFID {rfid} not found in local student_rfid_assignments")
            
            # Also check Firebase student_rfid_assignments collection if not found locally
            if not assignment_info and self.firebase_initialized and self.db:
                print(f"🔍 DEBUG: Checking Firebase student_rfid_assignments for RFID {rfid}")
                try:
                    assignments_ref = self.db.collection('student_rfid_assignments')
                    assignments_query = assignments_ref.where('rfid', '==', rfid).where('status', '==', 'active').limit(1)
                    assignments = assignments_query.get()
                    if assignments:
                        for doc in assignments:
                            assignment_data = doc.to_dict()
                            # Check if assignment is still valid (not expired)
                            expiry_time_str = assignment_data.get('expiry_time', '')
                            if expiry_time_str:
                                try:
                                    expiry_time = datetime.datetime.strptime(expiry_time_str, "%Y-%m-%d %H:%M:%S")
                                    if datetime.datetime.now() < expiry_time:
                                        assignment_info = assignment_data
                                        break
                                except Exception:
                                    # If expiry parsing fails, assume valid
                                    assignment_info = assignment_data
                                    break
                            else:
                                # No expiry time, assume valid
                                assignment_info = assignment_data
                                break
                except Exception as e:
                    print(f"⚠️ Error checking Firebase student_rfid_assignments for RFID {rfid}: {e}")
            
            # If RFID is assigned to a temporary student, handle entry/exit
            if assignment_info:
                print(f"🔍 RFID {rfid} is assigned to student (forgot ID): {assignment_info.get('name', 'Unknown')} - processing entry/exit")
                self.handle_student_forgot_id_rfid_tap(rfid)
                # Refocus the hidden field after processing to be ready for next RFID
                self.root.after(100, self.set_rfid_entry_focus)
                return
            
            # Check permanent student LAST - after temporary assignments
            # This ensures temporary assignments take precedence even if RFID exists in students collection
            student_info = None
            try:
                student_info = self.get_student_info_by_rfid(rfid)
                if student_info:
                    print(f"🔍 RFID {rfid} is assigned to permanent student: {student_info.get('name', 'Unknown')} - processing entry/exit")
                    self.handle_permanent_student_rfid_tap(rfid)
                    # Refocus the hidden field after processing to be ready for next RFID
                    self.root.after(100, self.set_rfid_entry_focus)
                    return
            except Exception as e:
                print(f"⚠️ Error checking student info for RFID {rfid}: {e}")
            
            # Check if RFID is an empty RFID before showing unknown error
            try:
                if self.is_empty_rfid_available(rfid):
                    print(f"🔍 Empty RFID detected: {rfid}")
                    
                    # Check if there's a pending student assignment (student who forgot ID)
                    if hasattr(self, 'pending_assignment_student') and self.pending_assignment_student:
                        print(f"✅ Pending student found - auto-assigning empty RFID {rfid} to student")
                        self.add_activity_log(f"Empty RFID {rfid} auto-assigned to pending student: {self.pending_assignment_student.get('name', 'Unknown')}")
                        
                        try:
                            # Automatically assign the RFID to the pending student
                            self.assign_rfid_to_student(rfid=rfid)
                            
                            # Verify that assignment was created successfully
                            if rfid not in self.student_rfid_assignments:
                                print(f"⚠️ Assignment failed - RFID {rfid} not found in student_rfid_assignments")
                                messagebox.showerror(
                                    "Assignment Failed",
                                    f"Failed to assign RFID {rfid} to student.\n\nPlease try again or use the RFID Assignment section."
                                )
                                # Refocus the hidden field after processing to be ready for next RFID
                                self.root.after(100, self.set_rfid_entry_focus)
                                return
                            
                            # After assignment, immediately process the entry with violation
                            # This will link the student info and record the forgot_id violation
                            self.handle_student_forgot_id_rfid_tap(rfid)
                            
                            # Clear pending assignment after successful assignment
                            self.pending_assignment_student = None
                            
                            print(f"✅ Successfully assigned empty RFID {rfid} to student and processed entry")
                        except Exception as e:
                            print(f"⚠️ Error auto-assigning RFID to pending student: {e}")
                            import traceback
                            traceback.print_exc()
                            messagebox.showerror(
                                "Assignment Error",
                                f"Failed to assign RFID {rfid} to student.\n\nError: {e}"
                            )
                        
                        # Refocus the hidden field after processing to be ready for next RFID
                        self.root.after(100, self.set_rfid_entry_focus)
                        return
                    else:
                        # No pending student - show helpful message
                        print(f"🔍 Empty RFID detected but no pending student assignment")
                        self.add_activity_log(f"Empty RFID tapped: {rfid}")
                        try:
                            messagebox.showinfo(
                                "Empty RFID Detected", 
                                f"Empty RFID {rfid} detected.\n\nPlease use the RFID Assignment section to assign this RFID to a student."
                            )
                        except Exception:
                            pass
                        # Refocus the hidden field after processing to be ready for next RFID
                        self.root.after(100, self.set_rfid_entry_focus)
                        return
            except Exception as e:
                print(f"⚠️ Error checking if RFID is empty: {e}")
            
            # RFID not found as visitor or student - show unknown error
            print(f"⚠️ Unknown RFID tapped: {rfid}")
            self.add_activity_log(f"Unknown RFID tapped: {rfid}")
            try:
                messagebox.showwarning(
                    "Unknown RFID", 
                    f"RFID {rfid} is not assigned to any student or visitor.\n\nPlease register or assign this RFID first."
                )
            except Exception:
                pass
            
            # Refocus the hidden field after processing to be ready for next RFID
            self.root.after(100, self.set_rfid_entry_focus)
            
        except Exception as e:
            print(f"ERROR: Error processing RFID input: {e}")
            import traceback
            traceback.print_exc()
            # Still refocus even on error
            self.root.after(100, self.set_rfid_entry_focus)
    
    def toggle_event_mode(self):
        """Toggle Event Mode on/off - bypasses uniform detection for all students"""
        try:
            self.event_mode_active = not self.event_mode_active
            
            if self.event_mode_active:
                # Event Mode ON - Stop any running detection immediately
                print("🎉 Event Mode being enabled - stopping all detection")
                
                # CRITICAL: Stop any running detection
                if hasattr(self, 'detection_system') and self.detection_system:
                    try:
                        self.detection_system.stop_detection()
                        print("🛑 Stopped detection system when Event Mode enabled")
                    except Exception as e:
                        print(f"⚠️ Warning: Could not stop detection system: {e}")
                
                # Set detection flags to False
                self.detection_active = False
                self.uniform_detection_complete = False
                
                # Hide requirements section
                try:
                    self.hide_requirements_section()
                    print("✅ Requirements section hidden when Event Mode enabled")
                except Exception as e:
                    print(f"⚠️ Warning: Could not hide requirements section: {e}")
                
                # Ensure camera feed is in guard mode (not detection)
                try:
                    # Stop guard camera if it's in detection mode, restart in guard mode
                    self.stop_guard_camera_feed()
                    time.sleep(0.2)  # Brief pause
                    self.initialize_guard_camera_feed()
                    print("✅ Camera feed reset to guard mode")
                except Exception as e:
                    print(f"⚠️ Warning: Could not reset camera feed: {e}")
                
                # Event Mode ON
                self.event_mode_btn.config(
                    text="Event Mode:\nON",
                    bg='#f59e0b',  # Orange/amber when ON
                    activebackground='#d97706'
                )
                self.add_activity_log("Event Mode enabled - Uniform detection bypassed for all students")
                messagebox.showinfo("Event Mode", "Event Mode is now ON\n\nAll students can enter by RFID tap only.\nUniform detection is bypassed.")
                print("🎉 Event Mode: ON - Uniform detection bypassed")
            else:
                # Event Mode OFF
                self.event_mode_btn.config(
                    text="Event Mode:\nOFF",
                    bg='#9ca3af',  # Gray when OFF
                    activebackground='#6b7280'
                )
                self.add_activity_log("Event Mode disabled - Normal uniform detection resumed")
                messagebox.showinfo("Event Mode", "Event Mode is now OFF\n\nNormal uniform detection is active.")
                print("🔄 Event Mode: OFF - Normal detection resumed")
                
        except Exception as e:
            print(f"ERROR: Error toggling event mode: {e}")
            messagebox.showerror("Error", f"Failed to toggle event mode: {e}")
    
    def handle_forgot_id(self):
        """Handle student who forgot their ID - open student verification tab"""
        try:
            # Create student forgot ID tab
            self.create_student_forgot_tab()
            
            # Switch to student forgot tab
            self.notebook.select(self.student_forgot_tab)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open student verification: {e}")
    
    def create_student_forgot_tab(self):
        """Create student forgot ID verification tab"""
        if self.student_forgot_tab is None:
            self.student_forgot_tab = tk.Frame(self.notebook, bg='#ffffff')
            self.notebook.add(self.student_forgot_tab, text="🔑 Student ID Verification")
            
            # Create student verification form content
            self.create_student_verification_form(self.student_forgot_tab)
    
    def create_student_verification_form(self, parent):
        """Create student ID verification form"""
        # Main container
        main_frame = tk.Frame(parent, bg='#ffffff')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Header with back button
        header_frame = tk.Frame(main_frame, bg='#ffffff')
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Back button at top
        back_btn = tk.Button(
            header_frame,
            text="← Back to Dashboard",
            command=self.back_to_dashboard,
            font=('Arial', 12, 'bold'),
            bg='#6b7280',
            fg='white',
            relief='raised',
            bd=3,
            padx=20,
            pady=8,
            cursor='hand2',
            activebackground='#4b5563',
            activeforeground='white'
        )
        back_btn.pack(anchor=tk.W, pady=(0, 10))
        
        title_label = tk.Label(
            header_frame,
            text="🔑 Student ID Verification",
            font=('Arial', 24, 'bold'),
            fg='#1e3a8a',
            bg='#ffffff'
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            header_frame,
            text="Enter student ID number to verify and link with available RFID",
            font=('Arial', 14),
            fg='#6b7280',
            bg='#ffffff'
        )
        subtitle_label.pack(pady=(5, 0))
        
        # Student ID input section - same width as Student Information section
        id_input_frame = tk.LabelFrame(
            main_frame,
            text="Student ID Verification",
            font=('Arial', 14, 'bold'),
            fg='#1e3a8a',
            bg='#ffffff',
            relief='groove',
            bd=2
        )
        id_input_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Student ID input
        id_input_content = tk.Frame(id_input_frame, bg='#ffffff')
        id_input_content.pack(fill=tk.X, padx=15, pady=15)
        
        id_label = tk.Label(
            id_input_content,
            text="Student ID Number:",
            font=('Arial', 14, 'bold'),
            fg='#374151',
            bg='#ffffff'
        )
        id_label.pack(anchor=tk.W, pady=(0, 10))
        
        # Center container for the input field and button
        input_center_frame = tk.Frame(id_input_content, bg='#ffffff')
        input_center_frame.pack(expand=True)
        
        # Entry widget - shorter width for student ID only (e.g., 02000285773 = 11 chars, so width=18 is good)
        self.student_id_entry = tk.Entry(
            input_center_frame,
            font=('Arial', 16, 'bold'),
            width=18,
            justify=tk.CENTER,
            relief='solid',
            bd=3,
            bg='#f0f9ff',
            fg='#1e3a8a',
            insertbackground='#1e3a8a'
        )
        self.student_id_entry.pack(pady=(0, 15))
        self.student_id_entry.bind('<Return>', lambda e: self.verify_student_id())
        
        # Verify button - same width as input field
        verify_btn = tk.Button(
            input_center_frame,
            text="🔍 Verify Student ID",
            command=self.verify_student_id,
            font=('Arial', 12, 'bold'),
            bg='#3b82f6',
            fg='white',
            relief='raised',
            bd=2,
            padx=20,
            pady=8,
            cursor='hand2',
            activebackground='#2563eb',
            activeforeground='white',
            width=18
        )
        verify_btn.pack(pady=(0, 5))
        
        # Student info display (initially hidden)
        self.student_info_frame = tk.LabelFrame(
            main_frame,
            text="Student Information",
            font=('Arial', 14, 'bold'),
            fg='#1e3a8a',
            bg='#ffffff',
            relief='groove',
            bd=2
        )
        self.student_info_frame.pack(fill=tk.X, pady=(0, 10))
        
        # RFID assignment section (initially hidden)
        self.rfid_assignment_frame = tk.LabelFrame(
            main_frame,
            text="📡 RFID Assignment",
            font=('Arial', 14, 'bold'),
            fg='#1e3a8a',
            bg='#ffffff',
            relief='groove',
            bd=2
        )
        self.rfid_assignment_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Action buttons - always visible at bottom
        buttons_frame = tk.Frame(main_frame, bg='#ffffff')
        buttons_frame.pack(fill=tk.X, pady=(10, 0), side=tk.BOTTOM)
        
        # Assign RFID button (initially hidden)
        self.assign_rfid_btn = tk.Button(
            buttons_frame,
            text="SUCCESS: Assign RFID",
            command=self.assign_rfid_to_student,
            font=('Arial', 11, 'bold'),
            bg='#10b981',
            fg='white',
            relief='raised',
            bd=2,
            padx=20,
            pady=8,
            cursor='hand2',
            activebackground='#059669',
            activeforeground='white'
        )
        self.assign_rfid_btn.pack(side=tk.RIGHT)
        self.assign_rfid_btn.pack_forget()  # Initially hidden
    
    def verify_student_id(self):
        """Verify student ID in Firebase and student credentials"""
        try:
            student_id = self.student_id_entry.get().strip()
            
            if not student_id:
                messagebox.showerror("Error", "Please enter a student ID number.")
                return
            
            # Normalize the entered ID (trim and remove surrounding quotes)
            norm_id = student_id.strip().strip('"').strip("'")

            # Check in Firebase students collection by student-number fields (don't accept fallback defaults)
            student_info = None
            try:
                if self.firebase_initialized and self.db:
                    students_ref = self.db.collection('students')
                    # First, try direct document id lookup (some setups use RFID as doc id)
                    try:
                        doc_ref = students_ref.document(norm_id)
                        doc = doc_ref.get()
                        if doc.exists:
                            data = doc.to_dict()
                            found_doc = doc
                        else:
                            found_doc = None
                    except Exception:
                        found_doc = None

                    # Get data from document ID lookup
                    doc_id_data = None
                    if found_doc is not None:
                        doc_id_data = found_doc.to_dict()
                        print(f"🔍 DEBUG verify_student_id: Found document by document ID, fields: {list(doc_id_data.keys())}")
                    
                    # Also try querying by 'Student Number' field (might be stored under RFID as document ID)
                    field_query_data = None
                    field_candidates = ['Student Number', 'student_number', 'student_id', 'StudentID', 'Student Id', 'student id']
                    for field in field_candidates:
                        try:
                            query = students_ref.where(field, '==', norm_id)
                            docs = query.get()
                            if docs and len(docs) > 0:
                                field_query_data = docs[0].to_dict()
                                print(f"🔍 DEBUG verify_student_id: Found document by field '{field}', fields: {list(field_query_data.keys())}")
                                break
                        except Exception:
                            continue

                    # If not found yet, do a trimmed-value scan as a last resort
                    if field_query_data is None:
                        try:
                            docs = students_ref.get()
                            for d in docs:
                                dd = d.to_dict()
                                # check candidate fields for trimmed match
                                for field in ('Student Number', 'student_number', 'student_id', 'StudentID', 'Student Id', 'student id'):
                                    if field in dd and dd.get(field):
                                        candidate = str(dd.get(field)).strip().strip('"').strip("'")
                                        if candidate == norm_id:
                                            field_query_data = dd
                                            print(f"🔍 DEBUG verify_student_id: Found document by scan, fields: {list(field_query_data.keys())}")
                                            break
                                if field_query_data is not None:
                                    break
                        except Exception:
                            pass
                    
                    # Choose the document with more complete data (prefer one with Gender field if available)
                    data = None
                    if doc_id_data and field_query_data:
                        # Prefer the one with Gender field
                        if 'Gender' in field_query_data or 'gender' in field_query_data:
                            data = field_query_data
                            print(f"🔍 DEBUG verify_student_id: Using field_query_data (has Gender)")
                        elif 'Gender' in doc_id_data or 'gender' in doc_id_data:
                            data = doc_id_data
                            print(f"🔍 DEBUG verify_student_id: Using doc_id_data (has Gender)")
                        else:
                            # Use whichever has more fields
                            data = field_query_data if len(field_query_data) > len(doc_id_data) else doc_id_data
                            print(f"🔍 DEBUG verify_student_id: Using document with more fields")
                    elif field_query_data:
                        data = field_query_data
                    elif doc_id_data:
                        data = doc_id_data

                    # If we have data, build student_info
                    if data:
                        # Debug: Log all available fields in Firebase document
                        print(f"🔍 DEBUG verify_student_id: All fields in Firebase document: {list(data.keys())}")
                        
                        # Check for Senior High School fields first
                        senior_high = None
                        for key in ('Senior High School', 'senior_high_school', 'senior_high', 'Strand', 'strand', 'Senior High', 'SHS'):
                            if key in data and data.get(key):
                                senior_high = str(data.get(key)).strip()
                                if senior_high:  # Only use if not empty after stripping
                                    print(f"🔍 DEBUG verify_student_id: Found Senior High School field '{key}' = '{senior_high}'")
                                    break
                        
                        # Get Course value - try multiple field name variations
                        course_val = None
                        course_field_names = ['Course', 'course', 'Department', 'department', 'Dept', 'dept', 'Program', 'program']
                        for field_name in course_field_names:
                            if field_name in data:
                                raw_value = data.get(field_name)
                                if raw_value:
                                    course_val = str(raw_value).strip()  # Strip whitespace and newlines
                                    if course_val:  # Only use if not empty after stripping
                                        print(f"🔍 DEBUG verify_student_id: Found Course field '{field_name}' = '{course_val}'")
                                        break
                        
                        # If still not found, try case-insensitive search
                        if not course_val:
                            for key, value in data.items():
                                if key and isinstance(value, str):
                                    key_lower = key.lower()
                                    if key_lower in ('course', 'department', 'dept', 'program'):
                                        course_val = str(value).strip()
                                        if course_val:
                                            print(f"🔍 DEBUG verify_student_id: Found Course field (case-insensitive) '{key}' = '{course_val}'")
                                            break
                        
                        # Use Senior High School format if found, otherwise use Course
                        if senior_high:
                            course_val = f"SHS {senior_high}"
                        elif not course_val:
                            course_val = 'Unknown'
                            print(f"⚠️ DEBUG verify_student_id: Course not found in any field, defaulting to 'Unknown'")
                        
                        # Get Gender value - try multiple field name variations
                        gender_val = None
                        gender_field_names = ['Gender', 'gender', 'Sex', 'sex', 'Gender\n', 'gender\n']
                        for field_name in gender_field_names:
                            if field_name in data:
                                raw_value = data.get(field_name)
                                if raw_value:
                                    gender_val = str(raw_value).strip()  # Strip whitespace and newlines
                                    if gender_val:  # Only use if not empty after stripping
                                        print(f"🔍 DEBUG verify_student_id: Found Gender field '{field_name}' = '{gender_val}'")
                                        break
                        
                        # If still not found, try case-insensitive search
                        if not gender_val:
                            for key, value in data.items():
                                if key and isinstance(value, str):
                                    key_lower = key.lower()
                                    if key_lower in ('gender', 'sex'):
                                        gender_val = str(value).strip()
                                        if gender_val:
                                            print(f"🔍 DEBUG verify_student_id: Found Gender field (case-insensitive) '{key}' = '{gender_val}'")
                                            break
                        
                        if not gender_val:
                            gender_val = 'Unknown'
                            print(f"⚠️ DEBUG verify_student_id: Gender not found in any field, defaulting to 'Unknown'")

                        student_info = {
                            'student_id': data.get('Student Number', data.get('student_number', norm_id)),
                            'name': data.get('Name', data.get('name', f'Student {norm_id}')),
                            'course': course_val,
                            'gender': gender_val,
                            'rfid': data.get('rfid', data.get('RFID', None))
                        }
                        
                        print(f"✅ DEBUG verify_student_id: Built student_info - Course: '{course_val}', Gender: '{gender_val}'")
                else:
                    # Check offline file explicitly (if exists)
                    import json
                    if os.path.exists("offline_students.json"):
                        with open("offline_students.json", "r") as f:
                            offline_students = json.load(f)
                        if student_id in offline_students:
                            student_info = offline_students[student_id]

            except Exception as e:
                print(f"WARNING: Error checking student ID in verify_student_id: {e}")

            # If still not found, show explicit invalid message
            if not student_info:
                messagebox.showerror("Invalid Number", 
                                   f"Student ID '{student_id}' is not found in the system.\n\n"
                                   "Please enter a valid student ID number from the student records.")
                return

            # If found, ensure student_info fields are normalized for display
            # Some callers expect keys: name, course, gender
            # Only set to 'Unknown' if value is truly missing or empty after stripping
            if 'name' not in student_info or not student_info.get('name'):
                student_info['name'] = student_info.get('name', f"Student {student_id}")
            
            # Check if course exists and is not empty/whitespace
            course_value = student_info.get('course', '').strip() if student_info.get('course') else ''
            if not course_value or course_value == 'Unknown':
                # Try to get from Firebase again if available
                if self.firebase_initialized and self.db:
                    try:
                        # Re-fetch to ensure we have latest data
                        students_ref = self.db.collection('students')
                        doc_ref = students_ref.document(norm_id)
                        doc = doc_ref.get()
                        if doc.exists:
                            data = doc.to_dict()
                            # Try to get course again
                            for field_name in ['Course', 'course', 'Department', 'department']:
                                if field_name in data:
                                    raw_val = str(data.get(field_name, '')).strip()
                                    if raw_val:
                                        course_value = raw_val
                                        student_info['course'] = course_value
                                        print(f"🔍 DEBUG verify_student_id: Re-fetched Course = '{course_value}'")
                                        break
                    except Exception as e:
                        print(f"⚠️ DEBUG verify_student_id: Could not re-fetch Course: {e}")
            
            if not course_value:
                student_info['course'] = 'Unknown'
            
            # Check if gender exists and is not empty/whitespace
            gender_value = student_info.get('gender', '').strip() if student_info.get('gender') else ''
            if not gender_value or gender_value == 'Unknown':
                # Try multiple methods to find Gender field
                if self.firebase_initialized and self.db:
                    try:
                        students_ref = self.db.collection('students')
                        
                        # Method 1: Try document ID lookup again
                        doc_ref = students_ref.document(norm_id)
                        doc = doc_ref.get()
                        if doc.exists:
                            data = doc.to_dict()
                            # Try to get gender
                            for field_name in ['Gender', 'gender', 'Sex', 'sex']:
                                if field_name in data:
                                    raw_val = str(data.get(field_name, '')).strip()
                                    if raw_val:
                                        gender_value = raw_val
                                        student_info['gender'] = gender_value
                                        print(f"🔍 DEBUG verify_student_id: Re-fetched Gender (doc ID) = '{gender_value}'")
                                        break
                        
                        # Method 2: If still not found, try querying by student_id field
                        if not gender_value or gender_value == 'Unknown':
                            try:
                                query = students_ref.where('student_id', '==', norm_id)
                                docs = query.get()
                                if docs:
                                    for doc in docs:
                                        data = doc.to_dict()
                                        for field_name in ['Gender', 'gender', 'Sex', 'sex']:
                                            if field_name in data:
                                                raw_val = str(data.get(field_name, '')).strip()
                                                if raw_val:
                                                    gender_value = raw_val
                                                    student_info['gender'] = gender_value
                                                    print(f"🔍 DEBUG verify_student_id: Re-fetched Gender (student_id query) = '{gender_value}'")
                                                    break
                                        if gender_value and gender_value != 'Unknown':
                                            break
                            except Exception as e:
                                print(f"⚠️ DEBUG verify_student_id: Query by student_id failed: {e}")
                        
                        # Method 3: If still not found, try scanning all documents for matching Student Number
                        if not gender_value or gender_value == 'Unknown':
                            try:
                                docs = students_ref.get()
                                for doc in docs:
                                    data = doc.to_dict()
                                    # Check if this document has the student ID
                                    student_num_match = False
                                    for field in ('Student Number', 'student_number', 'student_id'):
                                        if field in data:
                                            val = str(data.get(field, '')).strip()
                                            if val == norm_id:
                                                student_num_match = True
                                                break
                                    
                                    if student_num_match:
                                        # Found matching student, try to get gender
                                        for field_name in ['Gender', 'gender', 'Sex', 'sex']:
                                            if field_name in data:
                                                raw_val = str(data.get(field_name, '')).strip()
                                                if raw_val:
                                                    gender_value = raw_val
                                                    student_info['gender'] = gender_value
                                                    print(f"🔍 DEBUG verify_student_id: Re-fetched Gender (scan) = '{gender_value}' from doc ID: {doc.id}")
                                                    break
                                        if gender_value and gender_value != 'Unknown':
                                            break
                            except Exception as e:
                                print(f"⚠️ DEBUG verify_student_id: Scan method failed: {e}")
                                
                    except Exception as e:
                        print(f"⚠️ DEBUG verify_student_id: Could not re-fetch Gender: {e}")
            
            if not gender_value:
                student_info['gender'] = 'Unknown'
                print(f"⚠️ DEBUG verify_student_id: Gender still not found after all attempts")
            
            print(f"✅ DEBUG verify_student_id: Final student_info - Course: '{student_info.get('course')}', Gender: '{student_info.get('gender')}'")

            # Do NOT display student info immediately. Store a pending assignment
            # The guard will select an available empty RFID and assign it; only when
            # the temporary RFID is tapped by the student will the main UI show their info.
            try:
                self.pending_assignment_student = student_info
                # Show the RFID assignment UI so guard can pick an empty card
                self.show_rfid_assignment()
                # Log that the student was located and is awaiting temporary RFID assignment
                self.add_activity_log(f"Student located for temporary RFID assignment: {student_info['name']} - ID: {student_id}")
                # Do not update main screen or display full student info yet
            except Exception as e:
                print(f"WARNING: Failed to set pending assignment for student {student_id}: {e}")
            
            # Display student information
            self.display_student_info(student_info)
            
            # Show RFID assignment section
            self.show_rfid_assignment()
            
            # Add to activity log
            self.add_activity_log(f"Student ID verified: {student_info['name']} ({student_info['course']}) - ID: {student_id}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to verify student ID: {e}")
    
    def display_student_info(self, student_info):
        """Display verified student information"""
        # Clear previous content
        for widget in self.student_info_frame.winfo_children():
            widget.destroy()
        
        info_content = tk.Frame(self.student_info_frame, bg='#ffffff')
        info_content.pack(fill=tk.X, padx=20, pady=20)
        
        # Student details
        name_label = tk.Label(
            info_content,
            text=f"Name: {student_info['name']}",
            font=('Arial', 14, 'bold'),
            fg='#374151',
            bg='#ffffff'
        )
        name_label.pack(anchor=tk.W, pady=(0, 5))
        
        # Show 'Senior High School' instead of 'Course' when course starts with 'SHS'
        course_value = str(student_info.get('course', 'Unknown'))
        if course_value.upper().startswith('SHS'):
            shs_text = course_value[3:].strip()
            if shs_text:
                course_text = f"Senior High School: {shs_text}"
            else:
                course_text = "Senior High School"
        else:
            course_text = f"Course: {course_value}"

        course_label = tk.Label(
            info_content,
            text=course_text,
            font=('Arial', 14, 'bold'),
            fg='#374151',
            bg='#ffffff'
        )
        course_label.pack(anchor=tk.W, pady=(0, 5))
        
        gender_label = tk.Label(
            info_content,
            text=f"Gender: {student_info['gender']}",
            font=('Arial', 14, 'bold'),
            fg='#374151',
            bg='#ffffff'
        )
        gender_label.pack(anchor=tk.W, pady=(0, 5))
        
        id_label = tk.Label(
            info_content,
            text=f"Student ID: {self.student_id_entry.get()}",
            font=('Arial', 14, 'bold'),
            fg='#1e3a8a',
            bg='#ffffff'
        )
        id_label.pack(anchor=tk.W, pady=(5, 0))
    
    def show_rfid_assignment(self):
        """Show RFID assignment section - Tap RFID version"""
        # Clear previous content
        for widget in self.rfid_assignment_frame.winfo_children():
            widget.destroy()
        
        rfid_content = tk.Frame(self.rfid_assignment_frame, bg='#ffffff')
        rfid_content.pack(fill=tk.X, padx=20, pady=20)
        
        # Title section
        title_frame = tk.Frame(rfid_content, bg='#ffffff')
        title_frame.pack(fill=tk.X, pady=(0, 15))
        
        rfid_title_label = tk.Label(
            title_frame,
            text="📡 RFID Assignment for Student",
            font=('Arial', 16, 'bold'),
            fg='#1e3a8a',
            bg='#ffffff'
        )
        rfid_title_label.pack()
        
        rfid_subtitle_label = tk.Label(
            title_frame,
            text="Tap an empty RFID card to assign it to this student",
            font=('Arial', 12),
            fg='#6b7280',
            bg='#ffffff'
        )
        rfid_subtitle_label.pack(pady=(5, 0))
        
        # Tap RFID instruction section
        tap_frame = tk.Frame(rfid_content, bg='#f0f9ff', relief='solid', bd=2)
        tap_frame.pack(fill=tk.X, pady=(0, 15))
        
        tap_content = tk.Frame(tap_frame, bg='#f0f9ff')
        tap_content.pack(fill=tk.X, padx=20, pady=20)
        
        # Button to activate RFID tap mode
        self.tap_rfid_btn = tk.Button(
            tap_content,
            text="📱 Tap Empty RFID",
            command=self.activate_rfid_tap_mode,
            font=('Arial', 16, 'bold'),
            bg='#3b82f6',
            fg='white',
            relief='raised',
            bd=3,
            padx=30,
            pady=15,
            cursor='hand2',
            activebackground='#2563eb',
            activeforeground='white'
        )
        self.tap_rfid_btn.pack(pady=(0, 15))
        
        # RFID Entry field (initially hidden, shown after button click)
        self.rfid_tap_entry_frame = tk.Frame(tap_content, bg='#f0f9ff')
        self.rfid_tap_entry_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Label for the entry field
        self.rfid_entry_label = tk.Label(
            self.rfid_tap_entry_frame,
            text="RFID Number:",
            font=('Arial', 12, 'bold'),
            fg='#1e3a8a',
            bg='#f0f9ff'
        )
        self.rfid_entry_label.pack(anchor=tk.W, pady=(0, 5))
        
        # Entry field for RFID input (will be visible when button is clicked)
        self.rfid_tap_entry_var = tk.StringVar()
        self.rfid_tap_entry = tk.Entry(
            self.rfid_tap_entry_frame,
            textvariable=self.rfid_tap_entry_var,
            font=('Arial', 18, 'bold'),
            width=25,
            justify=tk.CENTER,
            relief='solid',
            bd=3,
            bg='#ffffff',
            fg='#1e3a8a',
            insertbackground='#1e3a8a',
            state='normal'
        )
        self.rfid_tap_entry.pack(fill=tk.X, pady=(0, 10))
        self.rfid_tap_entry.bind('<Return>', lambda e: self.process_rfid_tap_entry())
        
        # Initially hide the entry field
        self.rfid_tap_entry_frame.pack_forget()
        
        # Status label (will be updated when button is clicked)
        self.rfid_tap_status_label = tk.Label(
            tap_content,
            text="Click the button above, then tap an empty RFID card",
            font=('Arial', 12),
            fg='#6b7280',
            bg='#f0f9ff',
            justify=tk.CENTER
        )
        self.rfid_tap_status_label.pack()
        
        # Initially assignment mode is NOT active (waiting for button click)
        self.rfid_assignment_active = False
        
        # Ensure assign button is hidden
        self.assign_rfid_btn.pack_forget()
    
    def activate_rfid_tap_mode(self):
        """Activate RFID tap mode - enable assignment when empty RFID is tapped (direct tap, no dialog)"""
        try:
            # Set assignment mode flag
            self.rfid_assignment_active = True
            
            # Update button to show waiting state
            if hasattr(self, 'tap_rfid_btn'):
                self.tap_rfid_btn.config(state=tk.DISABLED, text="⏳ Waiting for RFID tap...")
            
            # Show status label
            if hasattr(self, 'rfid_tap_status_label'):
                self.rfid_tap_status_label.config(
                    text="⏳ Waiting for empty RFID tap...\n\nTap an empty RFID card now:",
                    fg='#1e40af',
                    font=('Arial', 12, 'bold')
                )
                self.rfid_tap_status_label.pack()
            
            # Show entry field (hidden initially, but will receive RFID from process_rfid_input)
            if hasattr(self, 'rfid_tap_entry_frame'):
                self.rfid_tap_entry_frame.pack(fill=tk.X, pady=(10, 0))
            
            print(f"✅ Student RFID assignment mode activated - waiting for RFID tap...")
            self.add_activity_log("Student RFID assignment mode activated - waiting for empty RFID tap")
            
            # Focus the hidden RFID entry field so taps are captured
            self.root.after(100, self.set_rfid_entry_focus)
            
        except Exception as e:
            print(f"ERROR: Error activating RFID tap mode: {e}")
            import traceback
            traceback.print_exc()
            self.rfid_assignment_active = False
            self.reset_rfid_tap_button()
    
    def process_rfid_tap_entry(self):
        """Process RFID from the visible entry field in RFID Assignment section"""
        try:
            rfid = self.rfid_tap_entry_var.get().strip()
            if not rfid:
                return
            
            print(f"🔍 RFID detected from tap entry field: {rfid}")
            
            # Check if we're in RFID assignment mode
            if hasattr(self, 'rfid_assignment_active') and self.rfid_assignment_active:
                # First check if RFID is already assigned
                if not self.is_empty_rfid_available(rfid):
                    print(f"⚠️ Tapped RFID {rfid} is not an available empty RFID")
                    messagebox.showerror(
                        "RFID Not Available",
                        f"RFID {rfid} is already assigned to a student or visitor.\n\nPlease return it first or choose an available RFID."
                    )
                    # Clear entry and reset button
                    self.rfid_tap_entry_var.set("")
                    self.reset_rfid_tap_button()
                    return
                
                # Check if student already has an RFID assigned
                if hasattr(self, 'pending_assignment_student') and self.pending_assignment_student:
                    student_id = str(self.pending_assignment_student.get('student_id', '')).strip()
                else:
                    student_id = self.student_id_entry.get().strip()
                
                if student_id:
                    has_rfid, existing_rfid = self.check_student_already_has_rfid(student_id)
                    if has_rfid:
                        print(f"⚠️ Student {student_id} already has RFID {existing_rfid}")
                        messagebox.showerror(
                            "Student Already Has RFID",
                            f"This student is already assigned to RFID {existing_rfid}.\n\nPlease return the existing RFID before assigning a new one."
                        )
                        # Clear entry and reset button
                        self.rfid_tap_entry_var.set("")
                        self.reset_rfid_tap_button()
                        return
                
                # All checks passed - proceed with assignment
                print(f"✅ Empty RFID detected in assignment mode: {rfid}")
                # Clear the entry field
                self.rfid_tap_entry_var.set("")
                # Automatically assign student to this RFID
                try:
                    self.assign_rfid_to_student(rfid=rfid)
                    # Verify assignment was successful
                    if rfid in getattr(self, 'student_rfid_assignments', {}):
                        print(f"✅ RFID {rfid} assigned successfully - student can now use this RFID for entry/exit")
                        print(f"✅ DEBUG: Verified assignment in student_rfid_assignments: {self.student_rfid_assignments[rfid]}")
                    else:
                        print(f"⚠️ WARNING: Assignment may have failed - RFID {rfid} not found in student_rfid_assignments after assignment")
                except Exception as e:
                    print(f"⚠️ ERROR: Failed to assign RFID {rfid}: {e}")
                    import traceback
                    traceback.print_exc()
                    messagebox.showerror("Assignment Error", f"Failed to assign RFID {rfid} to student.\n\nError: {e}")
                # Reset assignment mode after successful assignment
                # (assign_rfid_to_student also resets it, but ensure it's reset here too)
                self.rfid_assignment_active = False
                self.reset_rfid_tap_button()
            else:
                # Not in assignment mode - clear and refocus
                self.rfid_tap_entry_var.set("")
                if hasattr(self, 'rfid_tap_entry'):
                    self.rfid_tap_entry.focus_set()
                    
        except Exception as e:
            print(f"ERROR: Error processing RFID tap entry: {e}")
            import traceback
            traceback.print_exc()
    
    def reset_rfid_tap_button(self):
        """Reset the RFID tap button to initial state"""
        try:
            if hasattr(self, 'tap_rfid_btn'):
                self.tap_rfid_btn.config(state=tk.NORMAL, text="📱 Tap Empty RFID")
            # Hide the entry field
            if hasattr(self, 'rfid_tap_entry_frame'):
                self.rfid_tap_entry_frame.pack_forget()
            # Clear the entry field
            if hasattr(self, 'rfid_tap_entry_var'):
                self.rfid_tap_entry_var.set("")
            # Show status label again (if it was hidden)
            if hasattr(self, 'rfid_tap_status_label'):
                self.rfid_tap_status_label.config(
                    text="Click the button above, then tap an empty RFID card",
                    fg='#6b7280',
                    font=('Arial', 12)
                )
                self.rfid_tap_status_label.pack()
        except Exception as e:
            print(f"⚠️ Warning: Could not reset RFID tap button: {e}")
    
    def update_student_rfid_list(self):
        """Update the list of available RFID cards for student forgot ID from Firebase"""
        try:
            available_rfids = []
            
            # Load RFID cards from Firebase Empty RFID collection
            if self.firebase_initialized and self.db:
                print("INFO: Loading RFID cards for student forgot ID from Firebase...")
                empty_rfid_ref = self.db.collection('Empty RFID')
                docs = empty_rfid_ref.get()
                
                for doc in docs:
                    rfid_data = doc.to_dict() or {}
                    rfid_id = doc.id
                    
                    # CRITICAL: Check availability from multiple sources
                    is_available = True
                    
                    # Method 1: Check Empty RFID document fields
                    if rfid_data:
                        # Check assigned_to field - if it exists and has a value, RFID is unavailable
                        if 'assigned_to' in rfid_data and rfid_data['assigned_to']:
                            is_available = False
                            print(f"🔍 DEBUG: RFID {rfid_id} marked unavailable via Empty RFID assigned_to field: {rfid_data['assigned_to']}")
                        # Also check available field - if it's explicitly False, RFID is unavailable
                        if 'available' in rfid_data and not rfid_data['available']:
                            is_available = False
                            print(f"🔍 DEBUG: RFID {rfid_id} marked unavailable via Empty RFID available field: False")
                    
                    # Method 2: Check student_rfid_assignments collection for active assignments
                    # ALWAYS check this, even if Method 1 marked it as unavailable, to ensure consistency
                    try:
                        student_assignments_ref = self.db.collection('student_rfid_assignments')
                        student_assignments = student_assignments_ref.where('rfid', '==', rfid_id).where('status', '==', 'active').get()
                        
                        if student_assignments:
                            # Check if any assignment is still valid (not expired)
                            current_time = datetime.now()
                            for assignment_doc in student_assignments:
                                assignment_data = assignment_doc.to_dict()
                                expiry_time_str = assignment_data.get('expiry_time')
                                if expiry_time_str:
                                    try:
                                        expiry_dt = datetime.strptime(expiry_time_str, "%Y-%m-%d %H:%M:%S")
                                        if current_time < expiry_dt:
                                            # Still valid assignment found
                                            is_available = False
                                            print(f"🔍 DEBUG: RFID {rfid_id} marked unavailable - active student assignment found (expires: {expiry_time_str})")
                                            break
                                    except Exception as e:
                                        print(f"⚠️ Warning: Error parsing expiry_time for {rfid_id}: {e}")
                                        # If we can't parse expiry, assume it's still active
                                        is_available = False
                                        print(f"🔍 DEBUG: RFID {rfid_id} marked unavailable - active student assignment (error parsing expiry)")
                                        break
                                else:
                                    # No expiry, assume active
                                    is_available = False
                                    print(f"🔍 DEBUG: RFID {rfid_id} marked unavailable - active student assignment (no expiry)")
                                    break
                    except Exception as e:
                        print(f"⚠️ Warning: Could not check student_rfid_assignments for {rfid_id}: {e}")
                    
                    # Method 3: Check visitors collection for active visitor assignments
                    # ALWAYS check this, even if already marked unavailable, to ensure consistency
                    try:
                        visitors_ref = self.db.collection('visitors')
                        visitors = visitors_ref.where('rfid', '==', rfid_id).get()
                        
                        for visitor_doc in visitors:
                            visitor_data = visitor_doc.to_dict()
                            status = visitor_data.get('status', '')
                            time_out = visitor_data.get('time_out')
                            
                            # Skip if visitor has exited (status is 'exited' and has time_out)
                            if status == 'exited' and time_out:
                                print(f"🔍 DEBUG: RFID {rfid_id} visitor has exited (status: {status}, time_out: {time_out}), skipping")
                                continue
                            
                            # Check if assignment is still valid (not expired)
                            expiry_time_str = visitor_data.get('expiry_time')
                            current_time = datetime.now()
                            is_expired = False
                            
                            if expiry_time_str:
                                try:
                                    expiry_dt = datetime.strptime(expiry_time_str, "%Y-%m-%d %H:%M:%S")
                                    if current_time >= expiry_dt:
                                        # Assignment has expired, skip this visitor
                                        is_expired = True
                                        print(f"🔍 DEBUG: RFID {rfid_id} visitor assignment expired (expiry: {expiry_time_str})")
                                except Exception as e:
                                    print(f"⚠️ Warning: Error parsing visitor expiry_time for {rfid_id}: {e}")
                            
                            # Skip if expired
                            if is_expired:
                                continue
                            
                            # RFID is unavailable if visitor is:
                            # 1. Status is 'registered' (just registered, not yet used) - this means RFID is assigned
                            # 2. Status is 'active' (currently in the building)
                            # 3. Has time_in but no time_out (active visitor who has entered)
                            # Note: We only mark as unavailable if status is explicitly 'registered' or 'active', 
                            # or if they have time_in without time_out. We don't mark as unavailable if status is 'exited' or empty.
                            if status == 'registered' or status == 'active' or (visitor_data.get('time_in') and not time_out):
                                is_available = False
                                print(f"🔍 DEBUG: RFID {rfid_id} marked unavailable - visitor assignment found (status: {status}, time_out: {time_out}, expiry: {expiry_time_str})")
                                break
                    except Exception as e:
                        print(f"⚠️ Warning: Could not check visitors collection for {rfid_id}: {e}")
                    
                    # Mark availability map
                    self.student_rfid_availability_map[rfid_id] = is_available
                    # Include both available and unavailable in the list (but label them)
                    available_rfids.append((rfid_id, is_available))
                    if is_available:
                        print(f"✅ INFO: Available RFID for student forgot ID: {rfid_id}")
                    else:
                        print(f"❌ INFO: Unavailable RFID: {rfid_id} - checked Empty RFID, student_rfid_assignments, and visitors collections")
                
                print(f"SUCCESS: Loaded {len(available_rfids)} available RFID cards from Firebase")
            else:
                print("WARNING: Firebase not available - using empty list")
                available_rfids = []
            
            # Build display list - Show ALL RFIDs with availability labels
            # CRITICAL: Use dict to deduplicate - each RFID should appear only once
            rfid_dict = {}  # rfid_id -> is_available (ensures no duplicates)
            
            for item in available_rfids:
                if isinstance(item, tuple) and len(item) == 2:
                    rfid_id, is_available = item
                else:
                    rfid_id = item
                    is_available = True
                
                # CRITICAL: If RFID already exists, use the most restrictive availability (False takes precedence)
                if rfid_id in rfid_dict:
                    # If current is unavailable or existing is unavailable, mark as unavailable
                    if not is_available or not rfid_dict[rfid_id]:
                        rfid_dict[rfid_id] = False
                else:
                    rfid_dict[rfid_id] = is_available
            
            # Now build display list from deduplicated dict
            display_list = []
            self.student_rfid_display_map = {}
            for rfid_id, is_available in rfid_dict.items():
                # Show ALL RFIDs with availability status clearly labeled
                if is_available:
                    label = f"{rfid_id}(available)"  # Available: show with (available) label
                else:
                    label = f"{rfid_id}(not available)"  # Unavailable: show with (not available) label
                
                display_list.append(label)
                self.student_rfid_display_map[label] = rfid_id
                
                # Save availability in map (for all RFIDs, available or not)
                self.student_rfid_availability_map[rfid_id] = is_available

            # Update the dropdown with labeled entries if widget exists
            try:
                if hasattr(self, 'student_rfid_dropdown') and self.student_rfid_dropdown:
                    self.student_rfid_dropdown['values'] = display_list
                if hasattr(self, 'student_rfid_var'):
                    self.student_rfid_var.set("")
            except Exception:
                pass
            
        except Exception as e:
            print(f"ERROR: Error updating student RFID list: {e}")
            if hasattr(self, 'student_rfid_dropdown'):
                self.student_rfid_dropdown['values'] = []
            if hasattr(self, 'student_rfid_var'):
                self.student_rfid_var.set("")
    
    def assign_rfid_to_student(self, rfid=None):
        """Assign RFID to verified student for forgot ID
        
        Args:
            rfid: Optional RFID to assign. If provided, uses this directly.
                  If None, tries to read from dropdown (for backward compatibility)
        """
        try:
            # If RFID is provided as parameter (from tap), use it directly
            if rfid:
                selected_rfid = rfid
            else:
                # Try to get from dropdown if it exists (for backward compatibility)
                if hasattr(self, 'student_rfid_var'):
                    selected_rfid_display = self.student_rfid_var.get()
                    # Map display label back to raw RFID if necessary
                    selected_rfid = self.student_rfid_display_map.get(selected_rfid_display, selected_rfid_display) if hasattr(self, 'student_rfid_display_map') else selected_rfid_display
                else:
                    messagebox.showerror("Error", "No RFID provided and no dropdown available. Please tap an empty RFID card.")
                    return
            
            # CRITICAL: Check if selected RFID is unavailable (from dropdown selection handler)
            # This is a double-check in case the selection handler didn't catch it
            if selected_rfid in self.student_rfid_availability_map and not self.student_rfid_availability_map.get(selected_rfid, False):
                messagebox.showerror("Unavailable RFID", f"RFID {selected_rfid} is currently unavailable.\n\nThis RFID is already assigned to a student or visitor.\n\nPlease return it first or choose an available RFID.")
                return
            
            if hasattr(self, 'pending_assignment_student') and self.pending_assignment_student:
                student_info = self.pending_assignment_student
                # normalize student_id from pending info if available
                student_id = str(student_info.get('student_id', '')).strip()
            else:
                student_id = self.student_id_entry.get().strip()
                # Only try to get from dropdown if it exists
                if not rfid and hasattr(self, 'student_rfid_var'):
                    selected_rfid = self.student_rfid_var.get()
                
                if not selected_rfid:
                    messagebox.showerror("Error", "Please select an available RFID card.")
                    return
            
            # CRITICAL: Check if student already has an RFID assigned
            if student_id:
                has_rfid, existing_rfid = self.check_student_already_has_rfid(student_id)
                if has_rfid:
                    messagebox.showerror(
                        "Student Already Has RFID",
                        f"This student is already assigned to RFID {existing_rfid}.\n\nPlease return the existing RFID before assigning a new one."
                    )
                    return
            
            # CRITICAL: Check if selected RFID is unavailable (from dropdown selection handler)
            # This is a double-check in case the selection handler didn't catch it
            raw_rfid = selected_rfid
            if hasattr(self, 'student_rfid_display_map'):
                raw_rfid = self.student_rfid_display_map.get(selected_rfid, selected_rfid)
            if raw_rfid in self.student_rfid_availability_map and not self.student_rfid_availability_map.get(raw_rfid, False):
                messagebox.showerror("Unavailable RFID", f"RFID {raw_rfid} is currently unavailable.\n\nThis RFID is already assigned to a student or visitor.\n\nPlease return it first or choose an available RFID.")
                return
            
            # CRITICAL: Check Firebase for current availability status (not just local memory)
            is_available_in_firebase = True
            if self.firebase_initialized and self.db:
                try:
                    rfid_ref = self.db.collection('Empty RFID').document(selected_rfid)
                    rfid_doc = rfid_ref.get()
                    if rfid_doc.exists:
                        rfid_data = rfid_doc.to_dict() or {}
                        # Check if RFID is assigned or unavailable
                        if 'assigned_to' in rfid_data and rfid_data['assigned_to']:
                            is_available_in_firebase = False
                            assigned_to = rfid_data.get('assigned_to', 'unknown')
                            messagebox.showerror("Error", f"RFID {selected_rfid} is already assigned to a {assigned_to}. Choose another card or return it first.")
                            return
                        elif 'available' in rfid_data and not rfid_data['available']:
                            is_available_in_firebase = False
                            messagebox.showerror("Error", f"RFID {selected_rfid} is currently unavailable. Choose another card or return/clear it first.")
                            return
                except Exception as e:
                    print(f"⚠️ Warning: Could not check Firebase for RFID availability: {e}")
            
            # Re-check local availability map
            if selected_rfid in self.student_rfid_availability_map and not self.student_rfid_availability_map.get(selected_rfid, False):
                messagebox.showerror("Error", f"RFID {selected_rfid} is currently unavailable. Choose another card or return/clear it first.")
                return
            
            # Check if RFID is already assigned to a student (local memory)
            if selected_rfid in self.student_rfid_assignments:
                messagebox.showerror("Error", f"RFID {selected_rfid} is already assigned to another student.")
                return
            
            # Check if RFID is already assigned to a visitor (local memory)
            if hasattr(self, 'visitor_rfid_assignments') and selected_rfid in self.visitor_rfid_assignments:
                messagebox.showerror("Error", f"RFID {selected_rfid} is already assigned to a visitor. Choose another card or return it first.")
                return
            
            # Also check visitor_rfid_registry
            if hasattr(self, 'visitor_rfid_registry') and selected_rfid in self.visitor_rfid_registry:
                messagebox.showerror("Error", f"RFID {selected_rfid} is already assigned to a visitor. Choose another card or return it first.")
                return
            
            # Ensure we have student_info (use pending if present, else fetch)
            if not (hasattr(self, 'pending_assignment_student') and self.pending_assignment_student):
                student_info = self.get_student_info(student_id)
            else:
                student_info = self.pending_assignment_student
            
            # Create temporary assignment record
            assignment_info = {
                'student_id': student_id,
                'name': student_info['name'],
                'course': student_info['course'],
                'gender': student_info['gender'],
                'rfid': selected_rfid,
                'assignment_time': self.get_current_timestamp(),
                'expiry_time': (datetime.now() + timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S"),
                'status': 'active'  # active, returned
            }
            
            # Link student to RFID
            self.student_rfid_assignments[selected_rfid] = assignment_info
            print(f"✅ DEBUG: Added RFID {selected_rfid} to student_rfid_assignments")
            print(f"✅ DEBUG: Assignment info: {assignment_info}")
            print(f"✅ DEBUG: Current student_rfid_assignments keys: {list(self.student_rfid_assignments.keys())}")

            # Mark as unavailable locally so it won't be used for visitors or other students
            try:
                self.student_rfid_availability_map[selected_rfid] = False
            except Exception:
                pass
            try:
                self.rfid_availability_map[selected_rfid] = False
            except Exception:
                pass

            # Remove from available RFID list container if present
            if selected_rfid in self.student_forgot_rfids:
                try:
                    self.student_forgot_rfids.remove(selected_rfid)
                except Exception:
                    pass
            
            # Save to Firebase
            self.save_student_rfid_assignment_to_firebase(assignment_info)
            # Update Firebase availability with assigned_to='student'
            try:
                self.update_rfid_availability_in_firebase(selected_rfid, available=False, assigned_to='student')
            except Exception as e:
                print(f"⚠️ Warning: Could not update Firebase RFID availability: {e}")

            # CRITICAL: Refresh BOTH dropdowns to reflect unavailable state
            # Student dropdown (forgot ID)
            try:
                self.update_student_rfid_list()
                print(f"✅ Refreshed student RFID dropdown after assignment")
            except Exception as e:
                print(f"⚠️ Warning: Could not refresh student RFID dropdown: {e}")
            
            # Visitor dropdown (also refresh to show RFID as unavailable)
            try:
                self.load_rfid_from_firebase()
                print(f"✅ Refreshed visitor RFID dropdown after student assignment")
            except Exception as e:
                print(f"⚠️ Warning: Could not refresh visitor RFID dropdown: {e}")
            
            # Add to activity log
            self.add_activity_log(f"Temporary RFID assigned to student: {student_info['name']} ({student_info['course']}) - Student ID: {student_id} - RFID: {selected_rfid} - Expires in 24h")
            
            # Show confirmation (but do not display main screen info yet)
            self.show_green_success_message("RFID Assigned Successfully", 
                              f"Student Details:\n"
                              f"Name: {student_info['name']}\n"
                              f"Course: {student_info['course']}\n"
                              f"Gender: {student_info['gender']}\n"
                              f"Student ID: {student_id}\n"
                              f"Assigned RFID: {selected_rfid}\n\n"
                              f"Temporary RFID valid for 24 hours. Student must tap the card to register entry/exit.")
            
            # Clear assignment mode flag and reset button
            self.rfid_assignment_active = False
            self.reset_rfid_tap_button()
            
            # Clear form and pending assignment
            try:
                self.student_id_entry.delete(0, tk.END)
            except Exception:
                pass
            if hasattr(self, 'student_rfid_var'):
                self.student_rfid_var.set("")
            try:
                self.pending_assignment_student = None
            except Exception:
                pass
            
            # Hide sections
            for widget in self.student_info_frame.winfo_children():
                widget.destroy()
            for widget in self.rfid_assignment_frame.winfo_children():
                widget.destroy()
            
            # Ensure assign button is hidden
            self.assign_rfid_btn.pack_forget()
            
            # Return to dashboard
            self.back_to_dashboard()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to assign RFID: {e}")

    def return_temp_student_rfid_dialog(self):
        """Show a dialog to let guard return/unassign a temporary student RFID immediately"""
        try:
            # Build list of currently assigned temporary RFIDs for students
            assigned = []
            if hasattr(self, 'student_rfid_assignments') and isinstance(self.student_rfid_assignments, dict):
                for rfid, info in self.student_rfid_assignments.items():
                    assigned.append(f"{rfid} - {info.get('name', 'Unknown')}")

            if not assigned:
                messagebox.showinfo("No Assigned RFIDs", "There are no temporary student RFIDs currently assigned.")
                return

            # Prompt guard to choose which RFID to return
            choice = simpledialog.askstring("Return RFID", "Enter RFID to return (e.g. 0095129433)\nAvailable assigned:\n" + "\n".join(assigned))
            if not choice:
                return
            choice = choice.strip()

            # If the RFID exists in assignments, reset it
            if choice in self.student_rfid_assignments:
                try:
                    # Remove assignment record
                    del self.student_rfid_assignments[choice]
                except Exception:
                    pass

                # Mark as available in local lists and Firebase
                if choice not in self.available_rfids:
                    self.available_rfids.append(choice)
                try:
                    self.update_rfid_availability_in_firebase(choice, available=True)
                except Exception:
                    pass

                # Refresh student RFID list UI
                try:
                    self.update_student_rfid_list()
                except Exception:
                    pass

                self.add_activity_log(f"Temporary student RFID returned manually: {choice}")
                messagebox.showinfo("RFID Returned", f"Temporary student RFID {choice} has been returned and is now available.")
            else:
                messagebox.showwarning("Not Found", f"RFID {choice} is not a currently assigned temporary student RFID.")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to return temporary student RFID: {e}")
    
    def handle_manual_visitor(self):
        """Handle manual visitor entry - open visitor information tab"""
        try:
            # Create visitor information tab
            self.create_visitor_tab()
            
            # Switch to visitor tab
            self.notebook.select(self.visitor_tab)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open visitor form: {e}")
    
    def create_visitor_tab(self):
        """Create visitor information tab"""
        if self.visitor_tab is None:
            self.visitor_tab = tk.Frame(self.notebook, bg='#ffffff')
            self.notebook.add(self.visitor_tab, text="👤 Visitor Information")
            
            # Create visitor form content
            self.create_visitor_form(self.visitor_tab)
    
    def create_visitor_form(self, parent):
        """Create visitor information form"""
        # Main container
        main_frame = tk.Frame(parent, bg='#ffffff')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=30)
        
        # Header with back button
        header_frame = tk.Frame(main_frame, bg='#ffffff')
        header_frame.pack(fill=tk.X, pady=(0, 30))
        
        # Back button at top
        back_btn = tk.Button(
            header_frame,
            text="← Back to Dashboard",
            command=self.back_to_dashboard,
            font=('Arial', 12, 'bold'),
            bg='#6b7280',
            fg='white',
            relief='raised',
            bd=3,
            padx=20,
            pady=8,
            cursor='hand2',
            activebackground='#4b5563',
            activeforeground='white'
        )
        back_btn.pack(anchor=tk.W, pady=(0, 15))
        
        title_label = tk.Label(
            header_frame,
            text="👤 Visitor Registration Form",
            font=('Arial', 24, 'bold'),
            fg='#1e3a8a',
            bg='#ffffff'
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            header_frame,
            text="Please fill in the visitor information and ID details",
            font=('Arial', 14),
            fg='#6b7280',
            bg='#ffffff'
        )
        subtitle_label.pack(pady=(5, 0))
        
        # Form container
        form_frame = tk.LabelFrame(
            main_frame,
            text="Visitor Details",
            font=('Arial', 16, 'bold'),
            fg='#1e3a8a',
            bg='#ffffff',
            relief='groove',
            bd=2
        )
        form_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # Form fields
        fields_frame = tk.Frame(form_frame, bg='#ffffff')
        fields_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Visitor Name - Separated into fields
        name_frame = tk.Frame(fields_frame, bg='#ffffff')
        name_frame.pack(fill=tk.X, pady=(0, 15))
        
        name_label = tk.Label(
            name_frame,
            text="Visitor Name:",
            font=('Arial', 14, 'bold'),
            fg='#374151',
            bg='#ffffff'
        )
        name_label.pack(anchor=tk.W, pady=(0, 10))
        
        # Name fields in a grid layout aligned to the right
        name_grid = tk.Frame(name_frame, bg='#ffffff')
        name_grid.pack(anchor=tk.E, fill=tk.X)
        
        # Surname
        surname_label = tk.Label(
            name_grid,
            text="Surname:",
            font=('Arial', 12, 'bold'),
            fg='#374151',
            bg='#ffffff'
        )
        surname_label.grid(row=0, column=0, sticky=tk.W, padx=(0, 10), pady=(0, 5))
        
        self.visitor_surname_entry = tk.Entry(
            name_grid,
            font=('Arial', 12),
            width=25,
            relief='solid',
            bd=2,
            bg='#f9fafb',
            fg='#1f2937',
            justify=tk.LEFT
        )
        self.visitor_surname_entry.grid(row=1, column=0, sticky=tk.E, padx=(0, 10))
        
        # First Name
        firstname_label = tk.Label(
            name_grid,
            text="First Name:",
            font=('Arial', 12, 'bold'),
            fg='#374151',
            bg='#ffffff'
        )
        firstname_label.grid(row=0, column=1, sticky=tk.W, padx=(0, 10), pady=(0, 5))
        
        self.visitor_firstname_entry = tk.Entry(
            name_grid,
            font=('Arial', 12),
            width=25,
            relief='solid',
            bd=2,
            bg='#f9fafb',
            fg='#1f2937',
            justify=tk.LEFT
        )
        self.visitor_firstname_entry.grid(row=1, column=1, sticky=tk.E, padx=(0, 10))
        
        # Middle Initial
        mi_label = tk.Label(
            name_grid,
            text="M.I.:",
            font=('Arial', 12, 'bold'),
            fg='#374151',
            bg='#ffffff'
        )
        mi_label.grid(row=0, column=2, sticky=tk.W, padx=(0, 10), pady=(0, 5))
        
        self.visitor_mi_entry = tk.Entry(
            name_grid,
            font=('Arial', 12),
            width=8,
            relief='solid',
            bd=2,
            bg='#f9fafb',
            fg='#1f2937',
            justify=tk.LEFT
        )
        self.visitor_mi_entry.grid(row=1, column=2, sticky=tk.E, padx=(0, 10))
        
        # Suffix (Optional)
        suffix_label = tk.Label(
            name_grid,
            text="Suffix: (Optional)",
            font=('Arial', 12, 'bold'),
            fg='#374151',
            bg='#ffffff'
        )
        suffix_label.grid(row=0, column=3, sticky=tk.W, pady=(0, 5))
        
        self.visitor_suffix_entry = tk.Entry(
            name_grid,
            font=('Arial', 12),
            width=10,
            relief='solid',
            bd=2,
            bg='#f9fafb',
            fg='#1f2937',
            justify=tk.LEFT
        )
        self.visitor_suffix_entry.grid(row=1, column=3, sticky=tk.E)
        
        # Configure grid to align to the right
        name_grid.columnconfigure(0, weight=0)
        name_grid.columnconfigure(1, weight=0)
        name_grid.columnconfigure(2, weight=0)
        name_grid.columnconfigure(3, weight=0)
        
        
        # Purpose of Visit - same width as surname field
        purpose_frame = tk.Frame(fields_frame, bg='#ffffff')
        purpose_frame.pack(fill=tk.X, pady=(0, 15))
        
        purpose_label = tk.Label(
            purpose_frame,
            text="Purpose of Visit:",
            font=('Arial', 14, 'bold'),
            fg='#374151',
            bg='#ffffff'
        )
        purpose_label.pack(anchor=tk.W, pady=(0, 5))
        
        # Align to the right, same width as surname
        purpose_grid = tk.Frame(purpose_frame, bg='#ffffff')
        purpose_grid.pack(anchor=tk.E, fill=tk.X)
        
        self.visitor_purpose_entry = tk.Entry(
            purpose_grid,
            font=('Arial', 14),
            width=25,
            relief='solid',
            bd=2,
            bg='#f9fafb',
            fg='#1f2937',
            justify=tk.LEFT
        )
        self.visitor_purpose_entry.grid(row=0, column=0, sticky=tk.E)
        
        # ID Type Selection - Dropdown
        id_type_frame = tk.Frame(fields_frame, bg='#ffffff')
        id_type_frame.pack(fill=tk.X, pady=(0, 15))
        
        id_type_label = tk.Label(
            id_type_frame,
            text="ID Type Surrendered:",
            font=('Arial', 14, 'bold'),
            fg='#374151',
            bg='#ffffff'
        )
        id_type_label.pack(anchor=tk.W, pady=(0, 5))
        
        self.id_type_var = tk.StringVar(value="")
        
        # ID Type dropdown - aligned to the right, same width as surname field
        id_type_options = ["Driver's License", "National ID", "Passport", "Other"]
        # Align to the right, same width as surname
        id_type_grid = tk.Frame(id_type_frame, bg='#ffffff')
        id_type_grid.pack(anchor=tk.E, fill=tk.X)
        self.id_type_dropdown = ttk.Combobox(
            id_type_grid,
            textvariable=self.id_type_var,
            values=id_type_options,
            font=('Arial', 12),
            state='readonly',
            width=25
        )
        self.id_type_dropdown.grid(row=0, column=0, sticky=tk.E)
        self.id_type_dropdown.bind('<<ComboboxSelected>>', self.on_visitor_id_type_change)
        
        # Custom ID type input field (initially hidden, shown when "Other" is selected)
        self.custom_id_frame = tk.Frame(id_type_frame, bg='#ffffff')
        self.custom_id_frame.pack(fill=tk.X, pady=(10, 0))
        
        custom_id_label = tk.Label(
            self.custom_id_frame,
            text="Specify ID Type:",
            font=('Arial', 12, 'bold'),
            fg='#374151',
            bg='#ffffff'
        )
        custom_id_label.pack(anchor=tk.W, pady=(0, 5))
        
        self.custom_id_entry = tk.Entry(
            self.custom_id_frame,
            font=('Arial', 12),
            width=30,
            relief='solid',
            bd=2,
            bg='#f9fafb',
            fg='#1f2937'
        )
        self.custom_id_entry.pack(fill=tk.X)
        
        # Initially hide the custom ID input
        self.custom_id_frame.pack_forget()
        
        # Number of Companion - Dropdown
        companion_frame = tk.Frame(fields_frame, bg='#ffffff')
        companion_frame.pack(fill=tk.X, pady=(0, 15))
        
        companion_label = tk.Label(
            companion_frame,
            text="Number of Companion:",
            font=('Arial', 14, 'bold'),
            fg='#374151',
            bg='#ffffff'
        )
        companion_label.pack(anchor=tk.W, pady=(0, 5))
        
        self.companion_count_var = tk.StringVar(value="0")
        
        # Companion count dropdown - aligned to the right, same width as surname field
        companion_options = ["0", "1", "2", "3", "4", "5", "6+"]
        # Align to the right, same width as surname
        companion_grid = tk.Frame(companion_frame, bg='#ffffff')
        companion_grid.pack(anchor=tk.E, fill=tk.X)
        self.companion_count_dropdown = ttk.Combobox(
            companion_grid,
            textvariable=self.companion_count_var,
            values=companion_options,
            font=('Arial', 12),
            state='readonly',
            width=25
        )
        self.companion_count_dropdown.grid(row=0, column=0, sticky=tk.E)
        
        # Available Empty RFID Section - Tap RFID version
        rfid_section = tk.LabelFrame(
            main_frame,
            text="Available Empty RFID",
            font=('Arial', 16, 'bold'),
            fg='#1e3a8a',
            bg='#ffffff',
            relief='groove',
            bd=2
        )
        rfid_section.pack(fill=tk.X, pady=(0, 20))
        
        rfid_content = tk.Frame(rfid_section, bg='#ffffff')
        rfid_content.pack(fill=tk.X, padx=20, pady=20)
        
        # Button to activate RFID tap mode
        self.visitor_tap_rfid_btn = tk.Button(
            rfid_content,
            text="📱 Tap Empty RFID",
            command=self.activate_visitor_rfid_tap_mode,
            font=('Arial', 16, 'bold'),
            bg='#3b82f6',
            fg='white',
            relief='raised',
            bd=3,
            padx=30,
            pady=15,
            cursor='hand2',
            activebackground='#2563eb',
            activeforeground='white'
        )
        self.visitor_tap_rfid_btn.pack(pady=(0, 15))
        
        # RFID Entry field (initially hidden, shown after button click)
        self.visitor_rfid_entry_frame = tk.Frame(rfid_content, bg='#ffffff')
        self.visitor_rfid_entry_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Label for the entry field
        self.visitor_rfid_entry_label = tk.Label(
            self.visitor_rfid_entry_frame,
            text="RFID Number:",
            font=('Arial', 12, 'bold'),
            fg='#1e3a8a',
            bg='#ffffff'
        )
        self.visitor_rfid_entry_label.pack(anchor=tk.W, pady=(0, 5))
        
        # Entry field for RFID input (will be visible when button is clicked)
        self.visitor_rfid_entry_var = tk.StringVar()
        self.visitor_rfid_entry = tk.Entry(
            self.visitor_rfid_entry_frame,
            textvariable=self.visitor_rfid_entry_var,
            font=('Arial', 18, 'bold'),
            width=25,
            justify=tk.CENTER,
            relief='solid',
            bd=3,
            bg='#ffffff',
            fg='#1e3a8a',
            insertbackground='#1e3a8a',
            state='normal'
        )
        self.visitor_rfid_entry.pack(fill=tk.X, pady=(0, 10))
        self.visitor_rfid_entry.bind('<Return>', lambda e: self.process_visitor_rfid_tap_entry())
        
        # Initially hide the entry field
        self.visitor_rfid_entry_frame.pack_forget()
        
        # Status label (will be updated when button is clicked)
        self.visitor_rfid_tap_status_label = tk.Label(
            rfid_content,
            text="Click the button above, then tap an empty RFID card",
            font=('Arial', 12),
            fg='#6b7280',
            bg='#ffffff',
            justify=tk.CENTER
        )
        self.visitor_rfid_tap_status_label.pack()
        
        # Initially assignment mode is NOT active (waiting for button click)
        self.visitor_rfid_assignment_active = False
        
        # Action buttons
        buttons_frame = tk.Frame(main_frame, bg='#ffffff')
        buttons_frame.pack(fill=tk.X, pady=(20, 0))
        
        # Register Visitor button (hidden - using tap RFID method instead)
        register_btn = tk.Button(
            buttons_frame,
            text="SUCCESS: Register Visitor & Assign RFID",
            command=self.register_visitor,
            font=('Arial', 14, 'bold'),
            bg='#10b981',
            fg='white',
            relief='raised',
            bd=3,
            padx=30,
            pady=12,
            cursor='hand2',
            activebackground='#059669',
            activeforeground='white'
        )
        register_btn.pack_forget()  # Hide button - RFID tap handles registration
    
    def back_to_dashboard(self):
        """Return to dashboard tab"""
        if self.dashboard_tab:
            self.notebook.select(self.dashboard_tab)

    def on_student_rfid_selected(self, event=None):
        """Handler for student combobox selection - prevent selecting unavailable RFID"""
        try:
            sel = self.student_rfid_var.get()
            raw = self.student_rfid_display_map.get(sel, sel)
            # If availability map says unavailable, warn and revert selection
            if raw in self.student_rfid_availability_map and not self.student_rfid_availability_map.get(raw, False):
                messagebox.showerror("Unavailable RFID", f"RFID {raw} is currently unavailable.\n\nThis RFID is already assigned to a student or visitor.\n\nPlease return it first or choose an available RFID.")
                # Revert to empty selection
                if hasattr(self, 'student_rfid_var'):
                    self.student_rfid_var.set("")
                return False
            return True
        except Exception as e:
            print(f"⚠️ Warning: Error in on_student_rfid_selected: {e}")
            return False

    def on_visitor_rfid_selected(self, event=None):
        """Handler for visitor combobox selection - prevent selecting unavailable RFID"""
        try:
            sel = self.available_rfid_var.get()
            raw = self.rfid_display_map.get(sel, sel)
            if raw in self.rfid_availability_map and not self.rfid_availability_map.get(raw, True):
                messagebox.showerror("Unavailable RFID", f"RFID {raw} is currently unavailable.\n\nThis RFID is already assigned to a student or visitor.\n\nPlease return it first or choose an available RFID.")
                # Revert to empty selection
                self.available_rfid_var.set("")
                return False
            return True
        except Exception as e:
            print(f"⚠️ Warning: Error in on_visitor_rfid_selected: {e}")
            return False
    
    def debug_firebase_rfid_collection(self):
        """Debug function to show all RFID data in Firebase"""
        try:
            if self.firebase_initialized and self.db:
                print("INFO: Debugging Firebase Empty RFID collection...")
                empty_rfid_ref = self.db.collection('Empty RFID')
                docs = empty_rfid_ref.get()
                
                print(f"INFO: Found {len(docs)} documents in Empty RFID collection:")
                
                for doc in docs:
                    rfid_data = doc.to_dict()
                    rfid_id = doc.id
                    print(f"  - Document ID: {rfid_id}")
                    print(f"    Data: {rfid_data}")
                    print()
                
                if len(docs) == 0:
                    print("WARNING: Empty RFID collection is empty!")
                    print("INFO: You need to add RFID documents to the Empty RFID collection")
                    print("INFO: Example document structure:")
                    print("  Document ID: RFID001")
                    print("  Data: { 'rfid_number': 'RFID001', 'status': 'available' }")
                
            else:
                print("ERROR: Firebase not initialized")
                
        except Exception as e:
            print(f"ERROR: Failed to debug Firebase collection: {e}")
            import traceback
            traceback.print_exc()
    
    def load_assigned_rfids_from_firebase(self):
        """Load assigned RFIDs from Firebase into memory (for return dialog)"""
        try:
            if not self.firebase_initialized or not self.db:
                return
            
            # Initialize if not exists
            if not hasattr(self, 'student_rfid_assignments'):
                self.student_rfid_assignments = {}
            if not hasattr(self, 'visitor_rfid_assignments'):
                self.visitor_rfid_assignments = {}
            
            # Load student RFID assignments from Firebase
            try:
                student_assignments_ref = self.db.collection('student_rfid_assignments')
                # Get all assignments (check status in loop for backward compatibility)
                student_assignments = student_assignments_ref.get()
                
                for assignment_doc in student_assignments:
                    assignment_data = assignment_doc.to_dict()
                    rfid = assignment_data.get('rfid')
                    status = assignment_data.get('status', 'active')  # Default to 'active' if no status field
                    
                    if rfid and status != 'returned':  # Only load non-returned assignments
                        # Check if assignment is still valid (not expired)
                        expiry_time_str = assignment_data.get('expiry_time')
                        is_valid = True
                        
                        if expiry_time_str:
                            try:
                                expiry_dt = datetime.strptime(expiry_time_str, "%Y-%m-%d %H:%M:%S")
                                if datetime.now() >= expiry_dt:
                                    # Expired, skip
                                    is_valid = False
                            except Exception:
                                pass
                        
                        if is_valid:
                            # Still valid, add to memory
                            self.student_rfid_assignments[rfid] = {
                                'student_id': assignment_data.get('student_id'),
                                'name': assignment_data.get('name', 'Unknown'),
                                'course': assignment_data.get('course', 'Unknown'),
                                'gender': assignment_data.get('gender', 'Unknown'),
                                'rfid': rfid,
                                'assignment_time': assignment_data.get('assignment_time'),
                                'expiry_time': expiry_time_str,
                                'status': status
                            }
                            print(f"✅ Loaded student RFID assignment from Firebase: {rfid} - {assignment_data.get('name', 'Unknown')}")
            except Exception as e:
                print(f"⚠️ Warning: Could not load student RFID assignments from Firebase: {e}")
            
            # Load visitor RFID assignments from Firebase (check visitors collection)
            try:
                visitors_ref = self.db.collection('visitors')
                # Get visitors with active RFID assignments (has RFID and is active)
                visitors = visitors_ref.get()
                for visitor_doc in visitors:
                    visitor_data = visitor_doc.to_dict()
                    rfid = visitor_data.get('rfid')
                    if rfid:
                        # Check if visitor is still active
                        status = visitor_data.get('status', '')
                        time_out = visitor_data.get('time_out')
                        if status == 'active' or (visitor_data.get('time_in') and not time_out):
                            # Visitor is active, RFID is assigned
                            self.visitor_rfid_assignments[rfid] = {
                                'name': visitor_data.get('name', 'Unknown'),
                                'rfid': rfid,
                                'assignment_time': visitor_data.get('time_in', ''),
                                'status': 'active'
                            }
                            print(f"✅ Loaded visitor RFID assignment from Firebase: {rfid} - {visitor_data.get('name', 'Unknown')}")
            except Exception as e:
                print(f"⚠️ Warning: Could not load visitor RFID assignments from Firebase: {e}")
                
        except Exception as e:
            print(f"⚠️ Warning: Error loading assigned RFIDs from Firebase: {e}")
    
    def cleanup_expired_rfid_assignments(self):
        """Clean up expired or orphaned RFID assignments on startup"""
        try:
            if not self.firebase_initialized or not self.db:
                return
            
            print("INFO: Cleaning up expired RFID assignments...")
            current_time = datetime.now()
            cleaned_count = 0
            
            # Get all unavailable RFIDs from Firebase
            empty_rfid_ref = self.db.collection('Empty RFID')
            docs = empty_rfid_ref.get()
            
            for doc in docs:
                rfid_data = doc.to_dict()
                rfid_id = doc.id
                
                # Check if RFID is marked as unavailable
                is_unavailable = False
                if rfid_data:
                    if 'assigned_to' in rfid_data and rfid_data['assigned_to']:
                        is_unavailable = True
                    elif 'available' in rfid_data and not rfid_data['available']:
                        is_unavailable = True
                
                if is_unavailable:
                    # Check for active assignments in student_rfid_assignments collection
                    student_assignments = self.db.collection('student_rfid_assignments').where('rfid', '==', rfid_id).get()
                    has_active_student_assignment = False
                    
                    for assignment_doc in student_assignments:
                        assignment_data = assignment_doc.to_dict()
                        expiry_time_str = assignment_data.get('expiry_time')
                        if expiry_time_str:
                            try:
                                expiry_dt = datetime.strptime(expiry_time_str, "%Y-%m-%d %H:%M:%S")
                                if current_time < expiry_dt:
                                    has_active_student_assignment = True
                                    break
                            except Exception:
                                pass
                    
                    # Check for active assignments in visitor_rfid_assignments (via visitor_rfid_registry logic)
                    # Also check visitors collection
                    visitors_with_rfid = self.db.collection('visitors').where('rfid', '==', rfid_id).get()
                    has_active_visitor_assignment = False
                    
                    for visitor_doc in visitors_with_rfid:
                        visitor_data = visitor_doc.to_dict()
                        # Check if visitor is still active (has time_in but no time_out)
                        time_in = visitor_data.get('time_in')
                        time_out = visitor_data.get('time_out')
                        status = visitor_data.get('status', '')
                        
                        # If visitor has checked out (time_out exists and status is 'exited'), RFID should be available
                        if time_out and status == 'exited':
                            has_active_visitor_assignment = False  # Visitor has left, RFID can be freed
                            break
                        
                        # If visitor is active (no time_out or status is 'active'), keep RFID unavailable
                        if status == 'active' or (time_in and not time_out):
                            has_active_visitor_assignment = True
                            break
                        
                        # Check expiry if exists (only if visitor hasn't checked out)
                        expiry_time_str = visitor_data.get('expiry_time')
                        if expiry_time_str and not time_out:
                            try:
                                expiry_dt = datetime.strptime(expiry_time_str, "%Y-%m-%d %H:%M:%S")
                                if current_time < expiry_dt:
                                    has_active_visitor_assignment = True
                                    break
                            except Exception:
                                pass
                    
                    # If no active assignments found, reset RFID to available
                    if not has_active_student_assignment and not has_active_visitor_assignment:
                        print(f"INFO: Resetting expired/orphaned RFID {rfid_id} to available")
                        rfid_ref = self.db.collection('Empty RFID').document(rfid_id)
                        rfid_ref.update({
                            'available': True,
                            'assigned_to': None,
                            'last_updated': self.get_current_timestamp()
                        })
                        cleaned_count += 1
            
            if cleaned_count > 0:
                print(f"SUCCESS: Cleaned up {cleaned_count} expired/orphaned RFID assignments")
            else:
                print("INFO: No expired RFID assignments found")
                
        except Exception as e:
            print(f"WARNING: Failed to cleanup expired RFID assignments: {e}")
    
    def load_rfid_from_firebase(self):
        """Load all RFID cards from Firebase Empty RFID collection"""
        try:
            available_rfids = []
            
            if self.firebase_initialized and self.db:
                # Clean up expired assignments first
                self.cleanup_expired_rfid_assignments()
                
                print("INFO: Loading RFID cards from Firebase Empty RFID collection...")
                # Debug the collection first
                self.debug_firebase_rfid_collection()
                
                empty_rfid_ref = self.db.collection('Empty RFID')
                docs = empty_rfid_ref.get()  # Get all documents
                
                for doc in docs:
                    rfid_data = doc.to_dict() or {}
                    rfid_id = doc.id
                    
                    # CRITICAL: Check availability from multiple sources
                    is_available = True
                    
                    # Method 1: Check Empty RFID document fields
                    if rfid_data:
                        # Check assigned_to field - if it exists and has a value, RFID is unavailable
                        if 'assigned_to' in rfid_data and rfid_data['assigned_to']:
                            is_available = False
                            print(f"🔍 DEBUG: RFID {rfid_id} marked unavailable via Empty RFID assigned_to field: {rfid_data['assigned_to']}")
                        # Also check available field - if it's explicitly False, RFID is unavailable
                        if 'available' in rfid_data and not rfid_data['available']:
                            is_available = False
                            print(f"🔍 DEBUG: RFID {rfid_id} marked unavailable via Empty RFID available field: False")
                    
                    # Method 2: Check student_rfid_assignments collection for active assignments
                    # ALWAYS check this, even if Method 1 marked it as unavailable, to ensure consistency
                    try:
                        student_assignments_ref = self.db.collection('student_rfid_assignments')
                        student_assignments = student_assignments_ref.where('rfid', '==', rfid_id).where('status', '==', 'active').get()
                        
                        if student_assignments:
                            # Check if any assignment is still valid (not expired)
                            current_time = datetime.now()
                            for assignment_doc in student_assignments:
                                assignment_data = assignment_doc.to_dict()
                                expiry_time_str = assignment_data.get('expiry_time')
                                if expiry_time_str:
                                    try:
                                        expiry_dt = datetime.strptime(expiry_time_str, "%Y-%m-%d %H:%M:%S")
                                        if current_time < expiry_dt:
                                            # Still valid assignment found
                                            is_available = False
                                            print(f"🔍 DEBUG: RFID {rfid_id} marked unavailable - active student assignment found (expires: {expiry_time_str})")
                                            break
                                    except Exception as e:
                                        print(f"⚠️ Warning: Error parsing expiry_time for {rfid_id}: {e}")
                                        # If we can't parse expiry, assume it's still active
                                        is_available = False
                                        print(f"🔍 DEBUG: RFID {rfid_id} marked unavailable - active student assignment (error parsing expiry)")
                                        break
                                else:
                                    # No expiry, assume active
                                    is_available = False
                                    print(f"🔍 DEBUG: RFID {rfid_id} marked unavailable - active student assignment (no expiry)")
                                    break
                    except Exception as e:
                        print(f"⚠️ Warning: Could not check student_rfid_assignments for {rfid_id}: {e}")
                    
                    # Method 3: Check visitors collection for active visitor assignments
                    # ALWAYS check this, even if already marked unavailable, to ensure consistency
                    try:
                        visitors_ref = self.db.collection('visitors')
                        visitors = visitors_ref.where('rfid', '==', rfid_id).get()
                        
                        found_active_visitor = False
                        for visitor_doc in visitors:
                            visitor_data = visitor_doc.to_dict()
                            status = visitor_data.get('status', '')
                            time_out = visitor_data.get('time_out')
                            
                            # Skip if visitor has exited (status is 'exited' and has time_out)
                            if status == 'exited' and time_out:
                                print(f"🔍 DEBUG: RFID {rfid_id} visitor has exited (status: {status}, time_out: {time_out}), skipping")
                                continue  # Skip this visitor, check next one
                            
                            # Check if assignment is still valid (not expired)
                            expiry_time_str = visitor_data.get('expiry_time')
                            current_time = datetime.now()
                            is_expired = False
                            
                            if expiry_time_str:
                                try:
                                    expiry_dt = datetime.strptime(expiry_time_str, "%Y-%m-%d %H:%M:%S")
                                    if current_time >= expiry_dt:
                                        # Assignment has expired, skip this visitor
                                        is_expired = True
                                        print(f"🔍 DEBUG: RFID {rfid_id} visitor assignment expired (expiry: {expiry_time_str})")
                                except Exception as e:
                                    print(f"⚠️ Warning: Error parsing visitor expiry_time for {rfid_id}: {e}")
                            
                            # Skip if expired
                            if is_expired:
                                continue  # Skip this visitor, check next one
                            
                            # RFID is unavailable if visitor is:
                            # 1. Status is 'registered' (just registered, not yet used) - this means RFID is assigned
                            # 2. Status is 'active' (currently in the building)
                            # 3. Has time_in but no time_out (active visitor who has entered)
                            # Note: We only mark as unavailable if status is explicitly 'registered' or 'active', 
                            # or if they have time_in without time_out. We don't mark as unavailable if status is 'exited' or empty.
                            if status == 'registered' or status == 'active' or (visitor_data.get('time_in') and not time_out):
                                is_available = False
                                found_active_visitor = True
                                print(f"🔍 DEBUG: RFID {rfid_id} marked unavailable - visitor assignment found (status: {status}, time_out: {time_out}, expiry: {expiry_time_str})")
                                break
                    except Exception as e:
                        print(f"⚠️ Warning: Could not check visitors collection for {rfid_id}: {e}")
                        import traceback
                        traceback.print_exc()
                    
                    # Track availability (ALWAYS add RFID to list, even if all visitors are skipped)
                    self.rfid_availability_map[rfid_id] = is_available
                    # Include both types so UI shows status
                    available_rfids.append((rfid_id, is_available))
                    if is_available:
                        print(f"✅ INFO: Available RFID: {rfid_id}")
                    else:
                        print(f"❌ INFO: Unavailable RFID: {rfid_id} - checked Empty RFID, student_rfid_assignments, and visitors collections")
                
                print(f"SUCCESS: Loaded {len(available_rfids)} RFID cards from Firebase")
            else:
                print("WARNING: Firebase not available - using empty list")
                available_rfids = []
            
            # Build display list - Show ALL RFIDs with availability labels
            # CRITICAL: Use dict to deduplicate - each RFID should appear only once
            rfid_dict = {}  # rfid_id -> is_available (ensures no duplicates)
            
            for item in available_rfids:
                if isinstance(item, tuple) and len(item) == 2:
                    rfid_id, is_available = item
                else:
                    rfid_id = item
                    is_available = True
                
                # CRITICAL: If RFID already exists, use the most restrictive availability (False takes precedence)
                if rfid_id in rfid_dict:
                    # If current is unavailable or existing is unavailable, mark as unavailable
                    if not is_available or not rfid_dict[rfid_id]:
                        rfid_dict[rfid_id] = False
                else:
                    rfid_dict[rfid_id] = is_available
            
            # Now build display list from deduplicated dict
            display_list = []
            self.rfid_display_map = {}
            for rfid_id, is_available in rfid_dict.items():
                # Show ALL RFIDs with availability status clearly labeled
                if is_available:
                    label = f"{rfid_id}(available)"  # Available: show with (available) label
                else:
                    label = f"{rfid_id}(not available)"  # Unavailable: show with (not available) label
                
                display_list.append(label)
                self.rfid_display_map[label] = rfid_id
                
                # Save availability in map (for all RFIDs, available or not)
                self.rfid_availability_map[rfid_id] = is_available

            # Update the dropdown with labeled entries if it exists
            try:
                if hasattr(self, 'rfid_dropdown') and self.rfid_dropdown:
                    self.rfid_dropdown['values'] = display_list
                if hasattr(self, 'available_rfid_var'):
                    self.available_rfid_var.set("")
            except Exception:
                # Widget not ready yet; store maps for later UI refresh
                pass
            
        except Exception as e:
            print(f"ERROR: Failed to load RFID from Firebase: {e}")
            import traceback
            traceback.print_exc()
            self.rfid_dropdown['values'] = []
            self.available_rfid_var.set("")
    
    def update_available_rfid_list(self):
        """Update the list of available empty RFID cards from Firebase"""
        try:
            available_rfids = []
            
            # Try to get RFID list from Firebase first
            if self.firebase_initialized and self.db:
                try:
                    print("INFO: Fetching available RFID list from Firebase...")
                    empty_rfid_ref = self.db.collection('Empty RFID')
                    docs = empty_rfid_ref.where('available', '==', True).get()
                    
                    for doc in docs:
                        rfid_data = doc.to_dict()
                        rfid_id = doc.id
                        if rfid_data.get('available', True):
                            available_rfids.append(rfid_id)
                    
                    print(f"SUCCESS: Found {len(available_rfids)} available RFID cards in Firebase")
                    
                except Exception as e:
                    print(f"WARNING: Failed to fetch RFID from Firebase: {e}")
                    print("INFO: Using fallback RFID list")
            
            # Fallback: Use local available RFID list if Firebase fails
            if not available_rfids:
                available_rfids = self.available_rfids.copy()
            
            # Generate additional random available RFID numbers if needed
            import random
            while len(available_rfids) < 10:
                rfid_num = f"RF{random.randint(100000, 999999)}"
                if rfid_num not in available_rfids:
                    available_rfids.append(rfid_num)
            
            # Build display list and mapping
            display_list = []
            self.rfid_display_map = {}
            for item in available_rfids:
                if isinstance(item, tuple) and len(item) == 2:
                    rfid_id, is_available = item
                else:
                    rfid_id = item
                    is_available = True
                label = f"{rfid_id} ({'available' if is_available else 'unavailable'})"
                display_list.append(label)
                self.rfid_display_map[label] = rfid_id
                self.rfid_availability_map[rfid_id] = is_available

            # Update the dropdown with labeled entries (if exists)
            try:
                if hasattr(self, 'rfid_dropdown') and self.rfid_dropdown:
                    self.rfid_dropdown['values'] = display_list
                if display_list and hasattr(self, 'available_rfid_var'):
                    self.available_rfid_var.set(display_list[0])  # Select first one by default
            except Exception:
                pass
            
        except Exception as e:
            print(f"ERROR: Error updating RFID list: {e}")
            # Fallback to empty list
            self.rfid_dropdown['values'] = []
            self.available_rfid_var.set("")
    
    def update_rfid_availability_in_firebase(self, rfid_id, available=False, assigned_to=None):
        """Update RFID availability status in Firebase
        
        Args:
            rfid_id: The RFID card ID
            available: True if available, False if unavailable
            assigned_to: 'student', 'visitor', or None (if available)
        """
        try:
            if self.firebase_initialized and self.db:
                # Update the RFID document in Firebase
                rfid_ref = self.db.collection('Empty RFID').document(rfid_id)
                
                update_data = {
                    'available': available,
                    'last_updated': firestore.SERVER_TIMESTAMP
                }
                
                # Set assigned_to based on parameter or availability
                if available:
                    # If available, clear assigned_to
                    update_data['assigned_to'] = None
                elif assigned_to:
                    # If assigned, set assigned_to to the specified value
                    update_data['assigned_to'] = assigned_to
                else:
                    # If not specified but unavailable, check existing assignment
                    doc = rfid_ref.get()
                    if doc.exists:
                        existing_data = doc.to_dict()
                        if existing_data and 'assigned_to' in existing_data:
                            update_data['assigned_to'] = existing_data['assigned_to']
                        else:
                            update_data['assigned_to'] = 'unknown'  # Fallback
                    else:
                        # Document doesn't exist and we're marking as unavailable, set default
                        update_data['assigned_to'] = 'unknown'
                
                # Use set() with merge=True instead of update() so it works even if document doesn't exist
                # This ensures the document is created if it doesn't exist
                rfid_ref.set(update_data, merge=True)
                
                # Verify the update was successful by reading it back
                verify_doc = rfid_ref.get()
                if verify_doc.exists:
                    verify_data = verify_doc.to_dict()
                    print(f"SUCCESS: RFID {rfid_id} availability updated in Firebase: available={verify_data.get('available')}, assigned_to={verify_data.get('assigned_to')}")
                else:
                    print(f"⚠️ WARNING: RFID {rfid_id} document was not found after update attempt")
                return True
            else:
                print("WARNING: Firebase not available - RFID availability not updated")
                return False
        except Exception as e:
            print(f"ERROR: Failed to update RFID availability: {e}")
            return False
    
    def on_visitor_id_type_change(self, event=None):
        """Handle ID type dropdown selection change for visitor"""
        try:
            selected = self.id_type_var.get()
            if selected == "Other":
                # Show custom ID input
                self.custom_id_frame.pack(fill=tk.X, pady=(10, 0))
            else:
                # Hide custom ID input
                self.custom_id_frame.pack_forget()
                self.custom_id_entry.delete(0, tk.END)
        except Exception as e:
            print(f"Error handling ID type change: {e}")
    
    def on_id_type_change(self):
        """Handle ID type selection change"""
        try:
            if self.id_type_var.get() == "other":
                # Show custom ID input field
                self.custom_id_frame.pack(fill=tk.X, pady=(10, 0))
            else:
                # Hide custom ID input field
                self.custom_id_frame.pack_forget()
        except Exception as e:
            print(f"Error handling ID type change: {e}")
    
    def register_visitor(self, rfid=None):
        """Register visitor and assign RFID
        
        Args:
            rfid: Optional RFID to assign. If provided, uses this directly.
                  If None, tries to read from dropdown (for backward compatibility)
        """
        try:
            # Get form data - build name from separated fields
            surname = self.visitor_surname_entry.get().strip()
            firstname = self.visitor_firstname_entry.get().strip()
            mi = self.visitor_mi_entry.get().strip()
            suffix = self.visitor_suffix_entry.get().strip()
            
            # Build full name from components
            name_parts = []
            if surname:
                name_parts.append(surname)
            if firstname:
                name_parts.append(firstname)
            if mi:
                name_parts.append(mi + ".")
            if suffix:
                name_parts.append(suffix)
            
            visitor_name = " ".join(name_parts) if name_parts else ""
            
            visitor_purpose = self.visitor_purpose_entry.get().strip()
            id_type_display = self.id_type_var.get()
            
            # Convert display ID type to internal format
            id_type_map = {
                "Driver's License": "driver_license",
                "National ID": "national_id",
                "Passport": "passport",
                "Other": "other"
            }
            # Handle empty string as not_specified
            if not id_type_display or id_type_display == "":
                id_type = "not_specified"
            else:
                id_type = id_type_map.get(id_type_display, id_type_display.lower().replace(" ", "_"))
            
            # Get RFID
            if rfid:
                selected_rfid = rfid
            else:
                # Try to get from dropdown if it exists
                if hasattr(self, 'available_rfid_var'):
                    selected_rfid_display = self.available_rfid_var.get()
                    # Map display label back to raw RFID if necessary
                    selected_rfid = self.rfid_display_map.get(selected_rfid_display, selected_rfid_display) if hasattr(self, 'rfid_display_map') else selected_rfid_display
                else:
                    messagebox.showerror("Error", "No RFID provided. Please tap an empty RFID card.")
                    # Reset assignment mode
                    self.visitor_rfid_assignment_active = False
                    self.reset_visitor_rfid_tap_button()
                    return
            
            # CRITICAL: Check if selected RFID is unavailable (from dropdown selection handler)
            # This is a double-check in case the selection handler didn't catch it
            if selected_rfid in self.rfid_availability_map and not self.rfid_availability_map.get(selected_rfid, True):
                messagebox.showerror("Unavailable RFID", f"RFID {selected_rfid} is currently unavailable.\n\nThis RFID is already assigned to a student or visitor.\n\nPlease return it first or choose an available RFID.")
                # Reset assignment mode
                self.visitor_rfid_assignment_active = False
                self.reset_visitor_rfid_tap_button()
                return
            
            # Handle custom ID type
            if id_type == "other" or id_type_display == "Other":
                custom_id_type = self.custom_id_entry.get().strip()
                if not custom_id_type:
                    messagebox.showerror("Error", "Please specify the ID type.")
                    # Reset assignment mode
                    self.visitor_rfid_assignment_active = False
                    self.reset_visitor_rfid_tap_button()
                    return
                id_type = custom_id_type.lower().replace(" ", "_")
            
            # Validate required fields
            if not surname or not firstname:
                messagebox.showerror("Error", "Please enter visitor's surname and first name.")
                # Reset assignment mode
                self.visitor_rfid_assignment_active = False
                self.reset_visitor_rfid_tap_button()
                return
            
            if not visitor_purpose:
                visitor_purpose = "General Visit"
            
            if not selected_rfid:
                messagebox.showerror("Error", "Please select an available RFID card.")
                # Reset assignment mode
                self.visitor_rfid_assignment_active = False
                self.reset_visitor_rfid_tap_button()
                return
            
            # CRITICAL: Check if selected RFID is unavailable (from dropdown selection handler)
            # This is a double-check in case the selection handler didn't catch it
            raw_rfid = self.rfid_display_map.get(selected_rfid, selected_rfid)
            if raw_rfid in self.rfid_availability_map and not self.rfid_availability_map.get(raw_rfid, True):
                messagebox.showerror("Unavailable RFID", f"RFID {raw_rfid} is currently unavailable.\n\nThis RFID is already assigned to a student or visitor.\n\nPlease return it first or choose an available RFID.")
                # Reset assignment mode
                self.visitor_rfid_assignment_active = False
                self.reset_visitor_rfid_tap_button()
                return
            
            # CRITICAL: Check Firebase for current availability status (not just local memory)
            is_available_in_firebase = True
            if self.firebase_initialized and self.db:
                try:
                    rfid_ref = self.db.collection('Empty RFID').document(selected_rfid)
                    rfid_doc = rfid_ref.get()
                    if rfid_doc.exists:
                        rfid_data = rfid_doc.to_dict() or {}
                        # Check if RFID is assigned or unavailable
                        if 'assigned_to' in rfid_data and rfid_data['assigned_to']:
                            is_available_in_firebase = False
                            assigned_to = rfid_data.get('assigned_to', 'unknown')
                            messagebox.showerror("Error", f"RFID {selected_rfid} is already assigned to a {assigned_to}. Choose another card or return it first.")
                            # Reset assignment mode
                            self.visitor_rfid_assignment_active = False
                            self.reset_visitor_rfid_tap_button()
                            return
                        elif 'available' in rfid_data and not rfid_data['available']:
                            is_available_in_firebase = False
                            messagebox.showerror("Error", f"RFID {selected_rfid} is currently unavailable. Choose another card or return/clear it first.")
                            # Reset assignment mode
                            self.visitor_rfid_assignment_active = False
                            self.reset_visitor_rfid_tap_button()
                            return
                except Exception as e:
                    print(f"⚠️ Warning: Could not check Firebase for RFID availability: {e}")
            
            # Re-check local availability map
            if selected_rfid in self.rfid_availability_map and not self.rfid_availability_map.get(selected_rfid, True):
                messagebox.showerror("Error", f"RFID {selected_rfid} is currently unavailable. Choose another card or return/clear it first.")
                # Reset assignment mode
                self.visitor_rfid_assignment_active = False
                self.reset_visitor_rfid_tap_button()
                return
            
            # Check if RFID is already assigned to a visitor (local memory)
            if selected_rfid in self.visitor_rfid_registry:
                messagebox.showerror("Error", f"RFID {selected_rfid} is already assigned to another visitor.")
                # Reset assignment mode
                self.visitor_rfid_assignment_active = False
                self.reset_visitor_rfid_tap_button()
                return
            
            # Check if RFID is already assigned to a student (local memory)
            if hasattr(self, 'student_rfid_assignments') and selected_rfid in self.student_rfid_assignments:
                messagebox.showerror("Error", f"RFID {selected_rfid} is already assigned to a student. Choose another card or return it first.")
                # Reset assignment mode
                self.visitor_rfid_assignment_active = False
                self.reset_visitor_rfid_tap_button()
                return
            
            # Generate visitor ID
            import random
            visitor_id = f"VIS{random.randint(1000, 9999)}"
            
            # Get companion count
            companion_count = self.companion_count_var.get() or "0"
            
            # Create visitor info with temporary assignment (24 hours)
            expiry_time = (datetime.now() + timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")
            visitor_info = {
                'visitor_id': visitor_id,
                'name': visitor_name,
                'purpose': visitor_purpose,
                'id_type': id_type,
                'rfid': selected_rfid,
                'companion_count': companion_count,
                'registration_time': self.get_current_timestamp(),
                'expiry_time': expiry_time,
                'status': 'registered'  # registered, active, exited
            }

            # Link visitor to RFID (in-memory registry)
            self.visitor_rfid_registry[selected_rfid] = visitor_info
            # Also track assignment specially so we can enforce 24h expiry similar to student temporary RFIDs
            if not hasattr(self, 'visitor_rfid_assignments'):
                self.visitor_rfid_assignments = {}
            self.visitor_rfid_assignments[selected_rfid] = visitor_info
            # Mark as unavailable so students can't use it
            try:
                self.rfid_availability_map[selected_rfid] = False
            except Exception:
                pass
            try:
                self.student_rfid_availability_map[selected_rfid] = False
            except Exception:
                pass
            # Remove from available pool
            if selected_rfid in self.available_rfids:
                try:
                    self.available_rfids.remove(selected_rfid)
                except Exception:
                    pass
            
            # Add to activity log
            self.add_activity_log(f"Visitor registered: {visitor_name} - {visitor_purpose} (ID: {id_type}) - Assigned RFID: {selected_rfid}")
            
            # Save to Firebase (includes expiry)
            self.save_visitor_to_firebase(visitor_info)
            
            # Update RFID availability in Firebase with assigned_to='visitor'
            try:
                self.update_rfid_availability_in_firebase(selected_rfid, available=False, assigned_to='visitor')
            except Exception as e:
                print(f"⚠️ Warning: Could not update Firebase RFID availability: {e}")
            
            # Show confirmation
            messagebox.showinfo("Visitor Registered Successfully", 
                              f"Visitor Details:\n"
                              f"Name: {visitor_name}\n"
                              f"Purpose: {visitor_purpose}\n"
                              f"ID Type: {id_type.replace('_', ' ').title()}\n"
                              f"Visitor ID: {visitor_id}\n"
                              f"Assigned RFID: {selected_rfid}\n\n"
                              f"The visitor can now use the RFID card for entry and exit.")
            
            # Refresh RFID history table if it exists (tab is open)
            try:
                if hasattr(self, 'rfid_history_tree') and self.rfid_history_tree is not None:
                    # Small delay to ensure Firebase write is complete
                    self.root.after(500, self.refresh_rfid_history)
                    print(f"✅ Scheduled RFID history refresh after visitor registration")
            except Exception as e:
                print(f"⚠️ Warning: Could not refresh RFID history: {e}")
            
            # Clear form
            self.visitor_surname_entry.delete(0, tk.END)
            self.visitor_firstname_entry.delete(0, tk.END)
            self.visitor_mi_entry.delete(0, tk.END)
            self.visitor_suffix_entry.delete(0, tk.END)
            self.visitor_purpose_entry.delete(0, tk.END)
            self.id_type_var.set("")
            self.id_type_dropdown.set("")
            self.custom_id_entry.delete(0, tk.END)
            self.custom_id_frame.pack_forget()  # Hide custom ID field
            self.companion_count_var.set("0")
            self.companion_count_dropdown.set("0")
            
            # Clear and reset visitor RFID assignment
            if hasattr(self, 'visitor_rfid_entry_var'):
                self.visitor_rfid_entry_var.set("")
            self.visitor_rfid_assignment_active = False
            self.reset_visitor_rfid_tap_button()
            
            # CRITICAL: Refresh BOTH dropdowns to reflect unavailable state
            # Visitor dropdown
            try:
                self.load_rfid_from_firebase()
                print(f"✅ Refreshed visitor RFID dropdown after assignment")
            except Exception as e:
                print(f"⚠️ Warning: Could not refresh visitor RFID dropdown: {e}")
            
            # Student dropdown (forgot ID) - also refresh to show RFID as unavailable
            try:
                self.update_student_rfid_list()
                print(f"✅ Refreshed student RFID dropdown after visitor assignment")
            except Exception as e:
                print(f"⚠️ Warning: Could not refresh student RFID dropdown: {e}")
            
            # Return to dashboard
            self.back_to_dashboard()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to register visitor: {e}")
            import traceback
            traceback.print_exc()
    
    def activate_visitor_rfid_tap_mode(self):
        """Activate visitor RFID tap mode - show popup dialog for RFID entry (using simpledialog like Return RFID)"""
        try:
            # Set assignment mode flag
            self.visitor_rfid_assignment_active = True
            
            # Disable the button to prevent multiple clicks
            if hasattr(self, 'visitor_tap_rfid_btn'):
                self.visitor_tap_rfid_btn.config(state=tk.DISABLED, text="⏳ Waiting for RFID tap...")
            
            # Use simpledialog.askstring like the Return RFID dialog
            rfid = simpledialog.askstring(
                "Tap Empty RFID",
                "Enter RFID to assign (e.g. 0095129433)\n\nPlace an empty RFID card on the reader:"
            )
            
            # If user cancelled or closed dialog
            if not rfid:
                self.visitor_rfid_assignment_active = False
                self.reset_visitor_rfid_tap_button()
                return
            
            rfid = rfid.strip()
            if not rfid:
                self.visitor_rfid_assignment_active = False
                self.reset_visitor_rfid_tap_button()
                return
            
            # Process the RFID input
            self.visitor_rfid_entry_var.set(rfid)
            self.process_visitor_rfid_tap_entry()
            
        except Exception as e:
            print(f"ERROR: Error activating visitor RFID tap mode: {e}")
            import traceback
            traceback.print_exc()
            self.visitor_rfid_assignment_active = False
            self.reset_visitor_rfid_tap_button()
    
    def process_visitor_rfid_tap_entry(self):
        """Process RFID from the simpledialog entry field"""
        try:
            rfid = self.visitor_rfid_entry_var.get().strip()
            if not rfid:
                return
            
            print(f"🔍 Visitor RFID detected from dialog entry field: {rfid}")
            
            # Check if we're in visitor RFID assignment mode
            if hasattr(self, 'visitor_rfid_assignment_active') and self.visitor_rfid_assignment_active:
                # Check if tapped RFID is an available empty RFID
                if not self.is_empty_rfid_available(rfid):
                    print(f"⚠️ Tapped RFID {rfid} is not an available empty RFID")
                    messagebox.showerror(
                        "RFID Not Available",
                        f"RFID {rfid} is already assigned to a student or visitor.\n\nPlease return it first or choose an available RFID."
                    )
                    # Clear assignment mode and reset button
                    self.visitor_rfid_assignment_active = False
                    self.reset_visitor_rfid_tap_button()
                    return
                
                # All checks passed - proceed with registration
                print(f"✅ Empty RFID detected in visitor assignment mode: {rfid}")
                # Automatically register visitor with this RFID
                self.register_visitor(rfid=rfid)
            else:
                # Not in assignment mode - just clear
                self.visitor_rfid_entry_var.set("")
                    
        except Exception as e:
            print(f"ERROR: Error processing visitor RFID tap entry: {e}")
            import traceback
            traceback.print_exc()
    
    def reset_visitor_rfid_tap_button(self):
        """Reset the visitor RFID tap button to initial state"""
        try:
            if hasattr(self, 'visitor_tap_rfid_btn'):
                self.visitor_tap_rfid_btn.config(state=tk.NORMAL, text="📱 Tap Empty RFID")
            # Clear the entry field variable
            if hasattr(self, 'visitor_rfid_entry_var'):
                self.visitor_rfid_entry_var.set("")
            # Show status label again
            if hasattr(self, 'visitor_rfid_tap_status_label'):
                self.visitor_rfid_tap_status_label.config(
                    text="Click the button above, then tap an empty RFID card",
                    fg='#6b7280',
                    font=('Arial', 12)
                )
                self.visitor_rfid_tap_status_label.pack()
        except Exception as e:
            print(f"⚠️ Warning: Could not reset visitor RFID tap button: {e}")
    
    def get_current_timestamp(self):
        """Get current timestamp in readable format"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def format_document_id(self, timestamp):
        """Convert timestamp to safe document ID format for Firebase
        Converts 'YYYY-MM-DD HH:MM:SS' to 'YYYY-MM-DD_HH-MM-SS'
        """
        # Replace space with underscore, colons with hyphens
        formatted = timestamp.replace(' ', '_').replace(':', '-')
        return formatted
    
    def save_visitor_to_firebase(self, visitor_info):
        """Save visitor information to Firebase"""
        try:
            if self.firebase_initialized and self.db:
                # Save visitor registration
                visitor_ref = self.db.collection('visitors').document(visitor_info['visitor_id'])
                visitor_data = {
                    'visitor_id': visitor_info['visitor_id'],
                    'name': visitor_info['name'],
                    'purpose': visitor_info['purpose'],
                    'id_type': visitor_info['id_type'],
                    'rfid': visitor_info['rfid'],
                    'registration_time': visitor_info['registration_time'],
                    'status': visitor_info['status']
                }
                # Add expiry_time if it exists in visitor_info
                if 'expiry_time' in visitor_info:
                    visitor_data['expiry_time'] = visitor_info['expiry_time']
                # Add companion_count if it exists in visitor_info
                if 'companion_count' in visitor_info:
                    visitor_data['companion_count'] = visitor_info['companion_count']
                visitor_ref.set(visitor_data)
                print(f"SUCCESS: Visitor {visitor_info['name']} saved to Firebase")
            else:
                print("WARNING: Firebase not available - visitor data not saved to cloud")
        except Exception as e:
            print(f"ERROR: Failed to save visitor to Firebase: {e}")

    def return_temp_visitor_rfid_dialog(self):
        """Show a dialog to let guard return/unassign a temporary visitor RFID immediately"""
        try:
            # Build list of currently assigned temporary RFIDs for visitors
            assigned = []
            if hasattr(self, 'visitor_rfid_assignments') and isinstance(self.visitor_rfid_assignments, dict):
                for rfid, info in self.visitor_rfid_assignments.items():
                    assigned.append(f"{rfid} - {info.get('name', 'Unknown')}")

            if not assigned:
                messagebox.showinfo("No Assigned RFIDs", "There are no temporary visitor RFIDs currently assigned.")
                return

            choice = simpledialog.askstring("Return RFID", "Enter RFID to return (e.g. 0095129433)\nAvailable assigned:\n" + "\n".join(assigned))
            if not choice:
                return
            choice = choice.strip()

            if choice in self.visitor_rfid_assignments:
                try:
                    del self.visitor_rfid_assignments[choice]
                except Exception:
                    pass

                # Also remove from registry
                try:
                    if choice in self.visitor_rfid_registry:
                        del self.visitor_rfid_registry[choice]
                except Exception:
                    pass

                # Mark available in local list and Firebase
                if choice not in self.available_rfids:
                    self.available_rfids.append(choice)
                try:
                    self.update_rfid_availability_in_firebase(choice, available=True)
                except Exception:
                    pass

                # Refresh available RFID list in visitor UI
                try:
                    self.load_rfid_from_firebase()
                except Exception:
                    pass

                self.add_activity_log(f"Temporary visitor RFID returned manually: {choice}")
                messagebox.showinfo("RFID Returned", f"Temporary visitor RFID {choice} has been returned and is now available.")
            else:
                messagebox.showwarning("Not Found", f"RFID {choice} is not a currently assigned temporary visitor RFID.")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to return temporary visitor RFID: {e}")

    def return_assigned_rfid_dialog(self):
        """Unified dialog to return any assigned temporary RFID (student or visitor)."""
        try:
            # CRITICAL: Load assigned RFIDs from Firebase first (in case UI was restarted)
            self.load_assigned_rfids_from_firebase()
            
            # Build list of assigned RFIDs from both student and visitor assignments
            assigned_list = []
            assigned_map = {}  # rfid -> (type, display)

            if hasattr(self, 'student_rfid_assignments') and isinstance(self.student_rfid_assignments, dict):
                for rfid, info in self.student_rfid_assignments.items():
                    display = f"{rfid} - Student: {info.get('name', 'Unknown')}"
                    assigned_list.append(display)
                    assigned_map[rfid] = ('student', display)

            if hasattr(self, 'visitor_rfid_assignments') and isinstance(self.visitor_rfid_assignments, dict):
                for rfid, info in self.visitor_rfid_assignments.items():
                    display = f"{rfid} - Visitor: {info.get('name', 'Unknown')}"
                    assigned_list.append(display)
                    assigned_map[rfid] = ('visitor', display)
            
            # Also check Firebase directly for unavailable RFIDs that might not be in memory
            if self.firebase_initialized and self.db:
                try:
                    # Method 1: Check student_rfid_assignments collection directly
                    student_assignments_ref = self.db.collection('student_rfid_assignments')
                    all_student_assignments = student_assignments_ref.get()
                    
                    for assignment_doc in all_student_assignments:
                        assignment_data = assignment_doc.to_dict()
                        rfid = assignment_data.get('rfid')
                        status = assignment_data.get('status', 'active')
                        
                        if rfid and status != 'returned' and rfid not in assigned_map:
                            # Check if still valid (not expired)
                            expiry_time_str = assignment_data.get('expiry_time')
                            is_valid = True
                            if expiry_time_str:
                                try:
                                    expiry_dt = datetime.strptime(expiry_time_str, "%Y-%m-%d %H:%M:%S")
                                    if datetime.now() >= expiry_dt:
                                        is_valid = False
                                except Exception:
                                    pass
                            
                            if is_valid:
                                display = f"{rfid} - Student: {assignment_data.get('name', 'Unknown')} (from Firebase)"
                                assigned_list.append(display)
                                assigned_map[rfid] = ('student', display)
                                print(f"🔍 DEBUG: Found student RFID assignment in Firebase: {rfid}")
                    
                    # Method 2: Check visitors collection directly
                    visitors_ref = self.db.collection('visitors')
                    all_visitors = visitors_ref.get()
                    
                    for visitor_doc in all_visitors:
                        visitor_data = visitor_doc.to_dict()
                        rfid = visitor_data.get('rfid')
                        status = visitor_data.get('status', '')
                        time_out = visitor_data.get('time_out')
                        
                        if rfid and rfid not in assigned_map:
                            # Check if visitor is still active or registered
                            # Include 'registered' status (just registered, not yet used) and 'active' status
                            if status == 'registered' or status == 'active' or (visitor_data.get('time_in') and not time_out):
                                # Also check expiry if present
                                expiry_time_str = visitor_data.get('expiry_time')
                                is_valid = True
                                if expiry_time_str:
                                    try:
                                        expiry_dt = datetime.strptime(expiry_time_str, "%Y-%m-%d %H:%M:%S")
                                        if datetime.now() >= expiry_dt:
                                            is_valid = False
                                    except Exception:
                                        pass
                                
                                if is_valid:
                                    display = f"{rfid} - Visitor: {visitor_data.get('name', 'Unknown')} (from Firebase, status: {status})"
                                    assigned_list.append(display)
                                    assigned_map[rfid] = ('visitor', display)
                                    print(f"🔍 DEBUG: Found visitor RFID assignment in Firebase: {rfid} (status: {status})")
                                    break
                    
                    # Method 3: Check Empty RFID collection for unavailable RFIDs (as fallback)
                    # Also cross-check with actual assignments in case Empty RFID document is stale
                    empty_rfid_ref = self.db.collection('Empty RFID')
                    docs = empty_rfid_ref.get()
                    for doc in docs:
                        rfid_id = doc.id
                        rfid_data = doc.to_dict() or {}
                        
                        # Skip if already in assigned_map
                        if rfid_id in assigned_map:
                            continue
                        
                        # Check if RFID is unavailable based on Empty RFID document
                        is_unavailable = False
                        assigned_to = None
                        
                        if 'assigned_to' in rfid_data and rfid_data['assigned_to']:
                            is_unavailable = True
                            assigned_to = rfid_data.get('assigned_to')
                        elif 'available' in rfid_data and not rfid_data['available']:
                            is_unavailable = True
                            assigned_to = rfid_data.get('assigned_to', 'unknown')
                        
                        # If not marked in Empty RFID, check actual assignments as cross-reference
                        if not is_unavailable:
                            # Check if there's an active student assignment
                            student_check = student_assignments_ref.where('rfid', '==', rfid_id).where('status', '==', 'active').get()
                            if student_check:
                                is_unavailable = True
                                assigned_to = 'student'
                                for sd in student_check:
                                    name = sd.to_dict().get('name', 'Unknown')
                                    display = f"{rfid_id} - Student: {name} (from Empty RFID cross-check)"
                                    assigned_list.append(display)
                                    assigned_map[rfid_id] = ('student', display)
                                    print(f"🔍 DEBUG: Found unavailable RFID via cross-check: {rfid_id} (student)")
                                    break
                            
                            # Check if there's an active visitor assignment
                            if not is_unavailable:
                                visitor_check = visitors_ref.where('rfid', '==', rfid_id).get()
                                for vd in visitor_check:
                                    vdata = vd.to_dict()
                                    vstatus = vdata.get('status', '')
                                    if vstatus == 'registered' or vstatus == 'active' or (vdata.get('time_in') and not vdata.get('time_out')):
                                        is_unavailable = True
                                        assigned_to = 'visitor'
                                        name = vdata.get('name', 'Unknown')
                                        display = f"{rfid_id} - Visitor: {name} (from Empty RFID cross-check, status: {vstatus})"
                                        assigned_list.append(display)
                                        assigned_map[rfid_id] = ('visitor', display)
                                        print(f"🔍 DEBUG: Found unavailable RFID via cross-check: {rfid_id} (visitor, status: {vstatus})")
                                        break
                        
                        # If unavailable and not already added, add it
                        if is_unavailable and rfid_id not in assigned_map and assigned_to:
                            if assigned_to == 'student' or assigned_to == 'visitor':
                                display = f"{rfid_id} - {assigned_to.title()}: (from Empty RFID collection)"
                                assigned_list.append(display)
                                assigned_map[rfid_id] = (assigned_to, display)
                                print(f"🔍 DEBUG: Found unavailable RFID in Empty RFID collection: {rfid_id}")
                except Exception as e:
                    print(f"⚠️ Warning: Could not check Firebase for unavailable RFIDs: {e}")
                    import traceback
                    traceback.print_exc()

            if not assigned_list:
                messagebox.showinfo("No Assigned RFIDs", "There are no temporary RFIDs currently assigned.")
                return

            # Ask guard to enter the RFID (keeps it simple and consistent with existing dialogs)
            choice = simpledialog.askstring("Return RFID", "Enter RFID to return (e.g. 0095129433)\nAvailable assigned:\n" + "\n".join(assigned_list))
            if not choice:
                return
            choice = choice.strip()

            # Decide which handler to call - check both memory and Firebase
            # Check if it's a student or visitor assignment (from memory or Firebase)
            assignment_type = assigned_map.get(choice, ('', ''))[0]
            is_student = choice in getattr(self, 'student_rfid_assignments', {}) or assignment_type == 'student'
            is_visitor = choice in getattr(self, 'visitor_rfid_assignments', {}) or assignment_type == 'visitor'
            
            # If not found in assigned_map, try to query Firebase directly for this specific RFID
            if not is_student and not is_visitor:
                print(f"🔍 DEBUG: RFID {choice} not found in assigned_map, querying Firebase directly...")
                if self.firebase_initialized and self.db:
                    try:
                        # Check student_rfid_assignments
                        student_assignments = self.db.collection('student_rfid_assignments').where('rfid', '==', choice).get()
                        if student_assignments:
                            for assignment_doc in student_assignments:
                                assignment_data = assignment_doc.to_dict()
                                status = assignment_data.get('status', 'active')
                                if status != 'returned':
                                    is_student = True
                                    print(f"✅ Found student assignment for RFID {choice} in Firebase")
                                    break
                        
                        # Check visitors collection
                        if not is_student:
                            visitors = self.db.collection('visitors').where('rfid', '==', choice).get()
                            if visitors:
                                for visitor_doc in visitors:
                                    visitor_data = visitor_doc.to_dict()
                                    status = visitor_data.get('status', '')
                                    time_out = visitor_data.get('time_out')
                                    # Check for 'registered' status (just assigned) or 'active' status (in building)
                                    # or has time_in without time_out (active visitor)
                                    if status == 'registered' or status == 'active' or (visitor_data.get('time_in') and not time_out):
                                        is_visitor = True
                                        print(f"✅ Found visitor assignment for RFID {choice} in Firebase (status: {status})")
                                        break
                    except Exception as e:
                        print(f"⚠️ Warning: Could not query Firebase for RFID {choice}: {e}")
            
            if is_student:
                # forward to student return logic
                try:
                    # Remove from memory if exists
                    if choice in getattr(self, 'student_rfid_assignments', {}):
                        del self.student_rfid_assignments[choice]
                    
                    # Also update Firebase assignment status to 'returned'
                    if self.firebase_initialized and self.db:
                        try:
                            assignments_ref = self.db.collection('student_rfid_assignments')
                            assignments = assignments_ref.where('rfid', '==', choice).where('status', '==', 'active').get()
                            for assignment_doc in assignments:
                                assignment_doc.reference.update({'status': 'returned'})
                        except Exception:
                            pass
                    
                    # mark available and refresh BOTH dropdowns
                    if choice not in self.available_rfids:
                        self.available_rfids.append(choice)
                    try:
                        self.update_rfid_availability_in_firebase(choice, available=True)
                    except Exception:
                        pass
                    
                    # CRITICAL: Refresh BOTH dropdowns after return
                    try:
                        self.update_student_rfid_list()
                        print(f"✅ Refreshed student RFID dropdown after return")
                    except Exception:
                        pass
                    try:
                        self.load_rfid_from_firebase()
                        print(f"✅ Refreshed visitor RFID dropdown after student return")
                    except Exception:
                        pass
                    
                    self.add_activity_log(f"Temporary student RFID returned manually: {choice}")
                    messagebox.showinfo("RFID Returned", f"Temporary student RFID {choice} has been returned and is now available.")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to return student RFID: {e}")
                return

            if is_visitor:
                try:
                    # Remove from memory if exists
                    if choice in getattr(self, 'visitor_rfid_assignments', {}):
                        del self.visitor_rfid_assignments[choice]
                    if choice in getattr(self, 'visitor_rfid_registry', {}):
                        del self.visitor_rfid_registry[choice]
                    
                    # Also update Firebase visitor status
                    if self.firebase_initialized and self.db:
                        try:
                            visitors_ref = self.db.collection('visitors')
                            # Check for both 'registered' and 'active' status visitors with this RFID
                            visitors = visitors_ref.where('rfid', '==', choice).get()
                            for visitor_doc in visitors:
                                visitor_data = visitor_doc.to_dict()
                                status = visitor_data.get('status', '')
                                # Only update if visitor is 'registered' or 'active' (not 'returned')
                                if status == 'registered' or status == 'active':
                                    # Mark as returned (keep RFID for history tracking)
                                    visitor_doc.reference.update({
                                        'status': 'returned',  # Mark visitor as returned
                                        'time_out': self.get_current_timestamp() if not visitor_data.get('time_out') else visitor_data.get('time_out')
                                        # Note: We keep 'rfid' field for history tracking - RFID availability is managed separately
                                    })
                                    print(f"✅ Updated visitor record: cleared RFID {choice} and marked as returned")
                        except Exception as e:
                            print(f"⚠️ Warning: Could not update visitor record in Firebase: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    if choice not in self.available_rfids:
                        self.available_rfids.append(choice)
                    try:
                        self.update_rfid_availability_in_firebase(choice, available=True)
                    except Exception:
                        pass
                    
                    # CRITICAL: Refresh BOTH dropdowns after return
                    try:
                        self.load_rfid_from_firebase()
                        print(f"✅ Refreshed visitor RFID dropdown after return")
                    except Exception:
                        pass
                    try:
                        self.update_student_rfid_list()
                        print(f"✅ Refreshed student RFID dropdown after visitor return")
                    except Exception:
                        pass
                    
                    self.add_activity_log(f"Temporary visitor RFID returned manually: {choice}")
                    messagebox.showinfo("RFID Returned", f"Temporary visitor RFID {choice} has been returned and is now available.")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to return visitor RFID: {e}")
                return

            # If not found in either, warn
            messagebox.showwarning("Not Found", f"RFID {choice} is not a currently assigned temporary RFID.")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to return assigned RFID: {e}")

    def handle_rfid_history_tab(self):
        """Handle RFID history tab - create and switch to RFID history tab"""
        try:
            # Create RFID history tab
            self.create_rfid_history_tab()
            
            # Switch to RFID history tab
            self.notebook.select(self.rfid_history_tab)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open RFID history: {e}")
            import traceback
            traceback.print_exc()
    
    def create_rfid_history_tab(self):
        """Create RFID history tab"""
        if self.rfid_history_tab is None:
            self.rfid_history_tab = tk.Frame(self.notebook, bg='#ffffff')
            self.notebook.add(self.rfid_history_tab, text="📋 RFID History")
            
            # Create RFID history content
            self.create_rfid_history_content(self.rfid_history_tab)
    
    def create_rfid_history_content(self, parent):
        """Create RFID history content in tab format"""
        try:
            # Main container
            main_frame = tk.Frame(parent, bg='#ffffff')
            main_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=30)
            
            # Header with back button
            header_frame = tk.Frame(main_frame, bg='#ffffff')
            header_frame.pack(fill=tk.X, pady=(0, 30))
            
            # Back button at top
            back_btn = tk.Button(
                header_frame,
                text="← Back to Dashboard",
                command=self.back_to_dashboard,
                font=('Arial', 12, 'bold'),
                bg='#6b7280',
                fg='white',
                relief='raised',
                bd=3,
                padx=20,
                pady=8,
                cursor='hand2',
                activebackground='#4b5563',
                activeforeground='white'
            )
            back_btn.pack(anchor=tk.W, pady=(0, 15))
            
            # Title
            title_label = tk.Label(
                header_frame,
                text="📋 RFID Usage History",
                font=('Arial', 24, 'bold'),
                fg='#1e3a8a',
                bg='#ffffff'
            )
            title_label.pack()
            
            subtitle_label = tk.Label(
                header_frame,
                text="View and manage RFID card assignments",
                font=('Arial', 14),
                fg='#6b7280',
                bg='#ffffff'
            )
            subtitle_label.pack(pady=(5, 0))
            
            # Create frame for table with scrollbars
            table_frame = tk.Frame(main_frame, bg='#ffffff')
            table_frame.pack(fill=tk.BOTH, expand=True, pady=(20, 0))
            
            # Create Treeview for table
            columns = ('Name', 'Time in', 'Time out', 'Status', 'Companion', 'ID type', 'Assigned RFID')
            tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=20)
            
            # Configure column headings
            tree.heading('Name', text='Name')
            tree.heading('Time in', text='Time in')
            tree.heading('Time out', text='Time out')
            tree.heading('Status', text='Status')
            tree.heading('Companion', text='Companion')
            tree.heading('ID type', text='ID type')
            tree.heading('Assigned RFID', text='Assigned RFID')
            
            # Configure column widths
            tree.column('Name', width=200, anchor='w')
            tree.column('Time in', width=100, anchor='center')
            tree.column('Time out', width=100, anchor='center')
            tree.column('Status', width=100, anchor='center')
            tree.column('Companion', width=100, anchor='center')
            tree.column('ID type', width=120, anchor='center')
            tree.column('Assigned RFID', width=150, anchor='center')
            
            # Scrollbars
            v_scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
            h_scrollbar = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=tree.xview)
            tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
            
            # Grid layout
            tree.grid(row=0, column=0, sticky='nsew')
            v_scrollbar.grid(row=0, column=1, sticky='ns')
            h_scrollbar.grid(row=1, column=0, sticky='ew')
            table_frame.grid_rowconfigure(0, weight=1)
            table_frame.grid_columnconfigure(0, weight=1)
            
            # Load history data from Firebase
            history_data = []
            
            if self.firebase_initialized and self.db:
                try:
                    from datetime import datetime
                    
                    # Query visitors collection for active/registered visitors
                    try:
                        visitors_ref = self.db.collection('visitors')
                        try:
                            # Try to order by timestamp if it exists
                            visitors = visitors_ref.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(100).get()
                        except Exception:
                            # Fallback: get all visitors without ordering
                            visitors = visitors_ref.limit(100).get()
                        
                        for visitor_doc in visitors:
                            visitor_data = visitor_doc.to_dict()
                            rfid = visitor_data.get('rfid', '')
                            status_val = visitor_data.get('status', 'active')
                            
                            # Show entries with RFID assigned (including returned ones - we keep RFID for history)
                            if not rfid:
                                continue
                            
                            name = visitor_data.get('name', 'Unknown')
                            time_in_str = visitor_data.get('time_in', '')
                            time_out_str = visitor_data.get('time_out', '')
                            companion = str(visitor_data.get('companion_count', 0) or visitor_data.get('companion', 0) or '0')
                            
                            # Get ID type
                            id_type_val = visitor_data.get('id_type', '')
                            id_type = 'Not specified'
                            if id_type_val:
                                id_type_map = {
                                    'driver_license': "Driver's License",
                                    'national_id': "National ID",
                                    'passport': 'Passport',
                                    'not_specified': 'Not specified'
                                }
                                id_type = id_type_map.get(id_type_val, id_type_val.replace('_', ' ').title())
                            
                            time_in = ''
                            time_out = ''
                            
                            # Parse time_in if it exists
                            if time_in_str:
                                try:
                                    dt = datetime.strptime(time_in_str, "%Y-%m-%d %H:%M:%S")
                                    time_in = dt.strftime("%H:%M")
                                except:
                                    time_in = time_in_str[:5] if len(time_in_str) >= 5 else time_in_str
                            
                            # Parse time_out if it exists
                            if time_out_str:
                                try:
                                    dt = datetime.strptime(time_out_str, "%Y-%m-%d %H:%M:%S")
                                    time_out = dt.strftime("%H:%M")
                                except:
                                    time_out = time_out_str[:5] if len(time_out_str) >= 5 else time_out_str
                            
                            # Determine status: Only 2 statuses - "In-use" or "Returned"
                            # "In-use" = registered or active (has RFID assigned, not returned)
                            # "Returned" = RFID has been returned (status is 'returned')
                            if status_val == 'returned':
                                status = 'Returned'
                            else:
                                # Registered or active = In-use (even if time_out exists)
                                status = 'In-use'
                            
                            # Add entry if it has an RFID assigned
                            if rfid:
                                history_data.append({
                                    'name': name,
                                    'time_in': time_in,
                                    'time_out': time_out,
                                    'status': status,
                                    'companion': companion,
                                    'id_type': id_type,
                                    'rfid': rfid,
                                    'type': 'visitor'
                                })
                    except Exception as e:
                        print(f"⚠️ Warning: Could not load visitors: {e}")
                        import traceback
                        traceback.print_exc()
                    
                except Exception as e:
                    print(f"⚠️ Warning: Error loading RFID history: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Populate table with history data
            for item in history_data:
                tree.insert('', 'end', values=(
                    item['name'],
                    item['time_in'],
                    item['time_out'],
                    item['status'],
                    item['companion'],
                    item['id_type'],
                    item['rfid']
                ))
            
            # If no data, show message
            if not history_data:
                tree.insert('', 'end', values=('No RFID usage history found', '', '', '', '', '', ''))
            
            # Store tree reference for return function
            self.rfid_history_tree = tree
            
            # Button frame
            button_frame = tk.Frame(main_frame, bg='#ffffff')
            button_frame.pack(fill=tk.X, pady=(15, 0))
            
            # Return button
            return_btn = tk.Button(
                button_frame,
                text="Return",
                command=lambda: self.return_rfid_from_history_tab(),
                font=('Arial', 12, 'bold'),
                bg='#f97316',
                fg='white',
                relief='raised',
                bd=2,
                padx=20,
                pady=10,
                cursor='hand2',
                activebackground='#ea580c',
                activeforeground='white'
            )
            return_btn.pack(side=tk.RIGHT, padx=(10, 0))
            
            # Refresh button
            refresh_btn = tk.Button(
                button_frame,
                text="🔄 Refresh",
                command=lambda: self.refresh_rfid_history(),
                font=('Arial', 12),
                bg='#3b82f6',
                fg='white',
                relief='raised',
                bd=2,
                padx=20,
                pady=10,
                cursor='hand2',
                activebackground='#2563eb',
                activeforeground='white'
            )
            refresh_btn.pack(side=tk.RIGHT, padx=(10, 0))
            
            # Load initial data
            self.refresh_rfid_history()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create RFID history: {e}")
            import traceback
            traceback.print_exc()
    
    def refresh_rfid_history(self):
        """Refresh the RFID history table with latest data"""
        try:
            if not hasattr(self, 'rfid_history_tree') or self.rfid_history_tree is None:
                return
            
            tree = self.rfid_history_tree
            
            # Clear existing items
            for item in tree.get_children():
                tree.delete(item)
            
            # Load history data from Firebase
            history_data = []
            
            if self.firebase_initialized and self.db:
                try:
                    from datetime import datetime
                    
                    # Query visitors collection for all visitors with RFID assigned
                    try:
                        visitors_ref = self.db.collection('visitors')
                        # Get all visitors (no limit, we'll filter locally)
                        all_visitors = visitors_ref.get()
                        visitors_list = list(all_visitors)  # Convert to list once
                        
                        print(f"🔍 DEBUG: Found {len(visitors_list)} total visitors in Firebase")
                        
                        visitors_with_rfid = []
                        
                        for visitor_doc in visitors_list:
                            visitor_data = visitor_doc.to_dict()
                            if not visitor_data:
                                continue
                                
                            rfid = visitor_data.get('rfid', '')
                            # Only include visitors with RFID assigned (not None, not empty string)
                            if rfid and str(rfid).strip() != '':
                                visitors_with_rfid.append((visitor_doc, visitor_data))
                                print(f"✅ Found visitor with RFID: {visitor_data.get('name', 'Unknown')} - RFID: {rfid}")
                        
                        print(f"🔍 DEBUG: Found {len(visitors_with_rfid)} visitors with RFID assigned")
                        
                        # Sort by registration_time if available (most recent first)
                        try:
                            visitors_with_rfid.sort(key=lambda x: (
                                x[1].get('registration_time', '') or 
                                x[1].get('timestamp', '') or 
                                ''
                            ), reverse=True)
                        except Exception as e:
                            print(f"⚠️ Warning: Could not sort visitors: {e}")
                        
                        for visitor_doc, visitor_data in visitors_with_rfid:
                            rfid = visitor_data.get('rfid', '')
                            if not rfid or str(rfid).strip() == '':
                                continue
                                
                            name = visitor_data.get('name', 'Unknown')
                            # Time in and time out should only be set when visitor actually taps RFID
                            # Keep them empty if not set
                            time_in_str = visitor_data.get('time_in', '')
                            time_out_str = visitor_data.get('time_out', '')
                            status_val = visitor_data.get('status', 'registered')
                            companion = str(visitor_data.get('companion_count', 0) or visitor_data.get('companion', 0) or '0')
                            
                            # Get ID type
                            id_type_val = visitor_data.get('id_type', '')
                            id_type = 'Not specified'
                            if id_type_val:
                                id_type_map = {
                                    'driver_license': "Driver's License",
                                    'national_id': "National ID",
                                    'passport': 'Passport',
                                    'not_specified': 'Not specified'
                                }
                                id_type = id_type_map.get(id_type_val, id_type_val.replace('_', ' ').title())
                            
                            # Initialize time_in and time_out as empty
                            # Display both time_in and time_out if they exist (even after exit)
                            time_in = ''
                            time_out = ''
                            
                            # Parse time_in if it exists
                            if time_in_str:
                                try:
                                    dt = datetime.strptime(time_in_str, "%Y-%m-%d %H:%M:%S")
                                    time_in = dt.strftime("%H:%M")
                                except:
                                    time_in = time_in_str[:5] if len(time_in_str) >= 5 else time_in_str
                            
                            # Parse time_out if it exists
                            if time_out_str:
                                try:
                                    dt = datetime.strptime(time_out_str, "%Y-%m-%d %H:%M:%S")
                                    time_out = dt.strftime("%H:%M")
                                except:
                                    time_out = time_out_str[:5] if len(time_out_str) >= 5 else time_out_str
                            
                            # Determine status: Only 2 statuses - "In-use" or "Returned"
                            # "In-use" = registered or active (has RFID assigned, not returned)
                            # "Returned" = RFID has been returned (status is 'returned' only)
                            # Note: 'exited' status should not exist - timeout doesn't change status to 'exited'
                            if status_val == 'returned':
                                status = 'Returned'
                            else:
                                # Registered or active = In-use (even if time_out exists)
                                status = 'In-use'
                            
                            # Add to history data
                            history_data.append({
                                'name': name,
                                'time_in': time_in,  # Empty until visitor taps RFID
                                'time_out': time_out,  # Empty until visitor exits
                                'status': status,
                                'companion': companion,
                                'id_type': id_type,
                                'rfid': rfid,
                                'type': 'visitor'
                            })
                            
                        print(f"✅ Added {len(history_data)} visitors to history table")
                    except Exception as e:
                        print(f"⚠️ Warning: Could not load visitors: {e}")
                        import traceback
                        traceback.print_exc()
                    
                except Exception as e:
                    print(f"⚠️ Warning: Error loading RFID history: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Populate table with history data
            for item in history_data:
                tree.insert('', 'end', values=(
                    item['name'],
                    item['time_in'],
                    item['time_out'],
                    item['status'],
                    item['companion'],
                    item['id_type'],
                    item['rfid']
                ))
            
            # If no data, show message
            if not history_data:
                tree.insert('', 'end', values=('No RFID usage history found', '', '', '', '', '', ''))
                
        except Exception as e:
            print(f"⚠️ Error refreshing RFID history: {e}")
            import traceback
            traceback.print_exc()
    
    def return_rfid_from_history_tab(self):
        """Return RFID by showing popup dialog for RFID tap"""
        try:
            # Create popup dialog for returning RFID
            return_dialog = tk.Toplevel(self.root)
            return_dialog.title("Return RFID")
            return_dialog.geometry("450x200")
            return_dialog.resizable(False, False)
            
            # Center the dialog
            return_dialog.transient(self.root)
            return_dialog.grab_set()
            
            # Make dialog modal
            return_dialog.focus_set()
            
            # Main frame
            main_frame = tk.Frame(return_dialog, bg='#ffffff', padx=20, pady=20)
            main_frame.pack(fill=tk.BOTH, expand=True)
            
            # Instructions
            instruction_label1 = tk.Label(
                main_frame,
                text="Enter RFID to return (e.g. 0095129433)",
                font=('Arial', 10),
                bg='#ffffff',
                fg='#333333',
                anchor='w'
            )
            instruction_label1.pack(fill=tk.X, pady=(0, 5))
            
            instruction_label2 = tk.Label(
                main_frame,
                text="Place the RFID card to return on the reader:",
                font=('Arial', 10),
                bg='#ffffff',
                fg='#333333',
                anchor='w'
            )
            instruction_label2.pack(fill=tk.X, pady=(0, 10))
            
            # RFID input field
            rfid_entry_var = tk.StringVar()
            
            # Check if a row is selected in the table and pre-fill RFID
            prefill_rfid = ""
            if hasattr(self, 'rfid_history_tree') and self.rfid_history_tree is not None:
                try:
                    selected_item = self.rfid_history_tree.selection()
                    if selected_item:
                        item_values = self.rfid_history_tree.item(selected_item[0], 'values')
                        if item_values and len(item_values) >= 7:
                            prefill_rfid = item_values[6]  # Assigned RFID is the 7th column (index 6)
                            if prefill_rfid:
                                rfid_entry_var.set(prefill_rfid)
                except Exception as e:
                    print(f"⚠️ Could not get selected RFID from table: {e}")
            
            rfid_entry = tk.Entry(
                main_frame,
                textvariable=rfid_entry_var,
                font=('Arial', 12),
                width=30,
                relief='solid',
                bd=1
            )
            rfid_entry.pack(fill=tk.X, pady=(0, 15))
            rfid_entry.focus_set()
            
            # If RFID was pre-filled, select all text for easy editing
            if prefill_rfid:
                rfid_entry.select_range(0, tk.END)
            
            # Store reference for RFID auto-fill
            self.return_rfid_dialog = return_dialog
            self.return_rfid_entry_var = rfid_entry_var
            self.return_rfid_entry = rfid_entry
            
            # Activate RFID tap mode for return
            self.visitor_rfid_return_mode = True
            print("✅ RFID return mode activated - waiting for RFID tap")
            
            # Button frame
            button_frame = tk.Frame(main_frame, bg='#ffffff')
            button_frame.pack(fill=tk.X)
            
            # OK button
            def handle_ok():
                rfid = rfid_entry_var.get().strip()
                if not rfid:
                    messagebox.showwarning("No RFID", "Please enter or tap an RFID card to return.")
                    return
                
                # Process RFID return FIRST (while mode is still True)
                # process_rfid_return will set visitor_rfid_return_mode to False when done
                result = self.process_rfid_return(rfid)
                
                # Close dialog after processing
                return_dialog.destroy()
                self.return_rfid_dialog = None
                self.return_rfid_entry_var = None
                self.return_rfid_entry = None
            
            # Cancel button
            def handle_cancel():
                return_dialog.destroy()
                self.return_rfid_dialog = None
                self.return_rfid_entry_var = None
                self.return_rfid_entry = None
                self.visitor_rfid_return_mode = False
                print("❌ RFID return cancelled")
            
            ok_btn = tk.Button(
                button_frame,
                text="OK",
                command=handle_ok,
                font=('Arial', 10, 'bold'),
                bg='#4CAF50',
                fg='white',
                relief='raised',
                bd=2,
                padx=20,
                pady=5,
                cursor='hand2',
                activebackground='#45a049',
                activeforeground='white'
            )
            ok_btn.pack(side=tk.RIGHT, padx=(10, 0))
            
            cancel_btn = tk.Button(
                button_frame,
                text="Cancel",
                command=handle_cancel,
                font=('Arial', 10),
                bg='#f44336',
                fg='white',
                relief='raised',
                bd=2,
                padx=20,
                pady=5,
                cursor='hand2',
                activebackground='#da190b',
                activeforeground='white'
            )
            cancel_btn.pack(side=tk.RIGHT)
            
            # Handle Enter key
            def on_enter_key(event):
                handle_ok()
            
            rfid_entry.bind('<Return>', on_enter_key)
            
            # Handle Escape key
            def on_escape_key(event):
                handle_cancel()
            
            return_dialog.bind('<Escape>', on_escape_key)
            
            # Handle window close
            return_dialog.protocol("WM_DELETE_WINDOW", handle_cancel)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open return RFID dialog: {e}")
            import traceback
            traceback.print_exc()
            if hasattr(self, 'visitor_rfid_return_mode'):
                self.visitor_rfid_return_mode = False
    
    def process_rfid_return(self, rfid):
        """Process RFID return when card is tapped"""
        try:
            if not hasattr(self, 'visitor_rfid_return_mode') or not self.visitor_rfid_return_mode:
                return False  # Not in return mode
            
            rfid = str(rfid).strip()
            if not rfid:
                return False
            
            print(f"🔍 Processing RFID return for: {rfid}")
            
            # Find visitor with this RFID
            visitor_found = False
            visitor_name = 'Unknown'
            
            # Check Firebase to find visitor
            if self.firebase_initialized and self.db:
                try:
                    visitors_ref = self.db.collection('visitors')
                    visitors = visitors_ref.where('rfid', '==', rfid).get()
                    
                    for visitor_doc in visitors:
                        visitor_data = visitor_doc.to_dict()
                        status = visitor_data.get('status', '')
                        
                        # Only return if visitor is registered or active (not already returned)
                        if status in ('registered', 'active'):
                            visitor_found = True
                            visitor_name = visitor_data.get('name', 'Unknown')
                            
                            # Update visitor status to 'returned'
                            # Keep RFID in record for history purposes, but mark as returned
                            visitor_doc.reference.update({
                                'status': 'returned',
                                'time_out': self.get_current_timestamp() if not visitor_data.get('time_out') else visitor_data.get('time_out')
                                # Note: We keep 'rfid' field for history tracking - RFID availability is managed separately
                            })
                            
                            print(f"✅ Updated visitor record: {visitor_name} - RFID {rfid} marked as returned")
                            break
                except Exception as e:
                    print(f"⚠️ Warning: Could not query Firebase for RFID return: {e}")
            
            # Also check student assignments
            student_found = False
            if not visitor_found and self.firebase_initialized and self.db:
                try:
                    assignments_ref = self.db.collection('student_rfid_assignments')
                    assignments = assignments_ref.where('rfid', '==', rfid).where('status', '==', 'active').get()
                    
                    for assignment_doc in assignments:
                        assignment_data = assignment_doc.to_dict()
                        student_found = True
                        visitor_name = assignment_data.get('name', 'Unknown')
                        
                        # Update assignment status to 'returned'
                        assignment_doc.reference.update({'status': 'returned'})
                        print(f"✅ Updated student assignment: {visitor_name} - RFID {rfid} marked as returned")
                        break
                except Exception as e:
                    print(f"⚠️ Warning: Could not query student assignments: {e}")
            
            if not visitor_found and not student_found:
                messagebox.showwarning(
                    "RFID Not Found",
                    f"RFID {rfid} is not currently assigned to any visitor or student.\n\n"
                    f"Please check the RFID card and try again."
                )
                self.visitor_rfid_return_mode = False
                return False
            
            # Process return based on type
            if visitor_found:
                try:
                    # Remove from memory if exists
                    if hasattr(self, 'visitor_rfid_assignments') and rfid in self.visitor_rfid_assignments:
                        del self.visitor_rfid_assignments[rfid]
                    if hasattr(self, 'visitor_rfid_registry') and rfid in self.visitor_rfid_registry:
                        del self.visitor_rfid_registry[rfid]
                    
                    if rfid not in self.available_rfids:
                        self.available_rfids.append(rfid)
                    try:
                        self.update_rfid_availability_in_firebase(rfid, available=True)
                    except Exception:
                        pass
                    
                    try:
                        self.load_rfid_from_firebase()
                        self.update_student_rfid_list()
                    except Exception:
                        pass
                    
                    self.add_activity_log(f"Temporary visitor RFID returned from history: {rfid}")
                    messagebox.showinfo("RFID Returned", f"Visitor RFID {rfid} assigned to {visitor_name} has been returned and is now available.")
                    
                    # Refresh the history table
                    self.refresh_rfid_history()
                    
                    # Disable return mode
                    self.visitor_rfid_return_mode = False
                    return True
                    
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to return visitor RFID: {e}")
            
            elif student_found:
                try:
                    # Remove from memory if exists
                    if hasattr(self, 'student_rfid_assignments') and rfid in self.student_rfid_assignments:
                        del self.student_rfid_assignments[rfid]
                    
                    if rfid not in self.available_rfids:
                        self.available_rfids.append(rfid)
                    try:
                        self.update_rfid_availability_in_firebase(rfid, available=True)
                    except Exception:
                        pass
                    
                    try:
                        self.update_student_rfid_list()
                        self.load_rfid_from_firebase()
                    except Exception:
                        pass
                    
                    self.add_activity_log(f"Temporary student RFID returned from history: {rfid}")
                    messagebox.showinfo("RFID Returned", f"Student RFID {rfid} assigned to {visitor_name} has been returned and is now available.")
                    
                    # Refresh the history table
                    self.refresh_rfid_history()
                    
                    # Disable return mode
                    self.visitor_rfid_return_mode = False
                    return True
                    
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to return student RFID: {e}")
            
            # Disable return mode
            self.visitor_rfid_return_mode = False
            return False
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process RFID return: {e}")
            import traceback
            traceback.print_exc()
            if hasattr(self, 'visitor_rfid_return_mode'):
                self.visitor_rfid_return_mode = False
            return False
    
    def return_rfid_from_history(self, tree, history_window):
        """Return RFID selected from history table"""
        try:
            selected_item = tree.selection()
            if not selected_item:
                messagebox.showwarning("No Selection", "Please select a row from the table to return RFID.")
                return
            
            # Get RFID from selected row
            item_values = tree.item(selected_item[0], 'values')
            if not item_values or len(item_values) < 7:
                messagebox.showerror("Error", "Invalid row data.")
                return
            
            rfid = item_values[6]  # Assigned RFID is the 7th column (index 6)
            name = item_values[0]  # Name is the 1st column
            
            if not rfid or rfid == '':
                messagebox.showwarning("No RFID", "Selected row does not have an RFID assigned.")
                return
            
            # Confirm return
            confirm = messagebox.askyesno(
                "Confirm Return",
                f"Return RFID {rfid} assigned to {name}?\n\nThis will make the RFID available for assignment again."
            )
            
            if not confirm:
                return
            
            # Check if it's a visitor or student
            is_visitor = True
            is_student = False
            
            # Check Firebase to determine type
            if self.firebase_initialized and self.db:
                try:
                    visitors = self.db.collection('visitors').where('rfid', '==', rfid).get()
                    if visitors:
                        is_visitor = True
                        is_student = False
                    else:
                        student_assignments = self.db.collection('student_rfid_assignments').where('rfid', '==', rfid).get()
                        if student_assignments:
                            is_student = True
                            is_visitor = False
                except Exception:
                    pass
            
            # Process return based on type
            if is_visitor:
                try:
                    if hasattr(self, 'visitor_rfid_assignments') and rfid in self.visitor_rfid_assignments:
                        del self.visitor_rfid_assignments[rfid]
                    if hasattr(self, 'visitor_rfid_registry') and rfid in self.visitor_rfid_registry:
                        del self.visitor_rfid_registry[rfid]
                    
                    if self.firebase_initialized and self.db:
                        try:
                            visitors_ref = self.db.collection('visitors')
                            visitors = visitors_ref.where('rfid', '==', rfid).get()
                            for visitor_doc in visitors:
                                visitor_data = visitor_doc.to_dict()
                                status = visitor_data.get('status', '')
                                if status == 'registered' or status == 'active':
                                    visitor_doc.reference.update({
                                        'status': 'returned',
                                        'time_out': self.get_current_timestamp() if not visitor_data.get('time_out') else visitor_data.get('time_out')
                                        # Note: We keep 'rfid' field for history tracking - RFID availability is managed separately
                                    })
                        except Exception as e:
                            print(f"⚠️ Warning: Could not update visitor record: {e}")
                    
                    if rfid not in self.available_rfids:
                        self.available_rfids.append(rfid)
                    try:
                        self.update_rfid_availability_in_firebase(rfid, available=True)
                    except Exception:
                        pass
                    
                    try:
                        self.load_rfid_from_firebase()
                        self.update_student_rfid_list()
                    except Exception:
                        pass
                    
                    self.add_activity_log(f"Temporary visitor RFID returned from history: {rfid}")
                    messagebox.showinfo("RFID Returned", f"Visitor RFID {rfid} has been returned and is now available.")
                    
                    history_window.destroy()
                    self.show_rfid_history_dialog()
                    
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to return visitor RFID: {e}")
            
            elif is_student:
                try:
                    if hasattr(self, 'student_rfid_assignments') and rfid in self.student_rfid_assignments:
                        del self.student_rfid_assignments[rfid]
                    
                    if self.firebase_initialized and self.db:
                        try:
                            assignments_ref = self.db.collection('student_rfid_assignments')
                            assignments = assignments_ref.where('rfid', '==', rfid).where('status', '==', 'active').get()
                            for assignment_doc in assignments:
                                assignment_doc.reference.update({'status': 'returned'})
                        except Exception:
                            pass
                    
                    if rfid not in self.available_rfids:
                        self.available_rfids.append(rfid)
                    try:
                        self.update_rfid_availability_in_firebase(rfid, available=True)
                    except Exception:
                        pass
                    
                    try:
                        self.update_student_rfid_list()
                        self.load_rfid_from_firebase()
                    except Exception:
                        pass
                    
                    self.add_activity_log(f"Temporary student RFID returned from history: {rfid}")
                    messagebox.showinfo("RFID Returned", f"Student RFID {rfid} has been returned and is now available.")
                    
                    history_window.destroy()
                    self.show_rfid_history_dialog()
                    
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to return student RFID: {e}")
            else:
                messagebox.showwarning("Not Found", f"RFID {rfid} is not a currently assigned temporary RFID.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to return RFID: {e}")
            import traceback
            traceback.print_exc()

    def clear_rfid_document(self, rfid_id):
        """Clear the RFID document in Firestore by overwriting it with an empty dict."""
        try:
            if self.firebase_initialized and self.db:
                rfid_ref = self.db.collection('Empty RFID').document(rfid_id)
                # Overwrite with empty data
                rfid_ref.set({})
                print(f"SUCCESS: Cleared RFID document in Firebase: {rfid_id}")
                self.add_activity_log(f"Cleared RFID document in Firebase: {rfid_id}")
                return True
            else:
                print("WARNING: Firebase not initialized - cannot clear RFID document")
                return False
        except Exception as e:
            print(f"ERROR: Failed to clear RFID document {rfid_id}: {e}")
            return False

    def clear_rfid_dialog_for_visitors(self):
        try:
            rfid = simpledialog.askstring("Clear RFID", "Enter RFID to clear in Firebase (visitor empty pool)")
            if not rfid:
                return
            rfid = rfid.strip()
            confirm = messagebox.askyesno("Confirm Clear", f"Are you sure you want to CLEAR the RFID document '{rfid}' in Firebase? This will remove all fields and mark it empty.")
            if not confirm:
                return
            ok = self.clear_rfid_document(rfid)
            if ok:
                messagebox.showinfo("Cleared", f"RFID {rfid} cleared in Firebase.")
                try:
                    self.load_rfid_from_firebase()
                except Exception:
                    pass
            else:
                messagebox.showwarning("Failed", f"Failed to clear RFID {rfid} in Firebase. Check logs.")
        except Exception as e:
            messagebox.showerror("Error", f"Error clearing RFID: {e}")

    def clear_rfid_dialog_for_students(self):
        try:
            rfid = simpledialog.askstring("Clear RFID", "Enter RFID to clear in Firebase (student empty pool)")
            if not rfid:
                return
            rfid = rfid.strip()
            confirm = messagebox.askyesno("Confirm Clear", f"Are you sure you want to CLEAR the RFID document '{rfid}' in Firebase? This will remove all fields and mark it empty.")
            if not confirm:
                return
            ok = self.clear_rfid_document(rfid)
            if ok:
                messagebox.showinfo("Cleared", f"RFID {rfid} cleared in Firebase.")
                try:
                    self.update_student_rfid_list()
                except Exception:
                    pass
            else:
                messagebox.showwarning("Failed", f"Failed to clear RFID {rfid} in Firebase. Check logs.")
        except Exception as e:
            messagebox.showerror("Error", f"Error clearing RFID: {e}")
    
    def handle_visitor_rfid_tap(self, rfid):
        """Handle visitor RFID tap - Simple toggle: 1st tap=ENTRY, 2nd tap=EXIT"""
        try:
            # Validate registration/assignment exists (support assignment tracking)
            visitor_info = None
            if hasattr(self, 'visitor_rfid_assignments') and rfid in self.visitor_rfid_assignments:
                visitor_info = self.visitor_rfid_assignments[rfid]
            elif rfid in self.visitor_rfid_registry:
                visitor_info = self.visitor_rfid_registry[rfid]

            if not visitor_info:
                self.add_activity_log(f"Unknown RFID tapped: {rfid}")
                messagebox.showwarning("Unknown RFID", f"RFID {rfid} is not registered to any visitor.")
                return
            
            # Simple toggle system - initialize if not exists
            if not hasattr(self, 'rfid_status_tracker'):
                self.rfid_status_tracker = {}
            
            # Validate expiry if assignment exists
            expiry = visitor_info.get('expiry_time')
            if expiry:
                try:
                    expiry_dt = datetime.strptime(expiry, "%Y-%m-%d %H:%M:%S")
                    if datetime.now() > expiry_dt:
                        # Assignment expired
                        self.add_activity_log(f"Visitor temporary RFID expired for {visitor_info.get('name')} - RFID: {rfid}")
                        messagebox.showwarning("Expired RFID", "This temporary RFID has expired.")
                        # Clean up assignment if present
                        try:
                            if hasattr(self, 'visitor_rfid_assignments') and rfid in self.visitor_rfid_assignments:
                                del self.visitor_rfid_assignments[rfid]
                        except Exception:
                            pass
                        # Also unregister visitor from registry
                        if rfid in self.visitor_rfid_registry:
                            del self.visitor_rfid_registry[rfid]
                        # Make RFID available again
                        if rfid not in self.available_rfids:
                            self.available_rfids.append(rfid)
                        self.update_rfid_availability_in_firebase(rfid, available=True)
                        self.load_rfid_from_firebase()
                        return
                except Exception:
                    pass

            # Get current status (default to 'EXIT' for first tap)
            current_status = self.rfid_status_tracker.get(rfid, 'EXIT')
            
            if current_status == 'EXIT':
                # This tap is ENTRY
                self.rfid_status_tracker[rfid] = 'ENTRY'
                # Update main screen with visitor name (no detection)
                try:
                    person_info = {
                        'id': visitor_info.get('visitor_id', rfid),
                        'name': visitor_info.get('name', f'Visitor {rfid}'),
                        'type': 'visitor',
                        'course': 'N/A',
                        'gender': 'N/A',
                        'timestamp': self.get_current_timestamp(),
                        'status': 'TIME-IN',
                        'action': 'ENTRY',
                        'rfid': rfid
                    }
                    if self.main_screen_window and self.main_screen_window.winfo_exists():
                        self.update_main_screen_with_person(person_info)
                except Exception:
                    pass

                self.handle_visitor_timein(rfid, visitor_info)
                print(f"SUCCESS: Visitor {visitor_info.get('name', 'Unknown')} - ENTRY")
            else:
                # This tap is EXIT
                self.rfid_status_tracker[rfid] = 'EXIT'
                # Update main screen to TIME-OUT display for visitor
                try:
                    person_info = {
                        'id': visitor_info.get('visitor_id', rfid),
                        'name': visitor_info.get('name', f'Visitor {rfid}'),
                        'type': 'visitor',
                        'course': 'N/A',
                        'gender': 'N/A',
                        'timestamp': self.get_current_timestamp(),
                        'status': 'TIME-OUT',
                        'action': 'EXIT',
                        'rfid': rfid
                    }
                    if self.main_screen_window and self.main_screen_window.winfo_exists():
                        self.update_main_screen_with_person(person_info)
                except Exception:
                    pass

                self.handle_visitor_timeout(rfid, visitor_info)
                print(f"SUCCESS: Visitor {visitor_info.get('name', 'Unknown')} - EXIT")
                
        except Exception as e:
            print(f"ERROR: Error handling visitor RFID tap: {e}")
            self.add_activity_log(f"Error handling RFID {rfid}: {e}")
    
    def handle_visitor_timein(self, rfid, visitor_info):
        """Handle visitor time-in - NO DETECTION"""
        try:
            current_time = self.get_current_timestamp()
            
            # Add to active visitors
            self.active_visitors[rfid] = {
                **visitor_info,
                'time_in': current_time,
                'status': 'active'
            }
            
            # Update visitor registry
            self.visitor_rfid_registry[rfid]['status'] = 'active'
            
            # Log activity
            self.add_activity_log(f"Visitor Time-In: {visitor_info['name']} (RFID: {rfid}) - {current_time}")
            
            # Save to Firebase
            self.save_visitor_activity_to_firebase(rfid, 'time_in', current_time)

            # Update assignment last_used/status if using temporary visitor assignment
            try:
                if hasattr(self, 'visitor_rfid_assignments') and rfid in self.visitor_rfid_assignments:
                    self.visitor_rfid_assignments[rfid]['last_used'] = current_time
                    self.visitor_rfid_assignments[rfid]['status'] = 'active'
            except Exception:
                pass
            
            # Ensure recent entries shows this as ENTRY
            try:
                person_data_entry = {
                    'id': visitor_info.get('visitor_id', rfid),
                    'name': visitor_info.get('name', f'Visitor {rfid}'),
                    'type': 'visitor',
                    'timestamp': current_time,
                    'status': 'TIME-IN',
                    'action': 'ENTRY',
                    'guard_id': self.current_guard_id or 'Unknown'
                }
                self.add_to_recent_entries(person_data_entry)
            except Exception:
                pass

            # Update main screen
            if self.main_screen_window and self.main_screen_window.winfo_exists():
                self.update_main_screen_with_visitor(visitor_info, 'time_in', current_time)
            
            # Show success message - NO DETECTION for visitors
            self.show_green_success_message("Visitor Time-In", 
                              f"Visitor: {visitor_info['name']}\n"
                              f"Purpose: {visitor_info['purpose']}\n"
                              f"Time-In: {current_time}\n"
                              f"RFID: {rfid}\n\n"
                              f"Note: No uniform detection for visitors")
            
            # Ensure detection is NOT started for visitors
            if self.detection_active:
                self.stop_detection()
                self.add_activity_log("Detection stopped - Visitor entry (no uniform checking required)")
            
            # Refresh RFID history table if it exists (tab is open)
            try:
                if hasattr(self, 'rfid_history_tree') and self.rfid_history_tree is not None:
                    # Small delay to ensure Firebase write is complete
                    self.root.after(500, self.refresh_rfid_history)
                    print(f"✅ Scheduled RFID history refresh after visitor time-in")
            except Exception as e:
                print(f"⚠️ Warning: Could not refresh RFID history: {e}")
            
        except Exception as e:
            print(f"ERROR: Error handling visitor time-in: {e}")
            self.add_activity_log(f"Error processing time-in for RFID {rfid}: {e}")
    
    def handle_visitor_timeout(self, rfid, visitor_info):
        """Handle visitor time-out - NO DETECTION"""
        try:
            current_time = self.get_current_timestamp()
            time_in = self.active_visitors[rfid]['time_in']
            
            # Calculate duration
            from datetime import datetime
            time_in_dt = datetime.strptime(time_in, "%Y-%m-%d %H:%M:%S")
            time_out_dt = datetime.strptime(current_time, "%Y-%m-%d %H:%M:%S")
            duration = time_out_dt - time_in_dt
            
            # Ensure detection is NOT active for visitors
            if self.detection_active:
                self.stop_detection()
                self.add_activity_log("Detection stopped - Visitor exit (no uniform checking required)")

            # Restart guard camera (no detection) so system returns to standby
            try:
                self.initialize_guard_camera_feed()
            except Exception:
                pass
            
            # Log activity
            self.add_activity_log(f"Visitor Time-Out: {visitor_info['name']} (RFID: {rfid}) - {current_time} - Duration: {duration}")
            
            # Save to Firebase
            self.save_visitor_activity_to_firebase(rfid, 'time_out', current_time, duration=str(duration))

            # Update assignment last_used but DO NOT change status to 'exited'
            # Status should remain 'active' until RFID is returned via Return RFID button
            try:
                if hasattr(self, 'visitor_rfid_assignments') and rfid in self.visitor_rfid_assignments:
                    self.visitor_rfid_assignments[rfid]['last_used'] = current_time
                    # Keep status as 'active' - don't change to 'exited' here
                    # Status will only change to 'returned' when Return RFID button is clicked
            except Exception:
                pass
            
            # Update main screen
            if self.main_screen_window and self.main_screen_window.winfo_exists():
                self.update_main_screen_with_visitor(visitor_info, 'time_out', current_time, duration=str(duration))
            
            # Ensure recent entries shows this as EXIT
            try:
                person_data_exit = {
                    'id': visitor_info.get('visitor_id', rfid),
                    'name': visitor_info.get('name', f'Visitor {rfid}'),
                    'type': 'visitor',
                    'timestamp': current_time,
                    'status': 'TIME-OUT',
                    'action': 'EXIT',
                    'guard_id': self.current_guard_id or 'Unknown'
                }
                self.add_to_recent_entries(person_data_exit)
            except Exception:
                pass

            # Reset RFID for reuse
            self.reset_visitor_rfid(rfid)
            
            # Refresh RFID history table if it exists (tab is open)
            try:
                if hasattr(self, 'rfid_history_tree') and self.rfid_history_tree is not None:
                    # Small delay to ensure Firebase write is complete
                    self.root.after(500, self.refresh_rfid_history)
                    print(f"✅ Scheduled RFID history refresh after visitor time-out")
            except Exception as e:
                print(f"⚠️ Warning: Could not refresh RFID history: {e}")
            
            self.show_green_success_message("Visitor Time-Out", 
                              f"Visitor: {visitor_info['name']}\n"
                              f"Time-In: {time_in}\n"
                              f"Time-Out: {current_time}\n"
                              f"Duration: {duration}\n"
                              f"RFID: {rfid} - Now available for reuse\n\n"
                              f"Note: No uniform detection for visitors")
            
        except Exception as e:
            print(f"ERROR: Error handling visitor time-out: {e}")
            self.add_activity_log(f"Error processing time-out for RFID {rfid}: {e}")
    
    def reset_visitor_rfid(self, rfid):
        """Reset RFID card for reuse after visitor exit"""
        try:
            # Remove from active visitors
            if rfid in self.active_visitors:
                del self.active_visitors[rfid]
            
            # Update visitor registry - keep status as 'active' until RFID is returned
            # Don't change status to 'exited' - it should remain 'active' until Return RFID is clicked
            if rfid in self.visitor_rfid_registry:
                # Keep status as 'active' - status will only change to 'returned' when Return RFID button is clicked
                pass

            # If this RFID was a temporary visitor assignment, keep it assigned until expiry
            if hasattr(self, 'visitor_rfid_assignments') and rfid in self.visitor_rfid_assignments:
                # do not fully unassign here; leave assignment in place until expiry
                print(f"INFO: Temporary visitor RFID {rfid} retained until expiry: {self.visitor_rfid_assignments[rfid].get('expiry_time')}")
            else:
                # Add back to available RFID list
                if rfid not in self.available_rfids:
                    self.available_rfids.append(rfid)

                # Update RFID availability in Firebase
                self.update_rfid_availability_in_firebase(rfid, available=True)

                # Update RFID dropdown
                self.load_rfid_from_firebase()

                print(f"SUCCESS: RFID {rfid} reset and available for reuse")
            
        except Exception as e:
            print(f"ERROR: Error resetting RFID {rfid}: {e}")
    
    def handle_teacher_rfid_tap(self, rfid):
        """Handle teacher RFID tap - Simple toggle: 1st tap=ENTRY, 2nd tap=EXIT (no uniform detection)"""
        try:
            # Get teacher info from Firebase
            teacher_info = self.get_teacher_info_by_rfid(rfid)
            
            if not teacher_info:
                self.add_activity_log(f"Unknown RFID tapped: {rfid}")
                messagebox.showwarning("Unknown RFID", f"RFID {rfid} is not registered to any teacher.")
                return
            
            # Simple toggle system - initialize if not exists
            if not hasattr(self, 'rfid_status_tracker'):
                self.rfid_status_tracker = {}
            
            # Get current status from tracker first (like visitors do)
            # This ensures consistent toggle behavior even if Firebase check has timing issues
            current_status = self.rfid_status_tracker.get(rfid, 'EXIT')
            
            # Validate with Firebase check as fallback (if tracker doesn't have status, check Firebase)
            if current_status == 'EXIT':
                # Check Firebase to confirm teacher is actually outside
                has_entry = self.check_teacher_has_entry_record(rfid)
                if has_entry:
                    # Firebase says teacher is inside, but tracker says outside - trust Firebase
                    current_status = 'ENTRY'
                    self.rfid_status_tracker[rfid] = 'ENTRY'
            
            if current_status == 'EXIT':
                # This tap is ENTRY
                self.rfid_status_tracker[rfid] = 'ENTRY'
                
                # Handle time-in first (this saves to Firebase and adds to activity log)
                self.handle_teacher_timein(rfid, teacher_info)
                
                # Update main screen with teacher info (no detection)
                # This is called AFTER handle_teacher_timein so the activity log entry is already added
                try:
                    person_info = {
                        'id': rfid,
                        'name': teacher_info.get('name', f'Teacher {rfid}'),
                        'type': 'teacher',
                        'course': 'N/A',
                        'gender': teacher_info.get('gender', 'Unknown'),
                        'timestamp': self.get_current_timestamp(),
                        'status': 'TIME-IN',
                        'action': 'ENTRY',
                        'rfid': rfid
                    }
                    if self.main_screen_window and self.main_screen_window.winfo_exists():
                        self.update_main_screen_with_person(person_info)
                        # Note: Entry is already added by handle_teacher_timein via add_to_recent_entries
                        # No need to manually add again to avoid duplicates
                except Exception:
                    pass

                print(f"SUCCESS: Teacher {teacher_info.get('name', 'Unknown')} - ENTRY")
            else:
                # This tap is EXIT
                self.rfid_status_tracker[rfid] = 'EXIT'
                
                # Handle time-out first (this saves to Firebase and adds to activity log)
                self.handle_teacher_timeout(rfid, teacher_info)
                
                # Update main screen to TIME-OUT display for teacher
                # This is called AFTER handle_teacher_timeout so the activity log entry is already added
                try:
                    person_info = {
                        'id': rfid,
                        'name': teacher_info.get('name', f'Teacher {rfid}'),
                        'type': 'teacher',
                        'course': 'N/A',
                        'gender': teacher_info.get('gender', 'Unknown'),
                        'timestamp': self.get_current_timestamp(),
                        'status': 'TIME-OUT',
                        'action': 'EXIT',
                        'rfid': rfid
                    }
                    if self.main_screen_window and self.main_screen_window.winfo_exists():
                        self.update_main_screen_with_person(person_info)
                        # Note: Entry is already added by handle_teacher_timeout via add_to_recent_entries
                        # No need to manually add again to avoid duplicates
                except Exception:
                    pass

                print(f"SUCCESS: Teacher {teacher_info.get('name', 'Unknown')} - EXIT")
                
        except Exception as e:
            print(f"ERROR: Error handling teacher RFID tap: {e}")
            self.add_activity_log(f"Error handling teacher RFID {rfid}: {e}")
    
    def handle_teacher_timein(self, rfid, teacher_info):
        """Handle teacher time-in - NO DETECTION"""
        try:
            current_time = self.get_current_timestamp()
            
            # Log activity
            self.add_activity_log(f"entry(teacher) - {teacher_info.get('name', 'Unknown Teacher')}")
            
            # Save to Firebase
            self.save_teacher_activity_to_firebase(rfid, 'time_in', current_time, teacher_info)
            
            # Add to recent entries (screen update will be handled by handle_teacher_rfid_tap)
            try:
                person_data_entry = {
                    'id': rfid,
                    'name': teacher_info.get('name', f'Teacher {rfid}'),
                    'type': 'teacher',
                    'timestamp': current_time,
                    'status': 'TIME-IN',
                    'action': 'ENTRY',
                    'guard_id': self.current_guard_id or 'Unknown'
                }
                self.add_to_recent_entries(person_data_entry)
            except Exception:
                pass

        except Exception as e:
            print(f"ERROR: Error handling teacher time-in: {e}")
            self.add_activity_log(f"Error processing time-in for teacher RFID {rfid}: {e}")
    
    def handle_teacher_timeout(self, rfid, teacher_info):
        """Handle teacher time-out - NO DETECTION"""
        try:
            current_time = self.get_current_timestamp()
            
            # Get time_in from Firebase to calculate duration (optional)
            time_in = None
            try:
                if self.firebase_initialized and self.db:
                    # Get most recent time_in for this teacher
                    query = self.db.collection('teacher_activities').where('rfid', '==', rfid).where('activity_type', '==', 'time_in').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(1)
                    activities = query.get()
                    if activities:
                        for doc in activities:
                            activity_data = doc.to_dict()
                            time_in_str = activity_data.get('timestamp')
                            if time_in_str:
                                from datetime import datetime
                                time_in = datetime.strptime(time_in_str, "%Y-%m-%d %H:%M:%S")
                                break
            except Exception:
                pass
            
            # Log activity
            self.add_activity_log(f"exit(teacher) - {teacher_info.get('name', 'Unknown Teacher')}")
            
            # Save to Firebase
            self.save_teacher_activity_to_firebase(rfid, 'time_out', current_time, teacher_info)
            
            # Add to recent entries (screen update will be handled by handle_teacher_rfid_tap)
            try:
                person_data_exit = {
                    'id': rfid,
                    'name': teacher_info.get('name', f'Teacher {rfid}'),
                    'type': 'teacher',
                    'timestamp': current_time,
                    'status': 'TIME-OUT',
                    'action': 'EXIT',
                    'guard_id': self.current_guard_id or 'Unknown'
                }
                self.add_to_recent_entries(person_data_exit)
            except Exception:
                pass

        except Exception as e:
            print(f"ERROR: Error handling teacher time-out: {e}")
            self.add_activity_log(f"Error processing time-out for teacher RFID {rfid}: {e}")
    
    def save_teacher_activity_to_firebase(self, rfid, activity_type, timestamp, teacher_info):
        """Save teacher activity to Firebase"""
        try:
            if self.firebase_initialized and self.db:
                activity_data = {
                    'name': teacher_info.get('name', 'Unknown Teacher'),
                    'rfid': rfid,
                    'activity_type': activity_type,  # 'time_in' or 'time_out'
                    'timestamp': timestamp,
                    'gender': teacher_info.get('gender', 'Unknown')
                }
                
                # Save to teacher_activities collection (auto-generated document ID)
                activity_ref = self.db.collection('teacher_activities').document()
                activity_ref.set(activity_data)
                
                print(f"✅ SUCCESS: Teacher activity saved to Firebase: {activity_type} for {teacher_info.get('name', 'Unknown')} at {timestamp}")
            else:
                print("WARNING: Firebase not available - teacher activity not saved to cloud")
                
        except Exception as e:
            print(f"ERROR: Error saving teacher activity to Firebase: {e}")
            import traceback
            traceback.print_exc()
    
    def save_visitor_activity_to_firebase(self, rfid, activity_type, timestamp, duration=None):
        """Save visitor activity to Firebase"""
        try:
            if self.firebase_initialized and self.db:
                visitor_info = self.visitor_rfid_registry[rfid]
                activity_data = {
                    'visitor_id': visitor_info['visitor_id'],
                    'name': visitor_info['name'],
                    'rfid': rfid,
                    'activity_type': activity_type,  # 'time_in' or 'time_out'
                    'timestamp': timestamp,
                    'purpose': visitor_info['purpose']
                }
                
                if duration:
                    activity_data['duration'] = duration
                
                # Save to visitor_activities collection (using visitor_id_timestamp format)
                visitor_id = visitor_info.get('visitor_id')
                formatted_timestamp = self.format_document_id(timestamp)
                doc_id = f"{visitor_id}_{formatted_timestamp}"
                activity_ref = self.db.collection('visitor_activities').document(doc_id)
                activity_ref.set(activity_data)
                
                # Also update/maintain visitor document in visitors collection
                if visitor_id:
                    visitor_ref = self.db.collection('visitors').document(visitor_id)
                    visitor_doc = visitor_ref.get()
                    
                    visitor_update_data = {
                        'visitor_id': visitor_id,
                        'name': visitor_info.get('name', 'Unknown'),
                        'purpose': visitor_info.get('purpose', 'Unknown'),
                        'last_updated': timestamp
                    }
                    
                    # Update time_in or time_out based on activity type
                    if activity_type == 'time_in':
                        visitor_update_data['time_in'] = timestamp
                        visitor_update_data['status'] = 'active'
                    elif activity_type == 'time_out':
                        visitor_update_data['time_out'] = timestamp
                        # Don't change status to 'exited' - keep it as 'active' until RFID is returned
                        # Status will only change to 'returned' when Return RFID button is clicked
                        # Preserve existing status (should be 'active' or 'registered')
                        existing_data = visitor_doc.to_dict() if visitor_doc.exists else {}
                        if 'status' in existing_data and existing_data['status'] in ('active', 'registered'):
                            visitor_update_data['status'] = existing_data['status']
                        else:
                            visitor_update_data['status'] = 'active'
                    
                    # Set created_at only if document doesn't exist
                    if not visitor_doc.exists:
                        visitor_update_data['created_at'] = timestamp
                    else:
                        # Preserve existing registration_time if available
                        existing_data = visitor_doc.to_dict()
                        if 'registration_time' in existing_data:
                            visitor_update_data['registration_time'] = existing_data['registration_time']
                        if 'rfid' in existing_data:
                            visitor_update_data['rfid'] = existing_data['rfid']
                        if 'id_type' in existing_data:
                            visitor_update_data['id_type'] = existing_data['id_type']
                    
                    # Update or create visitor document
                    visitor_ref.set(visitor_update_data, merge=True)
                    print(f"SUCCESS: Visitor document updated in Firebase: {visitor_id}")
                
                print(f"SUCCESS: Visitor activity saved to Firebase: {activity_type} for {visitor_info['name']}")
            else:
                print("WARNING: Firebase not available - visitor activity not saved to cloud")
                
        except Exception as e:
            print(f"ERROR: Failed to save visitor activity to Firebase: {e}")
    
    def update_main_screen_with_visitor(self, visitor_info, activity_type, timestamp, duration=None):
        """Update main screen with visitor information"""
        try:
            if not self.main_screen_window or not self.main_screen_window.winfo_exists():
                return
            
            # Determine status based on activity type
            if activity_type == 'time_in':
                status = 'TIME-IN'
                action = 'ENTRY'
            else:  # time_out
                status = 'TIME-OUT'
                action = 'EXIT'
            
            # Create person_info dict for visitor
            person_info = {
                'id': visitor_info.get('visitor_id', visitor_info.get('rfid', 'Unknown')),
                'name': visitor_info.get('name', 'Unknown Visitor'),
                'type': 'visitor',
                'timestamp': timestamp,
                'status': status,
                'action': action,
                'rfid': visitor_info.get('rfid', 'Unknown'),
                'purpose': visitor_info.get('purpose', 'N/A'),
                'id_type': visitor_info.get('id_type', 'N/A')
            }
            
            # Update main screen using the standard function
            # This will automatically schedule the 15-second clear timer
            self.update_main_screen_with_person(person_info)
            
            print(f"📺 Main screen updated with visitor {activity_type}: {visitor_info['name']}")
            
        except Exception as e:
            print(f"ERROR: Error updating main screen with visitor: {e}")
    
    def handle_student_forgot_id_rfid_tap(self, rfid):
        """Handle student forgot ID RFID tap - Simple toggle: 1st tap=ENTRY, 2nd tap=EXIT"""
        try:
            if rfid not in self.student_rfid_assignments:
                self.add_activity_log(f"Unknown student RFID tapped: {rfid}")
                messagebox.showwarning("Unknown RFID", f"RFID {rfid} is not assigned to any student.")
                return
            
            assignment_info = self.student_rfid_assignments[rfid]
            
            # Simple toggle system - initialize if not exists
            if not hasattr(self, 'rfid_status_tracker'):
                self.rfid_status_tracker = {}
            
            # Get current status (default to 'EXIT' for first tap)
            current_status = self.rfid_status_tracker.get(rfid, 'EXIT')
            
            # Validate expiry: only active assignments within expiry_time are valid
            expiry = assignment_info.get('expiry_time')
            if expiry:
                try:
                    expiry_dt = datetime.strptime(expiry, "%Y-%m-%d %H:%M:%S")
                    if datetime.now() > expiry_dt:
                        # Assignment expired
                        self.add_activity_log(f"Temporary RFID expired for {assignment_info.get('name')} - RFID: {rfid}")
                        messagebox.showwarning("Expired RFID", "This temporary RFID has expired.")
                        # Clean up assignment
                        if rfid in self.student_rfid_assignments:
                            del self.student_rfid_assignments[rfid]
                        if rfid not in self.student_forgot_rfids:
                            self.student_forgot_rfids.append(rfid)
                        return
                except Exception:
                    pass

            if current_status == 'EXIT':
                # This tap is ENTRY -> now the student actually uses the temporary RFID
                self.rfid_status_tracker[rfid] = 'ENTRY'
                
                # CRITICAL: Ensure detection flags are NOT set for temporary RFID (detection is skipped)
                self.detection_active = False
                self.uniform_detection_complete = False
                
                # When student uses the temporary RFID, update main screen with their info
                try:
                    person_info = {
                        'id': assignment_info.get('student_id', rfid),
                        'name': assignment_info.get('name', f'Student {rfid}'),
                        'type': 'student',
                        'course': assignment_info.get('course', 'Unknown'),
                        'gender': assignment_info.get('gender', 'Unknown'),
                        'timestamp': self.get_current_timestamp(),
                        'status': 'TIME-IN',
                        'action': 'ENTRY',
                        'temporary_rfid': True,  # Flag to indicate this is temporary RFID (skip detection)
                        'skip_detection': True,   # Explicit flag to skip detection in UI
                        'rfid': rfid
                    }
                    # Update main screen only now that student physically tapped
                    if self.main_screen_window and self.main_screen_window.winfo_exists():
                        self.update_main_screen_with_person(person_info)
                except Exception:
                    pass

                self.handle_student_forgot_id_timein(rfid, assignment_info)
                print(f"SUCCESS: Student {assignment_info.get('name', 'Unknown')} - ENTRY (temporary RFID, detection skipped)")
            else:
                # This tap is EXIT
                self.rfid_status_tracker[rfid] = 'EXIT'
                self.handle_student_forgot_id_timeout(rfid, assignment_info)
                print(f"SUCCESS: Student {assignment_info.get('name', 'Unknown')} - EXIT")
                
        except Exception as e:
            print(f"ERROR: Error handling student forgot ID RFID tap: {e}")
            self.add_activity_log(f"Error handling student RFID {rfid}: {e}")
    
    def handle_student_forgot_id_timein(self, rfid, assignment_info):
        """Handle student forgot ID time-in - WITH DETECTION"""
        try:
            current_time = self.get_current_timestamp()
            
            # Log activity
            self.add_activity_log(f"Student Time-In (Forgot ID): {assignment_info['name']} (RFID: {rfid}) - {current_time}")
            
            # Add automatic violation for forgot ID (using empty RFID)
            self.add_activity_log(f"VIOLATION: Student {assignment_info['name']} forgot school ID - using temporary RFID")
            
            # Save to Firebase
            self.save_student_activity_to_firebase(rfid, 'time_in', current_time, assignment_info)
            
            # Save violation to Firebase
            student_id = assignment_info.get('student_id')
            if student_id:
                violation_details = f"Student forgot school ID and used temporary RFID: {rfid}"
                self.save_student_violation_to_firebase(student_id, 'forgot_id', current_time, assignment_info, violation_details)
            
            # CRITICAL: Ensure detection flags are NOT set for temporary RFID (detection is skipped)
            self.detection_active = False
            self.uniform_detection_complete = False
            
            # Update main screen
            if self.main_screen_window and self.main_screen_window.winfo_exists():
                # Show student information on the main screen using the student's real name
                # but DO NOT start uniform detection because this is a temporary RFID (violation)
                try:
                    person_info = {
                        'id': assignment_info.get('student_id', rfid),
                        'name': assignment_info.get('name', f'Student {rfid}'),
                        'type': 'student',
                        'course': assignment_info.get('course', 'Unknown'),
                        'gender': assignment_info.get('gender', 'Unknown'),
                        'timestamp': current_time,
                        'status': 'TIME-IN',
                        'action': 'ENTRY',
                        'violation': 'forgot_id',
                        'temporary_rfid': True,  # Flag to indicate this is temporary RFID (skip detection)
                        'skip_detection': True   # Explicit flag to skip detection in UI
                    }
                    self.update_main_screen_with_person(person_info)
                except Exception:
                    pass

            # Also add an ENTRY record to recent entries so the log shows both entry and exit
            try:
                entry_person_data = {
                    'id': assignment_info.get('student_id', rfid),
                    'name': assignment_info.get('name', f'Student {rfid}'),
                    'type': 'student',
                    'course': assignment_info.get('course', 'Unknown'),
                    'gender': assignment_info.get('gender', 'Unknown'),
                    'timestamp': current_time,
                    'status': 'TIME-IN',
                    'action': 'ENTRY',
                    'guard_id': self.current_guard_id or 'Unknown'
                }
                self.add_to_recent_entries(entry_person_data)
            except Exception:
                pass
            
            # Show success message
            self.show_green_success_message("Student Time-In (Forgot ID)", 
                              f"Student: {assignment_info['name']}\n"
                              f"Course: {assignment_info['course']}\n"
                              f"Student ID: {assignment_info['student_id']}\n"
                              f"Temporary RFID: {rfid}\n"
                              f"Time-In: {current_time}\n\n"
                              f"⚠️ VIOLATION: Forgot school ID\n"
                              f"Note: Uniform detection will start")
            
            # IMPORTANT: Do NOT start uniform detection for temporary RFID assignments.
            # This is a violation (forgot ID) and should be recorded but not processed by the detector.
            self.add_activity_log(f"NOTE: Temporary RFID used for student {assignment_info.get('name')} - detection skipped (violation)")
            
        except Exception as e:
            print(f"ERROR: Error handling student forgot ID time-in: {e}")
            self.add_activity_log(f"Error processing time-in for student RFID {rfid}: {e}")
    
    def handle_student_forgot_id_timeout(self, rfid, assignment_info):
        """Handle student forgot ID time-out - STOP DETECTION"""
        try:
            current_time = self.get_current_timestamp()
            
            # Stop detection for student exit
            if self.detection_active:
                self.stop_detection()
                self.add_activity_log("Detection stopped - Student exit (forgot ID)")

            # Restart guard camera feed (no detection) after student exit
            try:
                self.initialize_guard_camera_feed()
            except Exception:
                pass
            
            # Log activity
            self.add_activity_log(f"Student Time-Out (Forgot ID): {assignment_info['name']} (RFID: {rfid}) - {current_time}")
            
            # Finalize all pending violations when student exits
            self.finalize_session_violations(rfid)
            
            # Save to Firebase
            self.save_student_activity_to_firebase(rfid, 'time_out', current_time, assignment_info)
            
            # Update main screen with exit information
            if self.main_screen_window and self.main_screen_window.winfo_exists():
                # CRITICAL: Use assignment_info directly (it has all student info) instead of looking up again
                try:
                    exit_person_info = {
                        'id': assignment_info.get('student_id', rfid),
                        'name': assignment_info.get('name', f'Student {rfid}'),
                        'type': 'student',
                        'course': assignment_info.get('course', 'Unknown'),
                        'gender': assignment_info.get('gender', 'Unknown'),
                        'timestamp': current_time,
                        'status': 'TIME-OUT',
                        'action': 'EXIT',
                        'temporary_rfid': True,  # Flag to indicate this is temporary RFID
                        'skip_detection': True   # Explicit flag to skip detection in UI
                    }
                    # Ensure detection flags are NOT set
                    self.detection_active = False
                    self.uniform_detection_complete = False
                    # Update main screen with exit info
                    self.update_main_screen_with_person(exit_person_info)
                    print(f"✅ Main screen updated with student exit (temporary RFID): {assignment_info.get('name', 'Unknown')}")
                except Exception as e:
                    print(f"⚠️ Error updating main screen with exit: {e}")
                    # Fallback to the forgot-id specific update
                    try:
                        self.update_main_screen_with_student_forgot_id(assignment_info, 'time_out', current_time)
                    except Exception:
                        pass
            
            # Ensure recent entries shows this as EXIT
            try:
                person_data_exit = {
                    'id': assignment_info.get('student_id', rfid),
                    'name': assignment_info.get('name', f'Student {rfid}'),
                    'type': 'student',
                    'course': assignment_info.get('course', 'Unknown'),
                    'gender': assignment_info.get('gender', 'Unknown'),
                    'timestamp': current_time,
                    'status': 'TIME-OUT',
                    'action': 'EXIT',
                    'guard_id': self.current_guard_id or 'Unknown'
                }
                self.add_to_recent_entries(person_data_exit)
            except Exception:
                pass
            # Do NOT reset the temporary RFID assignment on exit; keep the temporary
            # RFID assigned for its full expiry period (24 hours). Update the in-memory
            # assignment record with last used time so it can be audited.
            try:
                if rfid in self.student_rfid_assignments:
                    self.student_rfid_assignments[rfid]['last_used'] = current_time
                    self.student_rfid_assignments[rfid]['status'] = 'active'
            except Exception:
                pass
            
            # Show success message
            self.show_green_success_message("Student Time-Out (Forgot ID)", 
                              f"Student: {assignment_info['name']}\n"
                              f"Course: {assignment_info['course']}\n"
                              f"Student ID: {assignment_info['student_id']}\n"
                              f"Temporary RFID: {rfid}\n"
                              f"Time-Out: {current_time}\n\n"
                              f"RFID: {rfid} - Now available for reuse\n"
                              f"Note: Detection stopped")
            
        except Exception as e:
            print(f"ERROR: Error handling student forgot ID time-out: {e}")
            self.add_activity_log(f"Error processing time-out for student RFID {rfid}: {e}")
    
    def reset_student_forgot_id_rfid(self, rfid):
        """Reset RFID card for reuse after student forgot ID exit"""
        try:
            # Remove from student RFID assignments
            if rfid in self.student_rfid_assignments:
                del self.student_rfid_assignments[rfid]
            
            # Add back to available RFID list
            if rfid not in self.student_forgot_rfids:
                self.student_forgot_rfids.append(rfid)
            
            # Update RFID availability in Firebase
            self.update_rfid_availability_in_firebase(rfid, available=True)
            
            # Update RFID dropdown
            self.update_student_rfid_list()
            
            print(f"SUCCESS: Student forgot ID RFID {rfid} reset and available for reuse")
            
        except Exception as e:
            print(f"ERROR: Error resetting student forgot ID RFID {rfid}: {e}")
    
    def save_student_activity_to_firebase(self, rfid, activity_type, timestamp, assignment_info, event_mode=False):
        """Save student forgot ID activity to Firebase"""
        try:
            if self.firebase_initialized and self.db:
                activity_data = {
                    'student_id': assignment_info['student_id'],
                    'name': assignment_info['name'],
                    'rfid': rfid,
                    'activity_type': activity_type,  # 'time_in' or 'time_out'
                    'timestamp': timestamp,
                    'course': assignment_info['course'],
                    'assignment_type': 'forgot_id',
                    'event_mode': event_mode  # Flag indicating event mode (detection bypassed)
                }
                
                # Save to student_activities collection (using student_id_timestamp format)
                student_id = assignment_info.get('student_id')
                formatted_timestamp = self.format_document_id(timestamp)
                doc_id = f"{student_id}_{formatted_timestamp}"
                activity_ref = self.db.collection('student_activities').document(doc_id)
                activity_ref.set(activity_data)
                
                # Also update/maintain student document in students collection
                if student_id:
                    student_ref = self.db.collection('students').document(student_id)
                    student_doc = student_ref.get()
                    
                    student_update_data = {
                        'student_id': student_id,
                        'name': assignment_info.get('name', 'Unknown'),
                        'course': assignment_info.get('course', 'Unknown'),
                        'last_updated': timestamp
                    }
                    
                    # Update time_in or time_out based on activity type
                    if activity_type == 'time_in':
                        student_update_data['time_in'] = timestamp
                        student_update_data['status'] = 'active'
                    elif activity_type == 'time_out':
                        student_update_data['time_out'] = timestamp
                        student_update_data['status'] = 'exited'
                    
                    # Set created_at only if document doesn't exist
                    if not student_doc.exists:
                        student_update_data['created_at'] = timestamp
                    
                    # Update or create student document
                    student_ref.set(student_update_data, merge=True)
                    print(f"SUCCESS: Student document updated in Firebase: {student_id}")
                
                print(f"SUCCESS: Student activity saved to Firebase: {activity_type} for {assignment_info['name']}")
            else:
                print("WARNING: Firebase not available - student activity not saved to cloud")
                
        except Exception as e:
            print(f"ERROR: Failed to save student activity to Firebase: {e}")
    
    def save_approved_violation_flag(self, student_id, rfid, timestamp):
        """Save approved violation flag to Firebase - valid for 24 hours"""
        try:
            if not self.firebase_initialized or not self.db:
                print("⚠️ WARNING: Firebase not available - cannot save approved violation flag")
                return False
            
            from datetime import datetime, timedelta
            
            # Calculate expiry time (24 hours from now) - use consistent datetime format
            # Parse the timestamp string to get datetime object, then add 24 hours
            try:
                # Try to parse the timestamp string
                if isinstance(timestamp, str):
                    # Parse timestamp (format: "YYYY-MM-DD HH:MM:SS")
                    approved_dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                else:
                    approved_dt = datetime.now()
            except (ValueError, TypeError):
                # If parsing fails, use current time
                approved_dt = datetime.now()
            
            # Add 24 hours
            expiry_dt = approved_dt + timedelta(hours=24)
            expiry_time = expiry_dt.strftime("%Y-%m-%d %H:%M:%S")
            
            # Also store as timestamp for easier comparison
            expiry_timestamp = expiry_dt.timestamp()
            
            # Create document in approved_violations collection
            # Use student_id as document ID for easy lookup
            approved_ref = self.db.collection('approved_violations').document(student_id)
            
            approved_data = {
                'student_id': student_id,
                'rfid': rfid,
                'approved_at': timestamp,
                'expiry_time': expiry_time,
                'expiry_timestamp': expiry_timestamp,  # Store as timestamp for easier comparison
                'status': 'active',
                'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Track when flag was created
            }
            
            approved_ref.set(approved_data)
            print(f"✅ Approved violation flag saved for student {student_id} (RFID: {rfid})")
            print(f"   - Approved at: {timestamp}")
            print(f"   - Expires at: {expiry_time}")
            print(f"   - Expiry timestamp: {expiry_timestamp}")
            return True
            
        except Exception as e:
            print(f"⚠️ ERROR: Failed to save approved violation flag: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def check_approved_violation_active(self, student_id, rfid=None):
        """Check if student has an active approved violation within 24 hours
        
        This function checks Firebase for an active approved violation flag.
        The flag persists across app restarts and is only deleted when it expires (after 24 hours).
        """
        try:
            if not self.firebase_initialized or not self.db:
                print(f"⚠️ DEBUG: Firebase not initialized - cannot check approved violation for {student_id}")
                return False
            
            from datetime import datetime
            
            print(f"🔍 DEBUG: Checking approved violation for student_id: {student_id}, rfid: {rfid}")
            
            # Check by student_id first (primary lookup)
            approved_ref = self.db.collection('approved_violations').document(student_id)
            approved_doc = approved_ref.get()
            
            print(f"🔍 DEBUG: Approved violation document exists (by student_id): {approved_doc.exists}")
            
            if not approved_doc.exists:
                # Also check by RFID if provided
                if rfid:
                    try:
                        rfid_query = self.db.collection('approved_violations').where('rfid', '==', rfid).where('status', '==', 'active').limit(1)
                        rfid_docs = rfid_query.get()
                        if rfid_docs:
                            approved_doc = rfid_docs[0]
                        else:
                            return False
                    except Exception as e:
                        print(f"⚠️ Warning: Could not check approved violations by RFID: {e}")
                        return False
                else:
                    return False
            
            approved_data = approved_doc.to_dict()
            if not approved_data:
                return False
            
            # Check if status is active
            if approved_data.get('status') != 'active':
                return False
            
            # Check if within expiry time - try timestamp first (more reliable), then fallback to string
            expiry_timestamp = approved_data.get('expiry_timestamp')
            expiry_time_str = approved_data.get('expiry_time')
            
            # Prefer timestamp comparison (more reliable)
            if expiry_timestamp:
                try:
                    import time
                    current_timestamp = time.time()
                    if current_timestamp < expiry_timestamp:
                        print(f"✅ Active approved violation found for student {student_id} (timestamp check)")
                        print(f"   - Current: {datetime.fromtimestamp(current_timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
                        print(f"   - Expires: {datetime.fromtimestamp(expiry_timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
                        return True
                    else:
                        print(f"⚠️ Approved violation expired for student {student_id} (timestamp check)")
                        print(f"   - Current: {datetime.fromtimestamp(current_timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
                        print(f"   - Expired at: {datetime.fromtimestamp(expiry_timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
                        # Clean up expired flag
                        try:
                            approved_ref.delete()
                            print(f"✅ Cleaned up expired approved violation flag for student {student_id}")
                        except Exception:
                            pass
                        return False
                except Exception as e:
                    print(f"⚠️ Warning: Could not compare expiry timestamp: {e}")
                    # Fall through to string comparison
            
            # Fallback to string comparison if timestamp not available
            if expiry_time_str:
                try:
                    expiry_time = datetime.strptime(expiry_time_str, "%Y-%m-%d %H:%M:%S")
                    current_time = datetime.now()
                    
                    if current_time < expiry_time:
                        print(f"✅ Active approved violation found for student {student_id} (string check) - expires at {expiry_time_str}")
                        return True
                    else:
                        print(f"⚠️ Approved violation expired for student {student_id} (string check) - expired at {expiry_time_str}")
                        # Clean up expired flag
                        try:
                            approved_ref.delete()
                            print(f"✅ Cleaned up expired approved violation flag for student {student_id}")
                        except Exception:
                            pass
                        return False
                except ValueError as e:
                    print(f"⚠️ Warning: Could not parse expiry time '{expiry_time_str}': {e}")
                    return False
            else:
                print(f"⚠️ Warning: No expiry_time or expiry_timestamp found in approved violation data")
                return False
            
        except Exception as e:
            print(f"⚠️ ERROR: Failed to check approved violation: {e}")
            return False
    
    def check_student_has_previous_violation(self, student_id):
        """Check if student has any previous violations in Firebase
        Returns True if student has at least one violation record, False otherwise"""
        try:
            if not self.firebase_initialized or not self.db:
                print(f"⚠️ DEBUG: Firebase not initialized - cannot check previous violations for {student_id}")
                return False
            
            # Check student_violations collection for this student
            violation_ref = self.db.collection('student_violations').document(student_id)
            violation_doc = violation_ref.get()
            
            if not violation_doc.exists:
                return False
            
            violation_data = violation_doc.to_dict()
            if not violation_data:
                return False
            
            # Check if violation_count exists and is greater than 0
            violation_count = violation_data.get('violation_count', 0)
            if violation_count > 0:
                print(f"✅ Student {student_id} has {violation_count} previous violation(s)")
                return True
            
            # Also check violation_history if it exists
            violation_history = violation_data.get('violation_history', [])
            if violation_history and len(violation_history) > 0:
                print(f"✅ Student {student_id} has {len(violation_history)} violation(s) in history")
                return True
            
            return False
            
        except Exception as e:
            print(f"⚠️ ERROR: Failed to check previous violations for {student_id}: {e}")
            return False
    
    def clear_all_approved_violations_on_startup(self):
        """Clear all approved violations on app startup - resets 24-hour timer"""
        try:
            if not self.firebase_initialized or not self.db:
                print("⚠️ WARNING: Firebase not available - cannot clear approved violations")
                return False
            
            print("🔄 Clearing all approved violations on app startup...")
            
            # Get all approved violations
            approved_ref = self.db.collection('approved_violations')
            all_approved = approved_ref.get()
            
            cleared_count = 0
            for doc in all_approved:
                try:
                    doc.reference.delete()
                    cleared_count += 1
                except Exception as e:
                    print(f"⚠️ Warning: Could not delete approved violation {doc.id}: {e}")
            
            print(f"✅ Cleared {cleared_count} approved violation(s) on app startup")
            print(f"ℹ️ 24-hour detection skip timer has been reset - all students will go through detection")
            return True
            
        except Exception as e:
            print(f"⚠️ ERROR: Failed to clear approved violations on startup: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def update_main_screen_with_student_forgot_id(self, assignment_info, activity_type, timestamp):
        """Update main screen with student forgot ID information"""
        try:
            if not self.main_screen_window or not self.main_screen_window.winfo_exists():
                return
            
            # This would update the main screen display
            # Implementation depends on your main screen structure
            print(f"📺 Main screen updated with student {activity_type}: {assignment_info['name']} (Forgot ID)")
            
        except Exception as e:
            print(f"ERROR: Error updating main screen with student forgot ID: {e}")
    
    def is_permanent_student_rfid(self, rfid):
        """Check if RFID belongs to a permanent student"""
        try:
            print(f"🔍 DEBUG is_permanent_student_rfid: Checking RFID = {rfid}")
            if self.firebase_initialized and self.db:
                students_ref = self.db.collection('students')
                
                # Method 1: Try query by 'rfid' field
                try:
                    query = students_ref.where('rfid', '==', rfid)
                    docs = query.get()
                    if len(docs) > 0:
                        print(f"✅ DEBUG is_permanent_student_rfid: Found via rfid field query ({len(docs)} docs)")
                        return True
                except Exception as e:
                    print(f"⚠️ DEBUG is_permanent_student_rfid: Query by rfid field failed: {e}")
                
                # Method 2: Try direct document lookup (document ID might be the RFID)
                try:
                    doc_ref = students_ref.document(rfid)
                    doc = doc_ref.get()
                    if doc.exists:
                        print(f"✅ DEBUG is_permanent_student_rfid: Found via direct document lookup (ID = {rfid})")
                        return True
                except Exception as e:
                    print(f"⚠️ DEBUG is_permanent_student_rfid: Direct document lookup failed: {e}")
                
                # Method 3: Try querying all students and check RFID field (fallback)
                try:
                    all_students = students_ref.limit(100).get()
                    for doc in all_students:
                        doc_data = doc.to_dict()
                        if doc_data.get('rfid') == rfid or doc_data.get('RFID') == rfid:
                            print(f"✅ DEBUG is_permanent_student_rfid: Found via scan (document ID: {doc.id})")
                            return True
                except Exception as e:
                    print(f"⚠️ DEBUG is_permanent_student_rfid: Scan method failed: {e}")
                
                print(f"❌ DEBUG is_permanent_student_rfid: Not found via any method")
                return False
            else:
                # Fallback: check known permanent student RFID
                result = rfid == '0095365253'
                print(f"🔍 DEBUG is_permanent_student_rfid: Firebase not initialized, fallback result = {result}")
                return result
        except Exception as e:
            print(f"ERROR: Error checking permanent student RFID: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def check_student_has_entry_record(self, rfid):
        """Check if student has an entry record in Firebase that is recent (within last 24 hours)"""
        from datetime import datetime
        
        try:
            if not self.firebase_initialized:
                print(f"🔍 DEBUG: Firebase not initialized - returning False for RFID {rfid}")
                return False
                
            # Query Firebase for recent entry records for this student
            # Prefer self.db (firestore client) if available
            db = self.db if getattr(self, 'db', None) else (getattr(self, 'firebase_admin', None) and getattr(self.firebase_admin, 'firestore', None) and self.firebase_admin.firestore.client())
            if db is None:
                print("WARNING: No Firestore client available for entry record check")
                return False
            
            # Get current time and calculate threshold (start of today)
            now = datetime.now()
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            threshold_time = today_start
            
            print(f"🔍 DEBUG check_student_has_entry_record: Checking RFID {rfid}")
            print(f"   Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Today start: {today_start.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Use firestore.Query constants for ordering
            try:
                order_desc = firestore.Query.DESCENDING
            except Exception:
                order_desc = None

            query = db.collection('student_activities').where('rfid', '==', rfid).where('activity_type', '==', 'time_in')
            try:
                if order_desc is not None:
                    student_activities = query.order_by('timestamp', direction=order_desc).limit(10).get()  # Get more to filter by date
                else:
                    student_activities = query.limit(10).get()
            except Exception as e:
                # Firestore may require a composite index for this query; fallback to fetching a batch
                print(f"WARNING: Order query failed (may need composite index): {e}")
                try:
                    docs = query.limit(50).get()
                    # sort locally by timestamp field (descending)
                    docs_list = list(docs)
                    docs_list.sort(key=lambda d: d.to_dict().get('timestamp', None) or 0, reverse=True)
                    student_activities = docs_list[:10] if docs_list else []
                except Exception as e2:
                    print(f"ERROR: Fallback query also failed: {e2}")
                    student_activities = []
            
            print(f"🔍 DEBUG: Found {len(student_activities)} time_in records for RFID {rfid}")
            
            if student_activities:
                # Find the most recent entry that is from today
                for entry_doc in student_activities:
                    latest_entry = entry_doc.to_dict()
                    entry_time_str = latest_entry.get('timestamp')
                    
                    if entry_time_str:
                        # Parse timestamp string (format: "YYYY-MM-DD HH:MM:SS")
                        try:
                            entry_time = datetime.strptime(entry_time_str, "%Y-%m-%d %H:%M:%S")
                            
                            # CRITICAL: Only consider entries from today (ignore old entries)
                            # Compare dates to ensure it's from today
                            if entry_time.date() < now.date():
                                print(f"🔍 DEBUG: Entry record found but from different date (entry: {entry_time_str}, today: {now.strftime('%Y-%m-%d')}), ignoring - treating as no entry")
                                continue  # Try next entry - this one is from a different day
                            
                            if entry_time < threshold_time:
                                print(f"🔍 DEBUG: Entry record found but before today start (from {entry_time_str}), ignoring - treating as no entry")
                                continue  # Try next entry
                            
                            print(f"🔍 DEBUG: Found entry record from today: {entry_time_str}")
                            
                            # Check if there's a corresponding exit record after this entry
                            try:
                                exit_activities = db.collection('student_activities').where('rfid', '==', rfid).where('activity_type', '==', 'time_out').where('timestamp', '>', entry_time_str).limit(1).get()
                            except Exception as e_exit:
                                # Fallback: fetch a batch and filter locally by timestamp
                                print(f"WARNING: Exit query failed (may need composite index): {e_exit}")
                                try:
                                    exits = db.collection('student_activities').where('rfid', '==', rfid).where('activity_type', '==', 'time_out').limit(50).get()
                                    exits_list = [d.to_dict() for d in exits]
                                    # Parse timestamps for proper comparison
                                    exit_activities = []
                                    for e in exits_list:
                                        exit_ts = e.get('timestamp')
                                        if exit_ts:
                                            try:
                                                exit_time = datetime.strptime(exit_ts, "%Y-%m-%d %H:%M:%S")
                                                if exit_time > entry_time:
                                                    exit_activities.append(e)
                                                    break
                                            except ValueError:
                                                pass
                                except Exception:
                                    exit_activities = []

                            if not exit_activities or len(exit_activities) == 0:
                                # No exit record found after entry - student is still inside (today)
                                print(f"✅ DEBUG: Student has active entry from today ({entry_time_str}) - no exit record found after it")
                                return True
                            else:
                                # Has exit record - student is outside
                                exit_time_str = exit_activities[0].get('timestamp') if exit_activities else 'Unknown'
                                print(f"🔍 DEBUG: Student has entry from today ({entry_time_str}) but also has exit record ({exit_time_str}) - treating as outside")
                                return False
                                
                        except ValueError as e:
                            print(f"⚠️ WARNING: Could not parse timestamp '{entry_time_str}': {e}")
                            continue  # Try next entry
                        
            print(f"🔍 DEBUG: No entry records from today found for RFID {rfid} - treating as outside (TIME-IN required)")
            return False
            
        except Exception as e:
            print(f"❌ Error checking student entry record: {e}")
            import traceback
            traceback.print_exc()
            # On error, default to False (no entry record) so student can TIME-IN
            return False
    
    def delete_most_recent_entry_record(self, rfid):
        """Delete the most recent entry record for a student from Firebase
        This is used when guard cancels entry after it was recorded"""
        try:
            if not self.firebase_initialized or not self.db:
                print(f"⚠️ WARNING: Firebase not initialized - cannot delete entry record for RFID {rfid}")
                return False
            
            from datetime import datetime
            
            # Query Firebase for the most recent entry record for this RFID
            query = self.db.collection('student_activities').where('rfid', '==', rfid).where('activity_type', '==', 'time_in')
            
            try:
                # Try to order by timestamp descending
                order_desc = firestore.Query.DESCENDING
                entry_docs = query.order_by('timestamp', direction=order_desc).limit(1).get()
            except Exception as e:
                # Fallback: fetch a batch and sort locally
                print(f"WARNING: Order query failed (may need composite index): {e}")
                try:
                    docs = query.limit(50).get()
                    docs_list = list(docs)
                    if docs_list:
                        # Sort locally by timestamp (descending)
                        docs_list.sort(key=lambda d: d.to_dict().get('timestamp', ''), reverse=True)
                        entry_docs = docs_list[:1]
                    else:
                        entry_docs = []
                except Exception as e2:
                    print(f"ERROR: Fallback query also failed: {e2}")
                    entry_docs = []
            
            if entry_docs:
                # Get the most recent entry document
                most_recent_doc = entry_docs[0]
                entry_data = most_recent_doc.to_dict()
                entry_time_str = entry_data.get('timestamp', '')
                
                # Check if entry is from today (only delete today's entries)
                if entry_time_str:
                    try:
                        entry_time = datetime.strptime(entry_time_str, "%Y-%m-%d %H:%M:%S")
                        now = datetime.now()
                        
                        # Only delete if entry is from today
                        if entry_time.date() == now.date():
                            # Delete the document
                            most_recent_doc.reference.delete()
                            print(f"✅ Deleted most recent entry record for RFID {rfid} (timestamp: {entry_time_str})")
                            return True
                        else:
                            print(f"⚠️ Most recent entry is not from today ({entry_time_str}) - not deleting")
                            return False
                    except ValueError as e:
                        print(f"⚠️ WARNING: Could not parse timestamp '{entry_time_str}': {e}")
                        return False
                else:
                    print(f"⚠️ WARNING: Entry record has no timestamp - not deleting")
                    return False
            else:
                print(f"⚠️ WARNING: No entry records found for RFID {rfid}")
                return False
                
        except Exception as e:
            print(f"❌ ERROR: Failed to delete entry record for RFID {rfid}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def check_teacher_has_entry_record(self, rfid):
        """Check if teacher has an entry record in Firebase that is recent (within last 24 hours)"""
        from datetime import datetime
        
        try:
            if not self.firebase_initialized:
                print(f"🔍 DEBUG: Firebase not initialized - returning False for teacher RFID {rfid}")
                return False
                
            # Query Firebase for recent entry records for this teacher
            # Prefer self.db (firestore client) if available
            db = self.db if getattr(self, 'db', None) else (getattr(self, 'firebase_admin', None) and getattr(self.firebase_admin, 'firestore', None) and self.firebase_admin.firestore.client())
            if db is None:
                print("WARNING: No Firestore client available for teacher entry record check")
                return False
            
            # Get current time and calculate threshold (start of today)
            now = datetime.now()
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            threshold_time = today_start
            
            print(f"🔍 DEBUG check_teacher_has_entry_record: Checking RFID {rfid}")
            print(f"   Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Today start: {today_start.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Use firestore.Query constants for ordering
            try:
                order_desc = firestore.Query.DESCENDING
            except Exception:
                order_desc = None

            query = db.collection('teacher_activities').where('rfid', '==', rfid).where('activity_type', '==', 'time_in')
            try:
                if order_desc is not None:
                    teacher_activities = query.order_by('timestamp', direction=order_desc).limit(10).get()  # Get more to filter by date
                else:
                    teacher_activities = query.limit(10).get()
            except Exception as e:
                # Firestore may require a composite index for this query; fallback to fetching a batch
                print(f"WARNING: Order query failed (may need composite index): {e}")
                try:
                    docs = query.limit(50).get()
                    # sort locally by timestamp field (descending)
                    docs_list = list(docs)
                    docs_list.sort(key=lambda d: d.to_dict().get('timestamp', None) or 0, reverse=True)
                    teacher_activities = docs_list[:10] if docs_list else []
                except Exception as e2:
                    print(f"ERROR: Fallback query also failed: {e2}")
                    teacher_activities = []
            
            print(f"🔍 DEBUG: Found {len(teacher_activities)} time_in records for teacher RFID {rfid}")
            
            if teacher_activities:
                # Find the most recent entry that is from today
                for entry_doc in teacher_activities:
                    latest_entry = entry_doc.to_dict()
                    entry_time_str = latest_entry.get('timestamp')
                    
                    if entry_time_str:
                        # Parse timestamp string (format: "YYYY-MM-DD HH:MM:SS")
                        try:
                            entry_time = datetime.strptime(entry_time_str, "%Y-%m-%d %H:%M:%S")
                            
                            # CRITICAL: Only consider entries from today (ignore old entries)
                            # Compare dates to ensure it's from today
                            if entry_time.date() < now.date():
                                print(f"🔍 DEBUG: Entry record found but from different date (entry: {entry_time_str}, today: {now.strftime('%Y-%m-%d')}), ignoring - treating as no entry")
                                continue  # Try next entry - this one is from a different day
                            
                            if entry_time < threshold_time:
                                print(f"🔍 DEBUG: Entry record found but before today start (from {entry_time_str}), ignoring - treating as no entry")
                                continue  # Try next entry
                            
                            print(f"🔍 DEBUG: Found entry record from today: {entry_time_str}")
                            
                            # Check if there's a corresponding exit record after this entry
                            try:
                                exit_activities = db.collection('teacher_activities').where('rfid', '==', rfid).where('activity_type', '==', 'time_out').where('timestamp', '>', entry_time_str).limit(1).get()
                            except Exception as e_exit:
                                # Fallback: fetch a batch and filter locally by timestamp
                                print(f"WARNING: Exit query failed (may need composite index): {e_exit}")
                                try:
                                    exits = db.collection('teacher_activities').where('rfid', '==', rfid).where('activity_type', '==', 'time_out').limit(50).get()
                                    exits_list = [d.to_dict() for d in exits]
                                    # Parse timestamps for proper comparison
                                    exit_activities = []
                                    for e in exits_list:
                                        exit_ts = e.get('timestamp')
                                        if exit_ts:
                                            try:
                                                exit_time = datetime.strptime(exit_ts, "%Y-%m-%d %H:%M:%S")
                                                if exit_time > entry_time:
                                                    exit_activities.append(e)
                                                    break
                                            except ValueError:
                                                pass
                                except Exception:
                                    exit_activities = []
                            
                            if not exit_activities:
                                # No exit record found after entry - teacher is still inside (today)
                                print(f"✅ DEBUG: Teacher has active entry from today ({entry_time_str}) - no exit record found")
                                return True
                            else:
                                # Has exit record - teacher is outside
                                print(f"🔍 DEBUG: Teacher has entry from today but also has exit record - treating as outside")
                                return False
                                
                        except ValueError as e:
                            print(f"⚠️ WARNING: Could not parse timestamp '{entry_time_str}': {e}")
                            continue  # Try next entry
                        
            print(f"🔍 DEBUG: No entry records from today found for teacher RFID {rfid} - treating as outside (TIME-IN required)")
            return False
            
        except Exception as e:
            print(f"❌ Error checking teacher entry record: {e}")
            import traceback
            traceback.print_exc()
            # On error, default to False (no entry record) so teacher can TIME-IN
            return False
    
    def check_person_has_entry_record(self, person_id, person_type):
        """Check if person (student/visitor/teacher) has an entry record in Firebase that is recent (from today)"""
        from datetime import datetime
        
        try:
            if not self.firebase_initialized:
                return False
            
            db = self.db if getattr(self, 'db', None) else (getattr(self, 'firebase_admin', None) and getattr(self.firebase_admin, 'firestore', None) and self.firebase_admin.firestore.client())
            if db is None:
                return False
            
            # Get current time and calculate threshold (start of today)
            now = datetime.now()
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            threshold_time = today_start
            
            # Determine collection and activity type based on person type
            if person_type.lower() == 'student':
                collection = 'student_activities'
                activity_type = 'time_in'
            elif person_type.lower() == 'visitor':
                collection = 'visitor_activities'
                activity_type = 'time_in'
            elif person_type.lower() == 'teacher':
                # Teachers might use teacher_activities or similar - adjust as needed
                collection = 'teacher_activities'
                activity_type = 'time_in'
            else:
                return False
            
            # Use person_id as RFID for query
            query = db.collection(collection).where('rfid', '==', person_id).where('activity_type', '==', activity_type)
            
            try:
                order_desc = firestore.Query.DESCENDING
                activities = query.order_by('timestamp', direction=order_desc).limit(1).get()
            except Exception:
                # Fallback without ordering
                try:
                    docs = query.limit(50).get()
                    docs_list = list(docs)
                    docs_list.sort(key=lambda d: d.to_dict().get('timestamp', None) or 0, reverse=True)
                    activities = docs_list[:1] if docs_list else []
                except Exception:
                    activities = []
            
            if activities:
                # Find the most recent entry that is from today
                for entry_doc in activities:
                    latest_entry = entry_doc.to_dict()
                    entry_time_str = latest_entry.get('timestamp')
                    
                    if entry_time_str:
                        # Parse timestamp string (format: "YYYY-MM-DD HH:MM:SS")
                        try:
                            entry_time = datetime.strptime(entry_time_str, "%Y-%m-%d %H:%M:%S")
                            
                            # CRITICAL: Only consider entries from today (ignore old entries)
                            if entry_time < threshold_time:
                                print(f"🔍 DEBUG: Entry record found but too old (from {entry_time_str}), ignoring - treating as no entry")
                                return False  # Old entry - treat as if person is outside
                            
                            print(f"🔍 DEBUG: Found recent entry record from {entry_time_str} (today) for {person_type}")
                            
                            # Check if there's a corresponding exit record after this entry
                            exit_activity_type = 'time_out'
                            try:
                                exit_query = db.collection(collection).where('rfid', '==', person_id).where('activity_type', '==', exit_activity_type).where('timestamp', '>', entry_time_str).limit(1)
                                exit_activities = exit_query.get()
                            except Exception:
                                # Fallback: fetch batch and filter locally
                                try:
                                    exits = db.collection(collection).where('rfid', '==', person_id).where('activity_type', '==', exit_activity_type).limit(50).get()
                                    exits_list = [d.to_dict() for d in exits]
                                    exit_activities = [e for e in exits_list if e.get('timestamp') and e.get('timestamp') > entry_time_str]
                                except Exception:
                                    exit_activities = []
                            
                            if not exit_activities:
                                # No exit record found after entry - person is still inside (today)
                                print(f"✅ DEBUG: {person_type} has active entry from today ({entry_time_str}) - no exit record found")
                                return True
                            else:
                                # Has exit record - person is outside
                                print(f"🔍 DEBUG: {person_type} has entry from today but also has exit record - treating as outside")
                                return False
                                
                        except ValueError as e:
                            print(f"⚠️ WARNING: Could not parse timestamp '{entry_time_str}': {e}")
                            continue  # Try next entry
            
            print(f"🔍 DEBUG: No recent entry records found for {person_type} {person_id} - treating as outside")
            return False
            
        except Exception as e:
            print(f"❌ Error checking person entry record: {e}")
            return False
    
    def handle_permanent_student_rfid_tap(self, rfid):
        """Handle permanent student RFID tap quickly without blocking UI/camera."""
        try:
            print(f"🔍 DEBUG handle_permanent_student_rfid_tap: Called with RFID = {rfid}")
            import threading

            def _worker():
                try:
                    print(f"🔍 DEBUG: RFID tap worker thread started for RFID: {rfid}")
                    # Ensure Firebase is ready
                    if not self.firebase_initialized:
                        print("INFO: Firebase not ready for permanent student RFID tap, initializing...")
                        try:
                            self.init_firebase()
                        except Exception as e:
                            print(f"WARNING: Firebase init error: {e}")
                        # If still not ready, retry soon without blocking
                        if not self.firebase_initialized:
                            print("WARNING: Firebase still not ready, retrying in 300 ms...")
                            try:
                                self.root.after(300, lambda: self.handle_permanent_student_rfid_tap(rfid))
                            except Exception:
                                pass
                            return

                    # Check if RFID is assigned to a visitor first
                    visitor_info = None
                    if hasattr(self, 'visitor_rfid_assignments') and rfid in getattr(self, 'visitor_rfid_assignments', {}):
                        visitor_info = self.visitor_rfid_assignments[rfid]
                    elif rfid in getattr(self, 'visitor_rfid_registry', {}):
                        visitor_info = self.visitor_rfid_registry[rfid]
                    
                    # Also check Firebase visitors collection
                    if not visitor_info and self.firebase_initialized and self.db:
                        try:
                            visitors_ref = self.db.collection('visitors')
                            visitors_query = visitors_ref.where('rfid', '==', rfid).limit(1)
                            visitors = visitors_query.get()
                            if visitors:
                                for doc in visitors:
                                    visitor_data = doc.to_dict()
                                    status = visitor_data.get('status', '')
                                    time_out = visitor_data.get('time_out')
                                    
                                    # Skip if visitor has exited (status is 'exited' and has time_out)
                                    if status == 'exited' and time_out:
                                        print(f"🔍 DEBUG: RFID {rfid} visitor has exited (status: {status}, time_out: {time_out}), skipping")
                                        continue
                                    
                                    # Only consider active visitors (registered or active status, or has time_in without time_out)
                                    if status not in ('registered', 'active') and not (visitor_data.get('time_in') and not time_out):
                                        print(f"🔍 DEBUG: RFID {rfid} visitor status is '{status}' (not active), skipping")
                                        continue
                                    
                                    # Check if assignment is still valid (not expired)
                                    expiry_time_str = visitor_data.get('expiry_time', '')
                                    if expiry_time_str:
                                        try:
                                            expiry_time = datetime.datetime.strptime(expiry_time_str, "%Y-%m-%d %H:%M:%S")
                                            if datetime.datetime.now() >= expiry_time:
                                                print(f"🔍 DEBUG: RFID {rfid} visitor assignment expired (expiry: {expiry_time_str}), skipping")
                                                continue
                                        except Exception:
                                            # If expiry parsing fails, assume valid
                                            pass
                                    
                                    # Valid active visitor found
                                    visitor_info = visitor_data
                                    break
                        except Exception as e:
                            print(f"⚠️ Error checking Firebase visitors for RFID {rfid}: {e}")
                    
                    # If RFID is assigned to a visitor, handle as visitor
                    if visitor_info:
                        print(f"🔍 DEBUG: RFID {rfid} is assigned to visitor: {visitor_info.get('name', 'Unknown')}")
                        self.handle_visitor_rfid_tap(rfid)
                        return

                    # Check if RFID belongs to a teacher
                    teacher_info = None
                    try:
                        teacher_info = self.get_teacher_info_by_rfid(rfid)
                        if teacher_info:
                            print(f"🔍 DEBUG: RFID {rfid} is assigned to teacher: {teacher_info.get('name', 'Unknown')}")
                            self.handle_teacher_rfid_tap(rfid)
                            return
                    except Exception as e:
                        print(f"⚠️ Error checking teacher for RFID {rfid}: {e}")

                    # Get student info from Firebase
                    print(f"🔍 DEBUG: Getting student info from Firebase for RFID: {rfid}")
                    student_info = self.get_student_info_by_rfid(rfid)
                    print(f"🔍 DEBUG: Got student_info: {student_info is not None}, keys: {list(student_info.keys()) if student_info else 'None'}")

                    if not student_info:
                        self.add_activity_log(f"Unknown permanent student RFID tapped: {rfid}")
                        try:
                            self.root.after(0, lambda: messagebox.showwarning(
                                "Unknown RFID", f"RFID {rfid} is not assigned to any permanent student, teacher, or visitor."))
                        except Exception:
                            pass
                        return

                    # Check if detection is currently active for this student
                    current_rfid_val = getattr(getattr(self, 'detection_system', None), 'detection_service', None)
                    current_rfid_val = getattr(current_rfid_val, 'current_rfid', None) if current_rfid_val else None

                    print(f"🔍 RFID Tap Debug:")
                    print(f"   Tapped RFID: {rfid}")
                    print(f"   Current detection RFID: {current_rfid_val}")
                    print(f"   Detection active: {getattr(self, 'detection_active', False)}")
                    print(f"   Event Mode active: {getattr(self, 'event_mode_active', False)}")

                    # CRITICAL: Check Event Mode FIRST - if active, bypass all detection logic
                    if getattr(self, 'event_mode_active', False):
                        print(f"🎉 Event Mode active in RFID tap handler - bypassing detection check")
                        # Stop any running detection immediately
                        if hasattr(self, 'detection_system') and self.detection_system:
                            try:
                                self.detection_system.stop_detection()
                                print("🛑 Stopped detection system in RFID tap handler (Event Mode)")
                            except Exception as e:
                                print(f"⚠️ Warning: Could not stop detection: {e}")
                        self.detection_active = False
                        # Hide requirements section
                        try:
                            self.hide_requirements_section()
                        except Exception:
                            pass

                    # CRITICAL: Ensure detection is fully stopped before checking entry record
                    # This prevents detection from restarting if entry was saved via APPROVE button
                    try:
                        # Aggressively stop any running detection
                        if hasattr(self, 'detection_system') and self.detection_system:
                            self.detection_system.stop_detection()
                            print(f"🛑 Stopped detection system before entry record check")
                        
                        # Reset detection flags to ensure clean state
                        self.detection_active = False
                        self.uniform_detection_complete = False
                        
                        # Small delay to ensure detection fully stops
                        import time
                        time.sleep(0.2)
                        
                        print(f"✅ Detection fully stopped and flags reset before entry check")
                    except Exception as e:
                        print(f"⚠️ WARNING: Could not stop detection before entry check: {e}")
                    
                    # CRITICAL: Check if student has entry record FIRST (before any retry logic)
                    # First check local cache (immediate, no Firebase delay), then check Firebase
                    import time
                    has_entry_record = False
                    
                    # Check local cache first (prevents race condition)
                    if hasattr(self, 'recent_entry_cache') and rfid in self.recent_entry_cache:
                        cache_timestamp = self.recent_entry_cache[rfid]
                        # Cache is valid for 5 minutes (300 seconds)
                        if time.time() - cache_timestamp < 300:
                            has_entry_record = True
                            print(f"✅ Entry record found in local cache for RFID {rfid}")
                        else:
                            # Cache expired, remove it
                            del self.recent_entry_cache[rfid]
                            print(f"⏰ Cache entry expired for RFID {rfid}, removed from cache")
                    
                    # If not in cache, check Firebase
                    if not has_entry_record:
                        has_entry_record = self.check_student_has_entry_record(rfid)
                        # If found in Firebase, add to cache
                        if has_entry_record:
                            if not hasattr(self, 'recent_entry_cache'):
                                self.recent_entry_cache = {}
                            self.recent_entry_cache[rfid] = time.time()
                            print(f"✅ Entry record found in Firebase for RFID {rfid}, added to cache")
                    
                    print(f"🔍 DEBUG: Checking entry record for RFID {rfid}")
                    print(f"   Has entry record: {has_entry_record}")
                    print(f"   Event Mode: {getattr(self, 'event_mode_active', False)}")
                    print(f"   Current detection RFID: {current_rfid_val}")
                    print(f"   Detection active flag: {getattr(self, 'detection_active', False)}")

                    if has_entry_record:
                        # Student is inside - this is EXIT (CRITICAL: stop any running detection first)
                        # If entry record exists, ALWAYS treat as EXIT - don't allow detection retry
                        # This prevents detection from starting again after entry is saved
                        print(f"🎯 Student {student_info.get('name', 'Unknown')} is inside - processing EXIT")
                        print(f"   Entry record found - treating as EXIT (no detection retry allowed)")
                        
                        # CRITICAL: Stop any running detection before processing exit
                        try:
                            if hasattr(self, 'detection_system') and self.detection_system:
                                self.detection_system.stop_detection()
                                print(f"🛑 Stopped detection system before processing EXIT")
                            self.detection_active = False
                            self.uniform_detection_complete = False
                        except Exception as e:
                            print(f"⚠️ Warning: Could not stop detection before EXIT: {e}")
                        
                        self.handle_permanent_student_timeout(rfid, student_info)
                        return  # Exit early - don't start detection
                    
                    # If detection is active for the same student, treat as retry (only if Event Mode is OFF and NO entry record)
                    if not getattr(self, 'event_mode_active', False):
                        if current_rfid_val == rfid and getattr(self, 'detection_active', False):
                            print(f"🔄 Same student tapping again - treating as retry (no entry record found)")
                            self.handle_permanent_student_timein(rfid, student_info)
                            return

                    # Student is outside - this is ENTRY (or retry)
                    print(f"🎯 Student {student_info.get('name', 'Unknown')} is outside - processing ENTRY/RETRY")
                    print(f"🔍 DEBUG: About to call handle_permanent_student_timein with Event Mode = {getattr(self, 'event_mode_active', False)}")
                    try:
                        print(f"🔍 DEBUG: Calling handle_permanent_student_timein NOW...")
                        self.handle_permanent_student_timein(rfid, student_info)
                        print(f"🔍 DEBUG: handle_permanent_student_timein returned")
                    except Exception as timein_error:
                        print(f"❌ ERROR in handle_permanent_student_timein: {timein_error}")
                        import traceback
                        traceback.print_exc()
                        self.add_activity_log(f"Error calling handle_permanent_student_timein: {timein_error}")
                except Exception as e:
                    print(f"ERROR: Error handling permanent student RFID tap (worker): {e}")
                    self.add_activity_log(f"Error handling permanent student RFID {rfid}: {e}")

            threading.Thread(target=_worker, daemon=True).start()
        except Exception as e:
            print(f"ERROR: Error scheduling RFID tap handling: {e}")
            self.add_activity_log(f"Error scheduling permanent student RFID {rfid}: {e}")
    
    def _ensure_student_number_from_firebase(self, student_info, rfid):
        """Helper function to ensure Student Number is present in student_info.
        Fetches from Firebase if missing or invalid (e.g., equals RFID).
        Returns updated student_info with guaranteed Student Number and RFID."""
        try:
            if not rfid:
                print(f"⚠️ WARNING: No RFID provided to _ensure_student_number_from_firebase")
                return student_info
            
            # Ensure RFID is in student_info before processing
            if not student_info.get('rfid'):
                student_info['rfid'] = rfid
            
            # Check if student_info has valid Student Number
            student_num = student_info.get('student_id') or student_info.get('student_number')
            
            # Validate: Student Number should not be RFID, 'Unknown', None, or empty
            is_valid = (student_num and 
                       student_num != 'Unknown' and 
                       str(student_num).strip() != '' and
                       str(student_num).strip() != str(rfid).strip())
            
            if not is_valid:
                # Fetch fresh data from Firebase
                print(f"🔍 Student Number missing or invalid, fetching from Firebase for RFID: {rfid}")
                full_student_info = self.get_student_info_by_rfid(rfid)
                if full_student_info:
                    student_num_from_firebase = full_student_info.get('student_id') or full_student_info.get('student_number')
                    if student_num_from_firebase and str(student_num_from_firebase).strip() != str(rfid).strip():
                        # Update student_info with complete Firebase data
                        student_info.update(full_student_info)
                        # CRITICAL: Explicitly preserve RFID in case update overwrote it
                        student_info['rfid'] = rfid
                        print(f"✅ Updated student_info with Student Number from Firebase: {student_num_from_firebase}")
                        return student_info
                    else:
                        print(f"⚠️ WARNING: Fetched Student Number from Firebase equals RFID: {student_num_from_firebase}")
                else:
                    print(f"⚠️ WARNING: Could not fetch Student Number from Firebase for RFID: {rfid}")
            else:
                print(f"✅ Student Number already valid: {student_num}")
                # Ensure RFID is still present even if Student Number was already valid
                student_info['rfid'] = rfid
            
            return student_info
        except Exception as e:
            print(f"⚠️ Error in _ensure_student_number_from_firebase: {e}")
            import traceback
            traceback.print_exc()
            # Ensure RFID is preserved even in error case
            if rfid and student_info:
                student_info['rfid'] = rfid
            return student_info
    
    def get_student_info_by_rfid(self, rfid):
        """Get student information by RFID (using document ID) - Enhanced with better error handling"""
        try:
            print(f"🔍 ========== FIREBASE LOOKUP START ==========")
            print(f"🔍 Looking up student with RFID: {rfid}")
            print(f"🔍 Firebase initialized: {self.firebase_initialized}")
            print(f"🔍 Database available: {self.db is not None}")
            
            # Try Firebase first
            if self.firebase_initialized and self.db:
                try:
                    # In your Firebase, RFID is stored as document ID, not as a field
                    students_ref = self.db.collection('students')
                    doc_ref = students_ref.document(rfid)  # Use RFID as document ID
                    print(f"🔍 Querying Firebase document ID: {rfid}")
                    doc = doc_ref.get()
                
                    if doc.exists:
                        student_data = doc.to_dict()
                        print(f"✅ SUCCESS: Found student document by RFID {rfid}: {student_data.get('Name', 'Unknown')}")
                        print(f"🔍 ALL AVAILABLE FIELDS IN FIREBASE: {list(student_data.keys())}")
                        print(f"🔍 Field values: {[(k, v) for k, v in student_data.items()]}")
                        
                        # CRITICAL: Explicitly check for "Student Number" field first
                        if 'Student Number' in student_data:
                            student_num_raw = student_data.get('Student Number')
                            print(f"🔍 Found 'Student Number' field in Firebase. Raw value: {repr(student_num_raw)} (type: {type(student_num_raw)})")
                        else:
                            print(f"⚠️ 'Student Number' field NOT found in Firebase document")

                        # Determine Senior High / Strand fields if present
                        senior_high = None
                        for key in ('Senior High School', 'senior_high_school', 'senior_high', 'Strand', 'strand'):
                            if key in student_data and student_data.get(key):
                                senior_high = str(student_data.get(key)).strip()
                                break

                        if senior_high:
                            course_val = f"SHS {senior_high}"
                        else:
                            course_val = student_data.get('Course', student_data.get('Department', 'Unknown Course'))

                        gender_val = student_data.get('Gender', student_data.get('gender', 'Unknown'))

                        # Return properly formatted student info
                        # IMPORTANT: Always include student_id (Student Number) and rfid field
                        # Strip whitespace from course if it exists
                        if course_val:
                            course_val = str(course_val).strip()
                        
                        # Get Student Number from Firebase - check multiple field name variations
                        student_number = None
                        field_names = ['Student Number', 'student_number', 'StudentNumber', 'student_id', 'Student_Id', 'STUDENT_NUMBER']
                        
                        print(f"🔍 Starting Student Number retrieval process...")
                        print(f"🔍 Checking field names in order: {field_names}")
                        
                        # First pass: exact match (prioritize "Student Number" with space)
                        for field_name in field_names:
                            print(f"🔍 Checking field: '{field_name}' - Present in document: {field_name in student_data}")
                            if field_name in student_data:
                                potential_value = student_data.get(field_name)
                                print(f"🔍   Raw value from '{field_name}': {repr(potential_value)} (type: {type(potential_value)})")
                                if potential_value is not None:
                                    student_number = str(potential_value).strip()
                                    print(f"🔍   After str() and strip(): {repr(student_number)}")
                                    # Only accept if it's not empty and not equal to RFID
                                    if student_number and student_number != '' and student_number != rfid:
                                        print(f"✅ SUCCESS: Found Student Number in field '{field_name}': {student_number}")
                                        break
                                    else:
                                        # If not valid, continue to next field
                                        reason = "empty" if not student_number or student_number == '' else "equals RFID"
                                        print(f"⚠️   Rejected value from '{field_name}': {reason} (value: {repr(student_number)}, RFID: {repr(rfid)})")
                                        student_number = None
                                else:
                                    print(f"⚠️   Field '{field_name}' exists but value is None")
                        
                        # Second pass: case-insensitive search if exact match failed
                        if not student_number:
                            print(f"⚠️ Student Number not found with exact match. Trying case-insensitive search...")
                            print(f"⚠️ Available fields in Firebase: {list(student_data.keys())}")
                            for key in student_data.keys():
                                # Case-insensitive check for student number fields
                                key_lower = key.lower().replace(' ', '').replace('_', '')
                                if key_lower in ['studentnumber', 'studentid', 'student_number', 'student_id']:
                                    potential_value = student_data.get(key)
                                    print(f"🔍 Checking case-insensitive match: '{key}' -> value: {repr(potential_value)}")
                                    if potential_value is not None:
                                        student_number = str(potential_value).strip()
                                        if student_number and student_number != '' and student_number != rfid:
                                            print(f"✅ Found Student Number in field '{key}' (case-insensitive): {student_number}")
                                            break
                                else:
                                    student_number = None
                        
                        # Third pass: Direct explicit access as final fallback
                        if not student_number:
                            print(f"⚠️ Attempting direct explicit access to 'Student Number' field...")
                            direct_value = student_data.get('Student Number')
                            if direct_value is not None:
                                direct_stripped = str(direct_value).strip()
                                print(f"🔍 Direct access result: {repr(direct_stripped)}")
                                if direct_stripped and direct_stripped != '' and direct_stripped != rfid:
                                    student_number = direct_stripped
                                    print(f"✅ Found Student Number via direct access: {student_number}")
                        
                        # If still not found, log detailed diagnostic information and try one more direct access
                        if not student_number:
                            print(f"❌ CRITICAL: Student Number field not found in Firebase for RFID: {rfid}")
                            print(f"❌ Field names checked: {field_names}")
                            print(f"❌ ALL Available fields in Firebase: {list(student_data.keys())}")
                            print(f"❌ Field name-value pairs:")
                            for k, v in student_data.items():
                                print(f"     - '{k}': {repr(v)}")
                            
                            # Final attempt: Check if "Student Number" field exists with any case variation
                            for key in student_data.keys():
                                if key.lower().replace(' ', '').replace('_', '') == 'studentnumber':
                                    potential_value = student_data.get(key)
                                    if potential_value is not None:
                                        potential_value = str(potential_value).strip()
                                        if potential_value and potential_value != '' and potential_value != rfid:
                                            student_number = potential_value
                                            print(f"✅ Found Student Number via case-insensitive key match '{key}': {student_number}")
                                            break
                            
                            # If still not found, return None - display logic will handle it
                            if not student_number:
                                student_number = None
                        else:
                            print(f"✅ FINAL Student Number retrieved: {student_number}")
                        
                        # Strip whitespace from name and gender
                        name_val = student_data.get('Name', f'Student {rfid}')
                        if name_val:
                            name_val = str(name_val).strip()
                        
                        if gender_val:
                            gender_val = str(gender_val).strip()
                        
                        # IMPORTANT: Never set 'id' to RFID - use Student Number or None
                        result = {
                            'student_id': student_number,  # Student Number from Firebase (primary ID for display)
                            'student_number': student_number,  # Alternative field name
                            'name': name_val,  # Name with whitespace stripped
                            'course': course_val,  # Course with whitespace stripped
                            'gender': gender_val,  # Gender with whitespace stripped
                            'rfid': rfid,  # Keep RFID for reference (not displayed as ID)
                            'id': student_number,  # Use Student Number as id (None if not found, NOT RFID)
                            # NOTE: Gmail is NOT included for privacy
                        }
                        print(f"🔍 ========== FIREBASE LOOKUP RESULT ==========")
                        print(f"🔍 Returning student_info with keys: {list(result.keys())}")
                        print(f"🔍 Student Number (student_id): {result.get('student_id')}")
                        print(f"🔍 RFID in result: {result.get('rfid')}")
                        print(f"🔍 Name: {result.get('name')}")
                        print(f"🔍 Course: {result.get('course')}")
                        print(f"🔍 ==============================================")
                        return result
                    else:
                        print(f"❌ WARNING: No student document found with RFID: {rfid}")
                        print(f"❌ Document ID {rfid} does not exist in 'students' collection")
                        return None
                except Exception as e:
                    print(f"❌ ERROR: Firebase query failed for RFID {rfid}: {e}")
                    import traceback
                    print(f"❌ Traceback:")
                    traceback.print_exc()
                    return None
            else:
                print(f"❌ WARNING: Firebase not available - cannot query student by RFID")
                print(f"   Firebase initialized: {self.firebase_initialized}")
                print(f"   Database object: {self.db}")
                return None
        except Exception as e:
            print(f"❌ ERROR: Unexpected error getting student info by RFID: {e}")
            import traceback
            print(f"❌ Traceback:")
            traceback.print_exc()
            return None

    def get_teacher_info_by_rfid(self, rfid):
        """Get teacher information by RFID (using document ID) - Similar to get_student_info_by_rfid()"""
        try:
            print(f"🔍 ========== TEACHER FIREBASE LOOKUP START ==========")
            print(f"🔍 Looking up teacher with RFID: {rfid}")
            print(f"🔍 Firebase initialized: {self.firebase_initialized}")
            print(f"🔍 Database available: {self.db is not None}")
            
            # Try Firebase first
            if self.firebase_initialized and self.db:
                try:
                    # In Firebase, RFID is stored as document ID, not as a field
                    teachers_ref = self.db.collection('teachers')
                    doc_ref = teachers_ref.document(rfid)  # Use RFID as document ID
                    print(f"🔍 Querying Firebase teachers document ID: {rfid}")
                    doc = doc_ref.get()
                
                    if doc.exists:
                        teacher_data = doc.to_dict()
                        print(f"✅ SUCCESS: Found teacher document by RFID {rfid}: {teacher_data.get('Name', teacher_data.get('name', 'Unknown'))}")
                        print(f"🔍 ALL AVAILABLE FIELDS IN FIREBASE: {list(teacher_data.keys())}")
                        
                        # Extract teacher information - handle various field name variations
                        name = teacher_data.get('Name', teacher_data.get('name', 'Unknown Teacher'))
                        gender = teacher_data.get('Gender', teacher_data.get('gender', 'Unknown'))
                        
                        # Return properly formatted teacher info
                        teacher_info = {
                            'rfid': rfid,
                            'name': str(name).strip(),
                            'gender': str(gender).strip(),
                            'type': 'teacher'
                        }
                        
                        print(f"✅ Teacher info retrieved: {teacher_info}")
                        return teacher_info
                    else:
                        print(f"❌ WARNING: No teacher document found with RFID: {rfid}")
                        print(f"❌ Document ID {rfid} does not exist in 'teachers' collection")
                        return None
                except Exception as e:
                    print(f"❌ ERROR: Firebase query failed for teacher RFID {rfid}: {e}")
                    import traceback
                    print(f"❌ Traceback:")
                    traceback.print_exc()
                    return None
            else:
                print(f"❌ WARNING: Firebase not available - cannot query teacher by RFID")
                print(f"   Firebase initialized: {self.firebase_initialized}")
                print(f"   Database object: {self.db}")
                return None
        except Exception as e:
            print(f"❌ ERROR: Unexpected error getting teacher info by RFID: {e}")
            import traceback
            print(f"❌ Traceback:")
            traceback.print_exc()
            return None

    def reset_camera_to_standby(self):
        """Reset camera UI and release any active capture objects to standby mode"""
        try:
            # Stop any active camera loops
            try:
                self.camera_active = False
            except Exception:
                pass
            try:
                self.guard_camera_active = False
            except Exception:
                pass

            # Release OpenCV capture objects if present
            try:
                if hasattr(self, 'camera_cap') and self.camera_cap:
                    try:
                        self.camera_cap.release()
                    except Exception:
                        pass
                    self.camera_cap = None
            except Exception:
                pass

            try:
                if hasattr(self, 'guard_camera_cap') and self.guard_camera_cap:
                    # Respect handoff flags: if a handoff is in progress, don't release here
                    if getattr(self, '_guard_keep_cap_on_stop', False) or getattr(self, '_cap_handed_to_detection', False):
                        print("ℹ️ reset_camera_to_standby: handoff in progress - not releasing guard_camera_cap")
                    else:
                        try:
                            self.guard_camera_cap.release()
                        except Exception:
                            pass
                        self.guard_camera_cap = None
            except Exception:
                pass

            # Update camera label to standby text if UI present
            try:
                if hasattr(self, 'camera_label') and self.camera_label:
                    standby_text = (
                        "📷 CAMERA FEED (STANDBY MODE)\n\n🔒 Camera is CLOSED\n\n"
                        "Camera will ONLY open when:\n• Student taps their ID\n• Detection process starts\n\n"
                        "No camera access during guard login\n\n💡 Camera preview will show here\nwhen detection is active"
                    )
                    self.camera_label.config(text=standby_text, bg='#dbeafe', fg='#374151')
                    try:
                        self.camera_label.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
                    except Exception:
                        pass
            except Exception:
                pass

            print("INFO: Camera reset to standby")
        except Exception as e:
            print(f"WARNING: Could not reset camera to standby: {e}")

    def get_guard_info_by_rfid(self, rfid):
        """Get guard information by RFID"""
        try:
            if self.firebase_initialized and self.db:
                guards_ref = self.db.collection('guards')
                query = guards_ref.where('rfid', '==', rfid)
                docs = query.get()
                
                if docs:
                    doc = docs[0]
                    guard_data = doc.to_dict()
                    return {
                        'rfid': guard_data.get('rfid'),
                        'name': guard_data.get('name'),
                        'role': guard_data.get('role'),
                        'department': guard_data.get('department'),
                        'status': guard_data.get('status')
                    }
            else:
                print("WARNING: Firebase not available - cannot query guard by RFID")
            return None
        except Exception as e:
            print(f"ERROR: Error getting guard info by RFID: {e}")
            return None
    
    def show_incomplete_uniform_popup(self, student_info):
        """Show incomplete uniform popup on main screen"""
        try:
            print(f"🎯 Showing incomplete uniform popup for: {student_info.get('name', 'Unknown')}")
            current_time = self.get_current_timestamp()
            
            # Show incomplete uniform popup
            self.show_red_error_message("Incomplete Uniform", 
                              f"Student: {student_info['name']}\n"
                              f"Course: {student_info['course']}\n"
                              f"Student ID: {student_info['student_id']}\n"
                              f"Time: {current_time}\n\n"
                              f"❌ Uniform verification incomplete!\n"
                              f"Detection stopped after 10 seconds.\n\n"
                              f"💡 Please tap your ID again to retry\n"
                              f"uniform detection.")
            
            print(f"✅ Incomplete uniform popup displayed successfully")
            
            # Update main screen with incomplete uniform status
            if self.main_screen_window and self.main_screen_window.winfo_exists():
                # You can add specific UI updates here if needed
                print(f"✅ Main screen window exists - can add UI updates")
            else:
                print(f"⚠️ Main screen window not available")
                
        except Exception as e:
            print(f"❌ Error showing incomplete uniform popup: {e}")
            self.add_activity_log(f"❌ Error showing incomplete uniform popup: {e}")
    
    def show_complete_uniform_popup(self, student_info):
        """Show complete uniform popup on main screen"""
        try:
            current_time = self.get_current_timestamp()
            
            # Show complete uniform popup
            self.show_green_success_message("Complete Uniform", 
                              f"Student: {student_info['name']}\n"
                              f"Course: {student_info['course']}\n"
                              f"Student ID: {student_info['student_id']}\n"
                              f"Entry Time: {current_time}\n\n"
                              f"✅ Uniform verification complete!\n"
                              f"🎉 Entry recorded successfully!\n\n"
                              f"Student is now inside the school.")
            
            # Update main screen with complete uniform status
            if self.main_screen_window and self.main_screen_window.winfo_exists():
                # You can add specific UI updates here if needed
                pass
                
        except Exception as e:
            print(f"❌ Error showing complete uniform popup: {e}")
            self.add_activity_log(f"❌ Error showing complete uniform popup: {e}")
    
    def show_missing_uniform_message(self):
        """Show missing uniform message in main UI"""
        try:
            # Show missing uniform message
            self.show_red_error_message("Missing Uniform", 
                              "Uniform verification incomplete!\n\n"
                              "Detection stopped after 10 seconds.\n"
                              "Please tap your ID again to retry\n"
                              "uniform detection.")
            
            # Update main screen with missing uniform status
            if self.main_screen_window and self.main_screen_window.winfo_exists():
                # You can add specific UI updates here if needed
                pass
                
        except Exception as e:
            print(f"❌ Error showing missing uniform message: {e}")
            self.add_activity_log(f"❌ Error showing missing uniform message: {e}")
    
    def record_complete_uniform_entry(self, rfid, student_info):
        """Record entry only when uniform is complete"""
        try:
            current_time = self.get_current_timestamp()
            
            # Check if Event Mode is active (should not happen for this function, but check anyway)
            event_mode = self.event_mode_active
            
            # Log activity
            if event_mode:
                self.add_activity_log(f"Student Entry (Complete Uniform - Event Mode): {student_info['name']} (RFID: {rfid}) - {current_time}")
            else:
                self.add_activity_log(f"Student Entry (Complete Uniform): {student_info['name']} (RFID: {rfid}) - {current_time}")
            
            # Clear pending violations when student passes (they succeeded, violations don't count)
            if rfid and rfid in self.active_session_violations:
                pending_count = len(self.active_session_violations[rfid])
                if pending_count > 0:
                    print(f"INFO: Student passed uniform detection - clearing {pending_count} pending violations (violations don't count when student passes)")
                    # Clear pending violations (they passed, so violations shouldn't be counted)
                    del self.active_session_violations[rfid]
                    # Update violation history entries to "cleared" status (optional - student passed)
                    try:
                        student_id = student_info.get('student_id')
                        if student_id and self.firebase_initialized and self.db:
                            violation_ref = self.db.collection('student_violations').document(student_id)
                            # Mark pending violations as "cleared" (student passed)
                            print(f"INFO: Pending violations cleared - student passed uniform detection")
                    except Exception as e:
                        print(f"WARNING: Failed to update violation history status: {e}")
            
            # Clear session tracking
            if rfid in self.active_detection_sessions:
                del self.active_detection_sessions[rfid]
            
            # Save to Firebase (with event_mode flag)
            self.save_permanent_student_activity_to_firebase(rfid, 'time_in', current_time, student_info, event_mode=event_mode)
            
            # CRITICAL: Add to local cache immediately to prevent race condition
            # This ensures that if RFID is tapped again before Firebase query completes, we know entry exists
            import time
            if not hasattr(self, 'recent_entry_cache'):
                self.recent_entry_cache = {}
            self.recent_entry_cache[rfid] = time.time()
            print(f"✅ Added RFID {rfid} to recent entry cache (timestamp: {self.recent_entry_cache[rfid]})")
            
            # Update main screen
            if self.main_screen_window and self.main_screen_window.winfo_exists():
                self.update_main_screen_with_permanent_student(student_info, 'time_in', current_time, rfid=rfid)
            
            # Show success message
            self.show_green_success_message("Entry Recorded", 
                              f"Student: {student_info['name']}\n"
                              f"Course: {student_info['course']}\n"
                              f"Student ID: {student_info['student_id']}\n"
                              f"Permanent RFID: {rfid}\n"
                              f"Entry Time: {current_time}\n\n"
                              f"✅ Complete uniform verified!\n"
                              f"🎉 Entry recorded successfully!")
                
        except Exception as e:
            print(f"❌ Error recording complete uniform entry: {e}")
            self.add_activity_log(f"❌ Error recording complete uniform entry: {e}")
    
    def handle_permanent_student_timein(self, rfid, student_info):
        """Handle permanent student time-in - START DETECTION ONLY (no entry log until uniform complete)"""
        try:
            # Debug: Log what we receive
            print(f"🔍 DEBUG handle_permanent_student_timein: rfid parameter = {rfid}")
            print(f"🔍 DEBUG handle_permanent_student_timein: student_info keys = {list(student_info.keys()) if student_info else 'None'}")
            print(f"🔍 DEBUG handle_permanent_student_timein: student_info rfid field = {student_info.get('rfid', 'NOT FOUND') if student_info else 'N/A'}")
            print(f"🔍 DEBUG handle_permanent_student_timein: event_mode_active = {getattr(self, 'event_mode_active', False)}")
            
            # CRITICAL: Check entry record FIRST (before any other logic)
            # First check local cache (immediate, no Firebase delay), then check Firebase
            import time
            has_entry_record = False
            
            # Check local cache first (prevents race condition)
            if hasattr(self, 'recent_entry_cache') and rfid in self.recent_entry_cache:
                cache_timestamp = self.recent_entry_cache[rfid]
                # Cache is valid for 5 minutes (300 seconds)
                if time.time() - cache_timestamp < 300:
                    has_entry_record = True
                    print(f"✅ Entry record found in local cache for RFID {rfid} (handle_permanent_student_timein)")
                else:
                    # Cache expired, remove it
                    del self.recent_entry_cache[rfid]
                    print(f"⏰ Cache entry expired for RFID {rfid}, removed from cache")
            
            # If not in cache, check Firebase
            if not has_entry_record:
                has_entry_record = self.check_student_has_entry_record(rfid)
                # If found in Firebase, add to cache
                if has_entry_record:
                    if not hasattr(self, 'recent_entry_cache'):
                        self.recent_entry_cache = {}
                    self.recent_entry_cache[rfid] = time.time()
                    print(f"✅ Entry record found in Firebase for RFID {rfid}, added to cache (handle_permanent_student_timein)")
            
            if has_entry_record:
                print(f"🛑 CRITICAL: Entry record found in handle_permanent_student_timein - this should be EXIT!")
                print(f"   Processing EXIT instead of starting detection")
                
                # Stop any running detection immediately
                try:
                    if hasattr(self, 'detection_system') and self.detection_system:
                        self.detection_system.stop_detection()
                    self.detection_active = False
                    self.uniform_detection_complete = False
                except Exception as e:
                    print(f"⚠️ Warning: Could not stop detection: {e}")
                
                # Process as EXIT
                self.handle_permanent_student_timeout(rfid, student_info)
                return  # Exit early - don't start detection
            
            current_time = self.get_current_timestamp()
            
            # CRITICAL: Check if student has active approved violation (24-hour detection skip)
            student_id = student_info.get('student_id') or student_info.get('student_number')
            has_approved_violation = False
            if student_id:
                try:
                    has_approved_violation = self.check_approved_violation_active(student_id, rfid)
                    if has_approved_violation:
                        print(f"✅ Student {student_info.get('name', 'Unknown')} has active approved violation - skipping detection for 24 hours")
                except Exception as e:
                    print(f"⚠️ WARNING: Could not check approved violation: {e}")
                    has_approved_violation = False
            
            # If approved violation is active, skip detection and allow entry directly
            if has_approved_violation:
                print(f"🎉 Approved violation active - bypassing uniform detection for {student_info.get('name', 'Unknown')}")
                # Add detailed activity log entry
                activity_msg = f"Student Entry (Approved Violation - Detection Skipped): {student_info['name']} (ID: {student_id}) - {current_time}"
                self.add_activity_log(activity_msg)
                print(f"📝 Activity log: {activity_msg}")
                
                # CRITICAL: Ensure no detection starts - stop any existing detection first
                if hasattr(self, 'detection_system') and self.detection_system:
                    try:
                        self.detection_system.stop_detection()
                        print("🛑 Stopped any existing detection before approved violation entry")
                    except Exception as e:
                        print(f"⚠️ Warning: Could not stop detection system: {e}")
                
                # Ensure detection_active is False
                self.detection_active = False
                # Set uniform_detection_complete to True since entry will be saved (no detection needed)
                self.uniform_detection_complete = True
                
                # CRITICAL: Enable only DENIED button when approved violation is detected (re-entry scenario)
                # Disable ACCESS GRANTED and CANCEL buttons - student already has violation, only DENIED should be enabled
                # This allows guard to stop the student if needed during re-entry
                if hasattr(self, 'approve_button') and self.approve_button:
                    try:
                        self.approve_button.config(state=tk.DISABLED)  # Disable ACCESS GRANTED button
                        print(f"❌ ACCESS GRANTED button disabled - approved violation detected (re-entry scenario)")
                    except Exception as e:
                        print(f"⚠️ WARNING: Could not disable ACCESS GRANTED button: {e}")
                
                if hasattr(self, 'cancel_button') and self.cancel_button:
                    try:
                        self.cancel_button.config(state=tk.DISABLED)  # Disable CANCEL button
                        print(f"❌ CANCEL button disabled - approved violation detected (re-entry scenario)")
                    except Exception as e:
                        print(f"⚠️ WARNING: Could not disable CANCEL button: {e}")
                
                if hasattr(self, 'deny_button') and self.deny_button:
                    try:
                        self.deny_button.config(state=tk.NORMAL)  # Enable DENIED button
                        print(f"✅ DENIED button enabled - approved violation detected (re-entry scenario)")
                    except Exception as e:
                        print(f"⚠️ WARNING: Could not enable DENIED button: {e}")
                
                # CRITICAL: Store student info for approved violation re-entry scenario
                # This will be used when DENIED is clicked to show HOLD status and enable ACCESS GRANTED
                self.approved_violation_student_info = student_info.copy() if student_info else None
                self.approved_violation_rfid = rfid
                print(f"✅ Stored student info for approved violation re-entry scenario: {student_info.get('name', 'Unknown') if student_info else 'Unknown'}")
                
                # Hide requirements section (no detection = no requirements needed)
                try:
                    self.hide_requirements_section()
                    print("✅ Requirements section hidden for approved violation entry")
                except Exception as e:
                    print(f"⚠️ Warning: Could not hide requirements section: {e}")
                
                # Ensure RFID is in student_info
                if not student_info.get('rfid'):
                    student_info['rfid'] = rfid
                
                # Ensure student_info has proper data
                student_info = self._ensure_student_number_from_firebase(student_info, rfid)
                
                # Record entry immediately (no detection needed, no violations counted)
                self.save_permanent_student_activity_to_firebase(rfid, 'time_in', current_time, student_info, event_mode=False)
                
                # Update main screen with student info
                try:
                    if self.main_screen_window and self.main_screen_window.winfo_exists():
                        student_info['approved_violation'] = True  # Flag to indicate approved violation entry
                        self.update_main_screen_with_permanent_student(student_info, 'time_in', current_time, rfid=rfid)
                        print("✅ Main screen updated with student info (Approved Violation - No Detection)")
                except Exception as e:
                    print(f"⚠️ Warning: Could not update main screen with student info: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Keep guard camera feed running in background (standby mode for guard monitoring)
                try:
                    if not hasattr(self, 'guard_camera_cap') or self.guard_camera_cap is None:
                        self.initialize_guard_camera_feed()
                    print("✅ Guard camera feed maintained in standby mode")
                except Exception as e:
                    print(f"⚠️ Warning: Could not maintain guard camera feed: {e}")
                
                # Auto-open gate
                self.open_gate()
                if self.main_screen_window and self.main_screen_window.winfo_exists():
                    self.update_main_screen_with_gate_status("OPEN", "APPROVED (VIOLATION APPROVED)")
                
                # Show success message
                self.show_green_success_message("Entry Recorded (Approved Violation)", 
                                  f"Student: {student_info['name']}\n"
                                  f"Course: {student_info['course']}\n"
                                  f"Student ID: {student_id}\n"
                                  f"Permanent RFID: {rfid}\n"
                                  f"Entry Time: {current_time}\n\n"
                                  f"✅ Entry recorded (Detection skipped)\n"
                                  f"🚪 Gate opened automatically\n"
                                  f"⏰ Valid for 24 hours")
                
                # Add to recent entries
                try:
                    student_number = student_info.get('student_id') or student_info.get('student_number') or rfid
                    person_data = {
                        'id': student_number,
                        'student_id': student_number,
                        'name': student_info.get('name', f'Student {rfid}'),
                        'type': 'student',
                        'course': student_info.get('course', 'Unknown'),
                        'gender': student_info.get('gender', 'Unknown'),
                        'timestamp': current_time,
                        'status': 'TIME-IN (APPROVED VIOLATION)',
                        'action': 'ENTRY',
                        'approved_violation': True,
                        'guard_id': self.current_guard_id or 'Unknown'
                    }
                    self.add_to_recent_entries(person_data)
                    print(f"📝 Added to recent entries: {student_info.get('name')} - ENTRY (Approved Violation - No Detection)")
                except Exception as e:
                    print(f"⚠️ Warning: Could not add to recent entries: {e}")
                    import traceback
                    traceback.print_exc()
                
                print(f"✅ Approved violation entry complete - detection bypassed successfully")
                
                # CRITICAL: Schedule timer to disable DENIED button after at least 10 seconds
                # Keep DENIED enabled for at least 10 seconds to allow guard to stop student if needed
                # Schedule a separate timer to disable DENIED button (minimum 10 seconds)
                def disable_denied_after_delay():
                    if hasattr(self, 'deny_button') and self.deny_button:
                        try:
                            self.deny_button.config(state=tk.DISABLED)
                            print(f"✅ DENIED button disabled after 10 seconds (approved violation re-entry)")
                        except Exception as e:
                            print(f"⚠️ WARNING: Could not disable DENIED button: {e}")
                
                # Schedule DENIED button disable after 10 seconds (10000 milliseconds)
                self.root.after(10000, disable_denied_after_delay)
                print(f"✅ Scheduled DENIED button disable after 10 seconds (approved violation re-entry)")
                
                # CRITICAL: Schedule 15-second timer to clear main screen and return to standby
                # This ensures the entry status is displayed for 15 seconds, then main screen clears
                # Note: DENIED button will be disabled after 10 seconds (above), before main screen clears
                self.clear_main_screen_after_gate_action()
                
                return  # Exit early - no detection needed
            
            # Check if Event Mode is active - bypass detection and auto-record entry
            event_mode = getattr(self, 'event_mode_active', False)
            print(f"🔍 DEBUG: Event Mode check: event_mode_active = {event_mode}")
            if event_mode:
                print(f"🎉 Event Mode active - bypassing uniform detection for {student_info.get('name', 'Unknown')}")
                # Add detailed activity log entry
                activity_msg = f"Student Entry (Event Mode): {student_info['name']} (ID: {student_info.get('student_id', rfid)}) - {current_time}"
                self.add_activity_log(activity_msg)
                print(f"📝 Activity log: {activity_msg}")
                
                # CRITICAL: Ensure no detection starts - stop any existing detection first
                if hasattr(self, 'detection_system') and self.detection_system:
                    try:
                        self.detection_system.stop_detection()
                        print("🛑 Stopped any existing detection before Event Mode entry")
                    except Exception as e:
                        print(f"⚠️ Warning: Could not stop detection system: {e}")
                
                # Ensure detection_active is False
                self.detection_active = False
                self.uniform_detection_complete = False
                
                # Hide requirements section (no detection = no requirements needed)
                try:
                    self.hide_requirements_section()
                    print("✅ Requirements section hidden for Event Mode")
                except Exception as e:
                    print(f"⚠️ Warning: Could not hide requirements section: {e}")
                
                # Ensure RFID is in student_info
                if not student_info.get('rfid'):
                    student_info['rfid'] = rfid
                
                # Ensure student_info has proper data
                student_info = self._ensure_student_number_from_firebase(student_info, rfid)
                
                # Record entry immediately (no detection needed)
                self.save_permanent_student_activity_to_firebase(rfid, 'time_in', current_time, student_info, event_mode=True)
                
                # In Event Mode, SHOW student info on main screen (but no detection)
                # Update main screen with student info for Event Mode entry
                try:
                    if self.main_screen_window and self.main_screen_window.winfo_exists():
                        # Ensure event_mode flag is set in student_info so update_main_screen_with_permanent_student can use it
                        student_info['event_mode'] = True
                        # Use update_main_screen_with_permanent_student to show student info
                        self.update_main_screen_with_permanent_student(student_info, 'time_in', current_time, rfid=rfid)
                        print("✅ Main screen updated with student info (Event Mode)")
                except Exception as e:
                    print(f"⚠️ Warning: Could not update main screen with student info: {e}")
                    import traceback
                    traceback.print_exc()
                    
                # Keep guard camera feed running in background (standby mode for guard monitoring)
                try:
                    # Ensure guard camera feed is running (for guard monitoring, not detection)
                    if not hasattr(self, 'guard_camera_cap') or self.guard_camera_cap is None:
                        # If guard camera is not running, initialize it
                        self.initialize_guard_camera_feed()
                    print("✅ Guard camera feed maintained in standby mode")
                except Exception as e:
                    print(f"⚠️ Warning: Could not maintain guard camera feed: {e}")
                
                # Auto-open gate
                self.open_gate()
                if self.main_screen_window and self.main_screen_window.winfo_exists():
                    self.update_main_screen_with_gate_status("OPEN", "APPROVED (EVENT MODE)")
                
                # Show success message
                self.show_green_success_message("Entry Recorded (Event Mode)", 
                                  f"Student: {student_info['name']}\n"
                                  f"Course: {student_info['course']}\n"
                                  f"Student ID: {student_info['student_id']}\n"
                                  f"Permanent RFID: {rfid}\n"
                                  f"Entry Time: {current_time}\n\n"
                                  f"✅ Entry recorded (Event Mode)\n"
                                  f"🚪 Gate opened automatically")
                
                # Add to recent entries with proper format
                try:
                    student_number = student_info.get('student_id') or student_info.get('student_number') or rfid
                    person_data = {
                        'id': student_number,
                        'student_id': student_number,
                        'name': student_info.get('name', f'Student {rfid}'),
                        'type': 'student',
                        'course': student_info.get('course', 'Unknown'),
                        'gender': student_info.get('gender', 'Unknown'),
                        'timestamp': current_time,
                        'status': 'TIME-IN (EVENT MODE)',
                        'action': 'ENTRY',  # CRITICAL: action must be 'ENTRY' to show in recent entries
                        'event_mode': True,  # CRITICAL: Flag to bypass duplicate suppression
                        'guard_id': self.current_guard_id or 'Unknown'
                    }
                    # Debug: Log what we're trying to add
                    print(f"🔍 DEBUG: Attempting to add to recent entries:")
                    print(f"   Name: {person_data['name']}")
                    print(f"   ID: {person_data['id']}")
                    print(f"   Action: {person_data['action']}")
                    print(f"   Status: {person_data['status']}")
                    print(f"   Event Mode: {person_data['event_mode']}")
                    print(f"   Timestamp: {person_data['timestamp']}")
                    
                    # Explicitly call add_to_recent_entries to ensure it's logged
                    self.add_to_recent_entries(person_data)
                    
                    # Force UI update to ensure listbox shows the entry
                    try:
                        if hasattr(self, 'recent_entries_listbox') and self.recent_entries_listbox:
                            self.recent_entries_listbox.update_idletasks()
                            self.root.update_idletasks()  # Force root window update
                            print(f"✅ UI updated after adding to recent entries")
                    except Exception as ui_e:
                        print(f"⚠️ Warning: Could not update UI: {ui_e}")
                    
                    print(f"📝 Added to recent entries: {student_info.get('name')} - ENTRY (Event Mode)")
                except Exception as e:
                    print(f"⚠️ Warning: Could not add to recent entries: {e}")
                    import traceback
                    traceback.print_exc()
                
                print(f"✅ Event Mode entry complete - detection bypassed successfully")
                return  # Exit early - no detection needed
            
            # SAFETY CHECK: Prevent conflicts if detection is already running
            if getattr(self, 'detection_active', False):
                current_rfid = getattr(getattr(self, 'detection_system', None), 'detection_service', None)
                current_rfid = getattr(current_rfid, 'current_rfid', None) if current_rfid else None
                
                if current_rfid == rfid:
                    print(f"🔄 Detection already running for same student - treating as retry")
                    # This is a retry - restart detection
                    if hasattr(self, 'detection_system') and self.detection_system:
                        print("🔄 Stopping current detection for retry...")
                        self.detection_system.stop_detection()
                        # Shorten wait to reduce camera downtime
                        time.sleep(0.1)
                else:
                    print(f"⚠️ Detection running for different student ({current_rfid}) - stopping first")
                    if hasattr(self, 'detection_system') and self.detection_system:
                        self.detection_system.stop_detection()
                        time.sleep(0.1)
            
            # CRITICAL: Disable APPROVE and CANCEL buttons when detection starts
            if hasattr(self, 'approve_button'):
                try:
                    self.approve_button.config(state=tk.DISABLED)
                    self.incomplete_student_info_for_approve = None  # Clear stored info
                    print(f"✅ APPROVE button disabled - detection starting")
                except Exception as e:
                    print(f"⚠️ WARNING: Could not disable APPROVE button: {e}")
            
            if hasattr(self, 'cancel_button'):
                try:
                    self.cancel_button.config(state=tk.DISABLED)
                    print(f"✅ CANCEL button disabled - detection starting")
                except Exception as e:
                    print(f"⚠️ WARNING: Could not disable CANCEL button: {e}")
            
            # Reset denied button state flags (but keep DENIED button enabled - enable it early during detection)
            # Only reset the flags, don't disable the button yet
            self.denied_button_clicked = False
            self.original_detection_state = None
            self.original_uniform_status = None
            
            # Enable DENIED button early during detection (guard can click it anytime during detection)
            if hasattr(self, 'deny_button'):
                try:
                    self.deny_button.config(state=tk.NORMAL)
                    print(f"✅ DENIED button enabled - detection started, guard can click anytime")
                except Exception as e:
                    print(f"⚠️ WARNING: Could not enable DENIED button: {e}")
            
            # Make checkboxes read-only when detection starts
            self._make_requirement_checkboxes_editable(False)
            
            # Log activity (detection started, not entry) - ONLY if Event Mode is OFF
            if not getattr(self, 'event_mode_active', False):
                self.add_activity_log(f"Student Detection Started: {student_info['name']} (RFID: {rfid}) - {current_time}")
            else:
                # Event Mode - entry should already be logged above, don't log detection started
                print(f"🛑 Event Mode active - NOT logging 'Detection Started'")
            
            # Initialize detection session tracking (if new session, clear old pending violations)
            # If this is a retry (session already exists), keep existing pending violations
            if rfid not in self.active_detection_sessions:
                # New session - initialize tracking
                self.active_detection_sessions[rfid] = current_time
                # Clear any old pending violations from previous sessions (cleanup)
                if rfid in self.active_session_violations:
                    print(f"INFO: Clearing old pending violations for new session (RFID: {rfid})")
                    del self.active_session_violations[rfid]
                print(f"INFO: New detection session started for RFID: {rfid}")
            else:
                print(f"INFO: Retry detection - keeping existing session violations (RFID: {rfid})")
            
            # DO NOT save to Firebase yet - wait for complete uniform
            # Update main screen immediately to show student info
            if self.main_screen_window and self.main_screen_window.winfo_exists():
                try:
                    # Create person info for main screen with student data from Firebase
                    # Use Student Number (student_id) as the ID (not RFID)
                    # IMPORTANT: Ensure we have Student Number from Firebase, not fallback to RFID
                    
                    # Strip whitespace from course if it exists
                    course_val = student_info.get('course', 'Unknown Course')
                    if course_val:
                        course_val = str(course_val).strip()
                    
                    # Ensure Student Number is ALWAYS fetched from Firebase before creating person_info
                    student_info = self._ensure_student_number_from_firebase(student_info, rfid)
                    
                    # Get Student Number - validate it's not RFID
                    student_number = student_info.get('student_id') or student_info.get('student_number')
                    if not student_number or student_number == 'Unknown' or str(student_number).strip() == str(rfid).strip():
                        # If still missing or equals RFID, fetch fresh from Firebase
                        print(f"⚠️ Student Number still missing or equals RFID, fetching fresh from Firebase...")
                        full_student_info = self.get_student_info_by_rfid(rfid)
                        if full_student_info:
                            student_number = full_student_info.get('student_id') or full_student_info.get('student_number')
                            # Validate student_number is valid (not None, not empty, not RFID)
                            if student_number and str(student_number).strip() != '' and str(student_number).strip() != str(rfid).strip():
                                # Update student_info with complete data
                                student_info.update(full_student_info)
                                course_val = full_student_info.get('course', course_val)
                                if course_val:
                                    course_val = str(course_val).strip()
                                print(f"✅ Updated student_info with Student Number from Firebase: {student_number}")
                            else:
                                print(f"⚠️ WARNING: Student Number from Firebase is invalid (None, empty, or equals RFID): {student_number}")
                                student_number = 'Student Number Not Found'
                        else:
                            student_number = 'Student Number Not Found'
                    
                    # CRITICAL: Check Event Mode before setting detection status
                    if getattr(self, 'event_mode_active', False):
                        print(f"🛑 Event Mode active - should not reach this code (detection start)")
                        # This should have been caught earlier, but as safety check, return here
                        return
                    
                    person_info = {
                        'id': student_number,  # Use Student Number as ID (never RFID or Unknown)
                        'student_id': student_number,  # Student Number from Firebase
                        'student_number': student_number,  # Alternative field name
                        'rfid': rfid,  # Keep RFID for reference (not displayed as ID)
                        'name': student_info.get('name', 'Unknown Student'),
                        'type': 'student',
                        'course': course_val,  # Course with whitespace stripped
                        'gender': student_info.get('gender', 'Unknown Gender'),
                        # NOTE: Gmail is NOT included for privacy
                        'timestamp': current_time,
                        'status': 'DETECTING',
                    }
                    # Set detection_active BEFORE updating main screen (only if Event Mode is OFF)
                    if not getattr(self, 'event_mode_active', False):
                        self.detection_active = True
                        self.uniform_detection_complete = False
                        self.update_main_screen_with_person(person_info)
                    print(f"✅ Main screen updated with student info: {student_info.get('name', 'Unknown')}")
                except Exception as e:
                    print(f"⚠️ Error updating main screen with student info: {e}")
            
            # CRITICAL: Check Event Mode again before starting detection
            if getattr(self, 'event_mode_active', False):
                print(f"🛑 Event Mode active - detection should not start")
                return  # Exit - detection should not run
            
            # Show detection started message
            self.show_green_success_message("Uniform Detection Started", 
                              f"Student: {student_info['name']}\n"
                              f"Course: {student_info['course']}\n"
                              f"Student ID: {student_info['student_id']}\n"
                              f"Permanent RFID: {rfid}\n"
                              f"Detection Started: {current_time}\n\n"
                              f"Note: Uniform detection in progress\n"
                              f"💡 Tip: If uniform detection fails, tap your ID again to retry")
            
            # Start detection for permanent students using external camera detection module
            # NOTE: This should NOT execute if Event Mode is active (checked above)
            try:
                person_name = student_info['name']
                course = student_info.get('course', 'Unknown')
                gender = student_info.get('gender', 'Unknown')
                
                # Ensure RFID is included in student_info for timer and display purposes
                # IMPORTANT: Create a new dict with all fields including Student Number to prevent any data loss
                # CRITICAL: Ensure RFID is in student_info before any processing
                if not student_info.get('rfid'):
                    student_info['rfid'] = rfid
                    print(f"🔍 DEBUG: Added RFID to student_info in handle_permanent_student_timein: {rfid}")
                
                # First ensure student_info has Student Number from Firebase (this will preserve RFID)
                student_info = self._ensure_student_number_from_firebase(student_info, rfid)
                
                # DEFENSIVE: Ensure RFID is still present after _ensure_student_number_from_firebase
                if not student_info.get('rfid'):
                    student_info['rfid'] = rfid
                    print(f"🔍 DEBUG: Defensively added RFID after _ensure_student_number_from_firebase: {rfid}")
                
                # Get Student Number and validate it's not RFID
                student_num = student_info.get('student_id') or student_info.get('student_number')
                
                # Validate: Student Number should not be RFID, 'Unknown', None, or empty
                if not student_num or student_num == 'Unknown' or str(student_num).strip() == str(rfid).strip():
                    # Fetch fresh data from Firebase to ensure Student Number is present
                    print(f"⚠️ Student Number missing or equals RFID, fetching fresh data from Firebase for RFID: {rfid}")
                    fresh_student_info = self.get_student_info_by_rfid(rfid)
                    if fresh_student_info:
                        student_info.update(fresh_student_info)
                        student_num = fresh_student_info.get('student_id') or fresh_student_info.get('student_number')
                        
                        # Validate again after fetch
                        if not student_num or str(student_num).strip() == str(rfid).strip():
                            print(f"⚠️ WARNING: Student Number still equals RFID or not found. Field may be missing in Firebase.")
                            student_num = 'Student Number Not Found'
                        else:
                            print(f"✅ Updated student_info with fresh Firebase data. Student Number: {student_num}")
                    else:
                        student_num = 'Student Number Not Found'
                
                # Strip whitespace from course if it exists
                course_val = student_info.get('course', 'Unknown Course')
                if course_val:
                    course_val = str(course_val).strip()
                
                # CRITICAL: Create student_info_with_rfid with ALL fields, including RFID
                # Start with student_info (which should now have all fields including RFID)
                # and update with specific values to ensure completeness
                student_info_with_rfid = dict(student_info)  # Copy all existing fields
                student_info_with_rfid.update({
                    'student_id': student_num,  # Student Number from Firebase (never RFID)
                    'student_number': student_num,  # Alternative field name
                    'name': student_info.get('name', 'Unknown Student'),
                    'course': course_val,  # Course with whitespace stripped
                    'gender': student_info.get('gender', 'Unknown Gender'),
                    'rfid': rfid,  # CRITICAL: Ensure RFID is always present
                    'id': student_num,  # Use Student Number as id for display
                    # NOTE: Gmail is NOT included for privacy
                })
                # Final defensive check - ensure RFID is present
                if not student_info_with_rfid.get('rfid'):
                    student_info_with_rfid['rfid'] = rfid
                    print(f"🔍 DEBUG: Final defensive add of RFID to student_info_with_rfid: {rfid}")
                print(f"🔍 DEBUG: student_info_with_rfid created with Student Number: {student_num} (RFID: {rfid})")
                
                print(f"🔍 External detection launch for: {person_name} | {course} | {gender}")
                print(f"🔍 DEBUG: student_info_with_rfid keys: {list(student_info_with_rfid.keys())}")
                print(f"🔍 DEBUG: student_info_with_rfid rfid: {student_info_with_rfid.get('rfid')}")
                print(f"🔍 DEBUG: rfid parameter value: {rfid}")
                # Store RFID separately for timer callback - ALWAYS store it here
                self.current_rfid_for_timer = rfid
                print(f"🔍 DEBUG: Stored current_rfid_for_timer = {self.current_rfid_for_timer}")
                
                # CRITICAL: Ensure student_info_with_rfid definitely has RFID before passing
                if not student_info_with_rfid.get('rfid'):
                    student_info_with_rfid['rfid'] = rfid
                    print(f"🔍 DEBUG: Added RFID to student_info_with_rfid before calling start_external_camera_detection: {rfid}")
                
                # CRITICAL: Final Event Mode check before calling detection
                if getattr(self, 'event_mode_active', False):
                    print(f"🛑 Event Mode active - NOT calling start_external_camera_detection")
                    # Do NOT log "Detection started" in Event Mode - entry should already be logged
                    return  # Do not start detection
                
                print(f"🔍 ========== CALLING start_external_camera_detection ==========")
                print(f"🔍 student_info_with_rfid keys being passed: {list(student_info_with_rfid.keys())}")
                print(f"🔍 student_info_with_rfid rfid being passed: {student_info_with_rfid.get('rfid')}")
                print(f"🔍 Event Mode check before detection start: {getattr(self, 'event_mode_active', False)}")
                
                # CRITICAL: Only call detection and log if Event Mode is OFF
                if not getattr(self, 'event_mode_active', False):
                    self.start_external_camera_detection(student_info_with_rfid)
                    self.add_activity_log(f"Detection started for permanent student: {person_name} ({course})")
                else:
                    print(f"🛑 Event Mode active - skipping detection start and log")
            except Exception as e:
                self.add_activity_log(f"Failed to start detection for permanent student: {str(e)}")
                print(f"ERROR: Error starting detection for permanent student: {e}")
            
        except Exception as e:
            print(f"ERROR: Error handling permanent student time-in: {e}")
            self.add_activity_log(f"Error processing time-in for permanent student RFID {rfid}: {e}")

    def start_external_camera_detection(self, student_info):
        """Use standalone camera detection script to run detection based on RFID-selected model."""
        try:
            # CRITICAL: Check Event Mode FIRST - if active, do NOT start detection
            if getattr(self, 'event_mode_active', False):
                print(f"🛑 Event Mode is ACTIVE - detection start blocked")
                print(f"🛑 start_external_camera_detection called but Event Mode prevents detection")
                # Stop any detection that might have started
                try:
                    if hasattr(self, 'detection_system') and self.detection_system:
                        self.detection_system.stop_detection()
                    self.detection_active = False
                    self.hide_requirements_section()
                except Exception:
                    pass
                return False  # Do not start detection
            
            # CRITICAL: Log what we receive at the start of this function
            print(f"🔍 ========== start_external_camera_detection RECEIVED ==========")
            print(f"🔍 student_info parameter keys: {list(student_info.keys()) if student_info else 'None'}")
            print(f"🔍 student_info parameter rfid: {student_info.get('rfid') if student_info else 'N/A'}")
            print(f"🔍 current_rfid_for_timer: {getattr(self, 'current_rfid_for_timer', 'NOT SET')}")
            print(f"🔍 Event Mode active: {getattr(self, 'event_mode_active', False)}")
            
            # CRITICAL: Ensure RFID is in student_info and backup
            if student_info:
                # First, get RFID from student_info if present
                rfid_from_student_info = student_info.get('rfid')
                
                if rfid_from_student_info:
                    # Store it in backup
                    self.current_rfid_for_timer = rfid_from_student_info
                    print(f"🔍 DEBUG: Backed up RFID from student_info to current_rfid_for_timer: {rfid_from_student_info}")
                elif hasattr(self, 'current_rfid_for_timer') and self.current_rfid_for_timer:
                    # RFID missing from student_info but we have backup
                    student_info['rfid'] = self.current_rfid_for_timer
                    print(f"🔍 DEBUG: Added missing RFID to student_info in start_external_camera_detection: {self.current_rfid_for_timer}")
                else:
                    print(f"⚠️ WARNING: No RFID found in student_info and no backup available!")
                    print(f"   student_info keys: {list(student_info.keys())}")
                    
                # Also ensure student_id/Student Number is present if missing
                if not student_info.get('student_id') and not student_info.get('student_number'):
                    print(f"🔍 DEBUG: Student Number missing from student_info, will be fetched in _run_stream")
            import threading
            import importlib.util
            import importlib.machinery
            import os
            import time

            script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'camera detection.py')
            if not os.path.exists(script_path):
                print(f"❌ camera detection.py not found at: {script_path}")
                return False

            # Dynamically load the module from path (filename contains space)
            loader = importlib.machinery.SourceFileLoader('camera_detection_ext', script_path)
            spec = importlib.util.spec_from_loader(loader.name, loader)
            camera_mod = importlib.util.module_from_spec(spec)
            loader.exec_module(camera_mod)

            # Determine model path using external module's mapping
            model_path = camera_mod.model_path_for_student({
                'student_id': student_info.get('student_id'),
                'name': student_info.get('name'),
                'course': student_info.get('course'),
                'gender': student_info.get('gender'),
                'rfid': student_info.get('rfid') or student_info.get('id')
            })

            if not model_path:
                print("⚠️ No model path resolved; aborting external detection")
                return False

            # Stop guard preview to free the camera before starting external detection
            try:
                self.stop_guard_camera_feed()
            except Exception:
                pass

            # Stop any ongoing external stream
            try:
                if hasattr(self, 'external_detection_stop_event') and self.external_detection_stop_event:
                    self.external_detection_stop_event.set()
                    time.sleep(0.05)
            except Exception:
                pass

            # Prepare stop event and start streaming frames into the UI
            self.external_detection_stop_event = threading.Event()
            
            # Reset requirement detection tracking for new student
            self._requirement_detection_counts = {}
            self._permanently_checked_requirements = set()
            self.requirements_hide_scheduled = False  # Reset flag for new detection
            
            # Disable retry button when starting new detection (removed approve_complete_uniform_btn)
            self.incomplete_student_info_for_retry = None
            
            # Disable ID input field during detection to prevent 2nd person from tapping
            if hasattr(self, 'person_id_entry'):
                self.person_id_entry.config(state=tk.DISABLED)
                print(f"🔒 ID input field disabled - detection in progress")
            
            # Mark detection as active
            self.detection_active = True
            
            # Note: current_rfid_for_timer should already be set by handle_permanent_student_timein
            # current_student_info_for_timer will be set in _run_stream
            
            # CRITICAL: Create a guaranteed immutable copy with RFID for the closure
            # This prevents any modifications to student_info from affecting the closure
            student_info_for_closure = dict(student_info) if student_info else {}
            # Ensure RFID is definitely in the closure copy
            if not student_info_for_closure.get('rfid') and hasattr(self, 'current_rfid_for_timer') and self.current_rfid_for_timer:
                student_info_for_closure['rfid'] = self.current_rfid_for_timer
                print(f"🔍 DEBUG: Added RFID to closure copy: {self.current_rfid_for_timer}")
            elif student_info_for_closure.get('rfid'):
                # Backup the RFID
                self.current_rfid_for_timer = student_info_for_closure.get('rfid')
                print(f"🔍 DEBUG: Backed up RFID from closure copy: {self.current_rfid_for_timer}")
            
            print(f"🔍 ========== CREATING CLOSURE ==========")
            print(f"🔍 student_info_for_closure keys: {list(student_info_for_closure.keys())}")
            print(f"🔍 student_info_for_closure rfid: {student_info_for_closure.get('rfid')}")
            print(f"🔍 current_rfid_for_timer: {getattr(self, 'current_rfid_for_timer', 'NOT SET')}")
            
            # Set uniform requirements based on course and gender
            course = student_info.get('course', '')
            gender = student_info.get('gender', '')
            self.set_uniform_requirements_by_course(course, gender)
            
            # Show uniform requirements section when student ID is tapped
            self.show_requirements_section(student_info)
            
            # Reset denied button state flags when new student ID is tapped (but keep button enabled)
            # Only reset flags, don't disable button - it will be enabled when detection starts
            self.denied_button_clicked = False
            self.original_detection_state = None
            self.original_uniform_status = None
            
            # Reset DENIED button text and command to original (but keep it enabled)
            if hasattr(self, 'deny_button') and self.deny_button:
                self.deny_button.config(
                    text="DENIED",
                    command=self.handle_interface_deny
                )
            
            # Make checkboxes read-only initially
            self._make_requirement_checkboxes_editable(False)

            def _run_stream():
                try:
                    # CRITICAL: Log what we receive at the very start
                    print(f"🔍 ========== _run_stream STARTED ==========")
                    print(f"🔍 student_info_for_closure keys: {list(student_info_for_closure.keys()) if student_info_for_closure else 'None'}")
                    print(f"🔍 student_info_for_closure rfid: {student_info_for_closure.get('rfid') if student_info_for_closure else 'N/A'}")
                    print(f"🔍 original student_info closure keys: {list(student_info.keys()) if student_info else 'None'}")
                    print(f"🔍 original student_info closure rfid: {student_info.get('rfid') if student_info else 'N/A'}")
                    print(f"🔍 current_rfid_for_timer at start: {getattr(self, 'current_rfid_for_timer', 'NOT SET')}")
                    
                    # CRITICAL: Use the guaranteed closure copy that has RFID
                    # student_info_for_closure is captured in the closure and guaranteed to have RFID
                    working_student_info = dict(student_info_for_closure) if student_info_for_closure else (dict(student_info) if student_info else {})
                    
                    # If RFID is missing, try to get it from backup or add it
                    if not working_student_info.get('rfid'):
                        if hasattr(self, 'current_rfid_for_timer') and self.current_rfid_for_timer:
                            working_student_info['rfid'] = self.current_rfid_for_timer
                            print(f"🔍 DEBUG: Added RFID from backup to working_student_info: {self.current_rfid_for_timer}")
                        else:
                            print(f"⚠️ WARNING: Cannot find RFID anywhere!")
                    
                    # Initialize timer when camera detection begins
                    import time
                    self.uniform_detection_timer_start = time.time()
                    self.uniform_detection_complete = False
                    # Ensure student_info has Student Number and RFID before storing it for timer
                    # Use a local variable to avoid UnboundLocalError when reassigning
                    validated_student_info = working_student_info
                    
                    # CRITICAL: Extract RFID from multiple sources
                    rfid_to_store = None
                    
                    # First, try to get RFID from working_student_info
                    if validated_student_info and validated_student_info.get('rfid'):
                        rfid_to_store = validated_student_info.get('rfid')
                        print(f"🔍 DEBUG: Found RFID in validated_student_info: {rfid_to_store}")
                    # Second, try to get from instance variable (backup)
                    elif hasattr(self, 'current_rfid_for_timer') and self.current_rfid_for_timer:
                        rfid_to_store = self.current_rfid_for_timer
                        print(f"🔍 DEBUG: Using RFID from current_rfid_for_timer backup: {rfid_to_store}")
                    # Third, try to get from closure variable directly
                    elif student_info and isinstance(student_info, dict) and student_info.get('rfid'):
                        rfid_to_store = student_info.get('rfid')
                        print(f"🔍 DEBUG: Found RFID in closure student_info: {rfid_to_store}")
                    
                    if not rfid_to_store:
                        print(f"❌ CRITICAL: No RFID found anywhere!")
                        print(f"   validated_student_info keys: {list(validated_student_info.keys()) if validated_student_info else 'None'}")
                        print(f"   validated_student_info rfid: {validated_student_info.get('rfid') if validated_student_info else 'N/A'}")
                        print(f"   current_rfid_for_timer: {getattr(self, 'current_rfid_for_timer', 'NOT SET')}")
                        print(f"   student_info (parameter) keys: {list(student_info.keys()) if student_info else 'None'}")
                        print(f"   student_info (parameter) rfid: {student_info.get('rfid') if student_info else 'N/A'}")
                    
                    if validated_student_info:
                        # CRITICAL: Ensure RFID is present BEFORE any processing
                        if rfid_to_store:
                            validated_student_info['rfid'] = rfid_to_store
                            # Also ensure current_rfid_for_timer is set (backup)
                            self.current_rfid_for_timer = rfid_to_store
                            print(f"🔍 DEBUG: Added RFID to validated_student_info and backup: {rfid_to_store}")
                        
                        # Ensure Student Number is present and valid (not RFID)
                        if rfid_to_store:
                            # This function now guarantees RFID is preserved
                            validated_student_info = self._ensure_student_number_from_firebase(validated_student_info, rfid_to_store)
                            
                            # Validate Student Number is not RFID
                            student_num = validated_student_info.get('student_id') or validated_student_info.get('student_number')
                            if not student_num or str(student_num).strip() == str(rfid_to_store).strip():
                                print(f"⚠️ WARNING: Student Number still missing or equals RFID after validation")
                                # Try one more fetch
                                full_student_info = self.get_student_info_by_rfid(rfid_to_store)
                                if full_student_info:
                                    validated_student_info.update(full_student_info)
                                    # CRITICAL: Preserve RFID after update
                                    validated_student_info['rfid'] = rfid_to_store
                        
                        # DEFENSIVE: Ensure RFID is present one more time before storing
                        if rfid_to_store:
                            validated_student_info['rfid'] = rfid_to_store
                            print(f"🔍 DEBUG: Defensively added RFID before storing: {rfid_to_store}")
                        elif not validated_student_info.get('rfid'):
                            print(f"⚠️ CRITICAL WARNING: Cannot add RFID - rfid_to_store is None!")
                    
                    # Make a deep copy to prevent any modifications from affecting our stored reference
                    import copy
                    self.current_student_info_for_timer = copy.deepcopy(validated_student_info) if validated_student_info else None
                    
                    # Final verification - ensure Student Number and RFID are in stored dict
                    if self.current_student_info_for_timer:
                        # CRITICAL: Ensure RFID is in the stored dict (guaranteed presence)
                        # Use rfid_to_store first, then fallback to current_rfid_for_timer
                        rfid_for_final_check = rfid_to_store if rfid_to_store else (self.current_rfid_for_timer if self.current_rfid_for_timer else None)
                        if rfid_for_final_check:
                            self.current_student_info_for_timer['rfid'] = rfid_for_final_check
                            # Also ensure backup is set
                            self.current_rfid_for_timer = rfid_for_final_check
                            print(f"🔍 DEBUG: Final check - RFID set in current_student_info_for_timer: {rfid_for_final_check}")
                        elif not self.current_student_info_for_timer.get('rfid'):
                            print(f"⚠️ CRITICAL WARNING: No RFID available to store in current_student_info_for_timer!")
                            print(f"   rfid_to_store: {rfid_to_store}")
                            print(f"   current_rfid_for_timer: {self.current_rfid_for_timer}")
                            print(f"   stored keys: {list(self.current_student_info_for_timer.keys())}")
                        
                        # Ensure Student Number is set as id field and validate it's not RFID
                        student_num = self.current_student_info_for_timer.get('student_id') or self.current_student_info_for_timer.get('student_number')
                        if student_num and self.current_rfid_for_timer and str(student_num).strip() != str(self.current_rfid_for_timer).strip():
                            self.current_student_info_for_timer['id'] = student_num
                        elif student_num:
                            self.current_student_info_for_timer['id'] = student_num
                        else:
                            print(f"⚠️ WARNING: Stored Student Number is invalid or equals RFID")
                    
                    print(f"🔍 ========== TIMER STORAGE VERIFICATION ==========")
                    print(f"🔍 Stored student_info for timer:")
                    print(f"   - Has student_id: {bool(self.current_student_info_for_timer.get('student_id') if self.current_student_info_for_timer else False)}")
                    print(f"   - Has rfid: {bool(self.current_student_info_for_timer.get('rfid') if self.current_student_info_for_timer else False)}")
                    print(f"   - Stored student_info keys: {list(self.current_student_info_for_timer.keys()) if self.current_student_info_for_timer else 'None'}")
                    print(f"   - Student Number in stored info: {self.current_student_info_for_timer.get('student_id') if self.current_student_info_for_timer else 'None'}")
                    print(f"   - RFID in stored info: {self.current_student_info_for_timer.get('rfid') if self.current_student_info_for_timer else 'None'}")
                    print(f"   - current_rfid_for_timer backup: {self.current_rfid_for_timer}")
                    print(f"   - rfid_to_store: {rfid_to_store if 'rfid_to_store' in locals() else 'N/A'}")
                    print(f"🔍 ===============================================")
                    
                    # Start background timer monitoring thread
                    if hasattr(self, 'uniform_detection_timer_thread') and self.uniform_detection_timer_thread and self.uniform_detection_timer_thread.is_alive():
                        # Stop existing timer thread if any
                        pass
                    
                    self.uniform_detection_timer_thread = threading.Thread(
                        target=self._monitor_uniform_detection_timer,
                        daemon=True
                    )
                    self.uniform_detection_timer_thread.start()
                    print(f"⏱️ 15-second uniform detection timer started")
                    
                    print(f"🎥 External detection streaming to UI using model: {model_path}")
                    # Stream frames into the UI live feed
                    camera_mod.run_detection_stream(
                        model_path,
                        frame_callback=self.update_camera_feed,
                        detected_callback=self.update_detected_classes_list,
                        stop_event=self.external_detection_stop_event,
                        conf=0.35,
                    )
                finally:
                    # Cleanup timer variables
                    # NOTE: DO NOT clear current_student_info_for_timer or current_rfid_for_timer here
                    # as the timer thread may still need them. They will be cleared after timeout handler completes.
                    self.uniform_detection_timer_start = None
                    self.uniform_detection_complete = False
                    # Keep current_student_info_for_timer and current_rfid_for_timer until timeout completes
                    
                    # After it exits, keep UI in standby view
                    try:
                        self.initialize_guard_camera_feed()
                        # Keep requirements section visible - only hide when gate control button is clicked
                        # Don't hide when detection stops - requirements stay visible until guard clicks a button
                        # if not self.requirements_hide_scheduled:
                        #     self.hide_requirements_section()
                        # else:
                        #     # Reset flag after scheduled hide completes
                        #     self.root.after(5100, lambda: setattr(self, 'requirements_hide_scheduled', False))
                    except Exception:
                        pass

            threading.Thread(target=_run_stream, daemon=True).start()
            return True
        except Exception as e:
            print(f"❌ Failed to start external camera detection: {e}")
            return False
    
    def handle_permanent_student_timeout(self, rfid, student_info):
        """Handle permanent student time-out - SAFE detection stop to prevent camera conflicts"""
        try:
            current_time = self.get_current_timestamp()
            
            print(f"🛑 Processing EXIT for student: {student_info.get('name', 'Unknown')}")
            print(f"🔍 DEBUG handle_permanent_student_timeout: rfid parameter = {rfid}")
            print(f"🔍 DEBUG handle_permanent_student_timeout: student_info keys = {list(student_info.keys()) if student_info else 'None'}")
            print(f"🔍 DEBUG handle_permanent_student_timeout: student_info rfid field = {student_info.get('rfid', 'NOT FOUND') if student_info else 'N/A'}")
            
            # CRITICAL: Ensure RFID and Student Number are in student_info before proceeding
            if not student_info.get('rfid') and rfid:
                student_info['rfid'] = rfid
                print(f"🔍 DEBUG: Added RFID to student_info in handle_permanent_student_timeout: {rfid}")
            
            # Ensure Student Number is present - fetch from Firebase if missing
            if rfid:
                student_info = self._ensure_student_number_from_firebase(student_info, rfid)
                # Validate Student Number
                student_num = student_info.get('student_id') or student_info.get('student_number')
                if not student_num or str(student_num).strip() == str(rfid).strip():
                    print(f"🔍 DEBUG: Student Number missing or equals RFID in timeout, fetching from Firebase...")
                    full_student_info = self.get_student_info_by_rfid(rfid)
                    if full_student_info:
                        student_info.update(full_student_info)
                        print(f"✅ Updated student_info with complete Firebase data in timeout handler")
                        print(f"✅ Student Number: {student_info.get('student_id')}")
            
            # SAFELY stop detection to prevent camera conflicts
            if hasattr(self, 'detection_system') and self.detection_system:
                print("🛑 Safely stopping detection system...")
                self.detection_system.stop_detection()
                time.sleep(0.5)  # Wait for detection to stop properly
            
            # Check if detection was active
            if getattr(self, 'detection_active', False):
                print("🛑 Detection was active - ensuring clean stop")
                self.detection_active = False
            
            # Restart guard camera feed (no detection) after permanent student exit
            try:
                print("🔄 Restarting guard camera feed...")
                self.initialize_guard_camera_feed()
                # Hide requirements section when student exits
                self.hide_requirements_section()
                print("✅ Guard camera feed restarted")
            except Exception as e:
                print(f"⚠️ Error restarting guard camera feed: {e}")
            
            # Check if Event Mode is active
            event_mode = self.event_mode_active
            if event_mode:
                self.add_activity_log(f"Student Time-Out (Event Mode - No Detection): {student_info['name']} (RFID: {rfid}) - {current_time}")
            else:
                self.add_activity_log(f"Student Time-Out (Permanent): {student_info['name']} (RFID: {rfid}) - {current_time}")
            
            # Finalize all pending violations when student exits
            self.finalize_session_violations(rfid)
            
            # Save to Firebase (with event_mode flag)
            self.save_permanent_student_activity_to_firebase(rfid, 'time_out', current_time, student_info, event_mode=event_mode)
            
            # CRITICAL: Clear entry from local cache when student exits
            # This allows student to enter again later
            if hasattr(self, 'recent_entry_cache') and rfid in self.recent_entry_cache:
                del self.recent_entry_cache[rfid]
                print(f"✅ Removed RFID {rfid} from recent entry cache (student exited)")
            
            # Update main screen - pass RFID explicitly to ensure Student Number can be fetched
            if self.main_screen_window and self.main_screen_window.winfo_exists():
                self.update_main_screen_with_permanent_student(student_info, 'time_out', current_time, rfid=rfid)
            
            # Ensure recent entries shows this as EXIT for permanent student
            try:
                person_data_exit = {
                    'id': student_info.get('student_id', rfid),
                    'name': student_info.get('name', f'Student {rfid}'),
                    'type': 'student',
                    'course': student_info.get('course', 'Unknown'),
                    'gender': student_info.get('gender', 'Unknown'),
                    'timestamp': current_time,
                    'status': 'TIME-OUT',
                    'action': 'EXIT',
                    'guard_id': self.current_guard_id or 'Unknown'
                }
                self.add_to_recent_entries(person_data_exit)
            except Exception:
                pass
            # Show success message
            self.show_green_success_message("Student Time-Out", 
                              f"Student: {student_info['name']}\n"
                              f"Course: {student_info['course']}\n"
                              f"Student ID: {student_info['student_id']}\n"
                              f"Permanent RFID: {rfid}\n"
                              f"Time-Out: {current_time}\n\n"
                              f"Note: Detection stopped")
            
        except Exception as e:
            print(f"ERROR: Error handling permanent student time-out: {e}")
            self.add_activity_log(f"Error processing time-out for permanent student RFID {rfid}: {e}")
    
    def save_permanent_student_activity_to_firebase(self, rfid, activity_type, timestamp, student_info, event_mode=False):
        """Save permanent student activity to Firebase"""
        try:
            if self.firebase_initialized and self.db:
                activity_data = {
                    'student_id': student_info['student_id'],
                    'name': student_info['name'],
                    'rfid': rfid,
                    'activity_type': activity_type,  # 'time_in' or 'time_out'
                    'timestamp': timestamp,
                    'course': student_info['course'],
                    'student_type': 'permanent',
                    'event_mode': event_mode  # Flag indicating event mode (detection bypassed)
                }
                
                # Save to student_activities collection (using student_id_timestamp format)
                student_id = student_info.get('student_id')
                formatted_timestamp = self.format_document_id(timestamp)
                doc_id = f"{student_id}_{formatted_timestamp}"
                activity_ref = self.db.collection('student_activities').document(doc_id)
                activity_ref.set(activity_data)
                print(f"✅ SUCCESS: Saved to student_activities collection - Document ID: {doc_id}")
                print(f"   Fields: activity_type={activity_type}, timestamp={timestamp}, event_mode={event_mode}")
                
                # Also update/maintain student document in students collection
                if student_id:
                    student_ref = self.db.collection('students').document(student_id)
                    student_doc = student_ref.get()
                    
                    student_update_data = {
                        'student_id': student_id,
                        'name': student_info.get('name', 'Unknown'),
                        'course': student_info.get('course', 'Unknown'),
                        'last_updated': timestamp
                    }
                    
                    # Update time_in or time_out based on activity type
                    if activity_type == 'time_in':
                        student_update_data['time_in'] = timestamp
                        student_update_data['status'] = 'active'
                        print(f"✅ SUCCESS: Updated time_in field in students/{student_id} = {timestamp}")
                    elif activity_type == 'time_out':
                        student_update_data['time_out'] = timestamp
                        student_update_data['status'] = 'exited'
                        print(f"✅ SUCCESS: Updated time_out field in students/{student_id} = {timestamp}")
                    
                    # Set created_at only if document doesn't exist
                    if not student_doc.exists:
                        student_update_data['created_at'] = timestamp
                    
                    # Update or create student document
                    student_ref.set(student_update_data, merge=True)
                    print(f"✅ SUCCESS: Student document updated in Firebase: students/{student_id}")
                
                print(f"✅ SUCCESS: Permanent student activity saved to Firebase: {activity_type} for {student_info['name']}")
                print(f"📍 Firebase locations:")
                print(f"   - Full history: student_activities/{doc_id}")
                print(f"   - Latest time: students/{student_id} (time_in/time_out fields)")
            else:
                print("WARNING: Firebase not available - permanent student activity not saved to cloud")
                
        except Exception as e:
            print(f"ERROR: Failed to save permanent student activity to Firebase: {e}")
    
    def update_main_screen_with_permanent_student(self, student_info, activity_type, timestamp, rfid=None):
        """Update main screen with permanent student information - Enhanced to prevent Unknown status"""
        try:
            if not self.main_screen_window or not self.main_screen_window.winfo_exists():
                return
            
            # Ensure we have valid student info
            if not student_info:
                print("ERROR: No student info provided to update_main_screen_with_permanent_student")
                return
            
            # Create person info for main screen with proper fallbacks
            # Use Student Number (student_id) as the ID (not RFID)
            # CRITICAL: Get RFID from parameter first, then from student_info
            if not rfid:
                rfid = student_info.get('rfid')
            
            # If still no RFID, try to infer from student_info
            if not rfid:
                # Check if 'id' field might be RFID (10-digit numeric string)
                potential_id = student_info.get('id')
                if potential_id and str(potential_id).strip().isdigit() and len(str(potential_id).strip()) == 10:
                    rfid = str(potential_id).strip()
                    print(f"🔍 DEBUG: Inferred RFID from 'id' field: {rfid}")
                else:
                    rfid = 'Unknown RFID'
            
            if not rfid or rfid == 'Unknown RFID':
                print(f"⚠️ WARNING: No RFID available - keys: {list(student_info.keys())}")
                print(f"⚠️ Cannot fetch Student Number without RFID!")
            
            print(f"🔍 ========== update_main_screen_with_permanent_student ==========")
            print(f"🔍 student_info keys: {list(student_info.keys()) if student_info else 'None'}")
            print(f"🔍 student_info rfid: {student_info.get('rfid') if student_info else 'N/A'}")
            print(f"🔍 student_info student_id: {student_info.get('student_id') if student_info else 'N/A'}")
            print(f"🔍 rfid variable: {rfid}")
            
            # CRITICAL: Ensure Student Number is present before creating person_info
            student_num = student_info.get('student_id') or student_info.get('student_number')
            print(f"🔍 Initial student_num: {student_num}")
            
            # Validate Student Number - must not be None, empty, 'Unknown', or equal to RFID
            is_valid_student_num = (student_num and 
                                  student_num != 'Unknown' and 
                                  str(student_num).strip() != '' and
                                  (not rfid or rfid == 'Unknown RFID' or str(student_num).strip() != str(rfid).strip()))
            
            if not is_valid_student_num:
                # Fetch from Firebase if missing or invalid
                if rfid and rfid != 'Unknown RFID':
                    print(f"🔍 Student Number missing/invalid in permanent student display, fetching from Firebase for RFID: {rfid}")
                    fresh_student_info = self.get_student_info_by_rfid(rfid)
                    if fresh_student_info:
                        student_num = fresh_student_info.get('student_id') or fresh_student_info.get('student_number')
                        print(f"🔍 Fetched student_num from Firebase: {student_num}")
                        # Validate that we got a valid student number (not None, not empty, not RFID)
                        if student_num and str(student_num).strip() != '' and str(student_num).strip() != str(rfid).strip():
                            # Update student_info with complete data
                            student_info.update(fresh_student_info)
                            # CRITICAL: Preserve RFID after update
                            student_info['rfid'] = rfid
                            print(f"✅ Updated student_info with Student Number from Firebase: {student_num}")
                        else:
                            print(f"⚠️ WARNING: Fetched Student Number from Firebase is invalid (None, empty, or equals RFID): {student_num}")
                            student_num = None
                    else:
                        print(f"⚠️ WARNING: get_student_info_by_rfid returned None for RFID: {rfid}")
                else:
                    print(f"⚠️ WARNING: Cannot fetch Student Number - RFID is invalid: {rfid}")
            
            print(f"🔍 Final student_num before creating person_info: {student_num}")
            
            # Final attempt: If student_num is still None, try one more direct Firebase lookup
            if not student_num and rfid and rfid != 'Unknown RFID':
                print(f"🔍 Final attempt: Fetching Student Number directly from Firebase for RFID: {rfid}")
                final_student_info = self.get_student_info_by_rfid(rfid)
                if final_student_info:
                    student_num = final_student_info.get('student_id') or final_student_info.get('student_number')
                    if student_num and str(student_num).strip() != '' and str(student_num).strip() != str(rfid).strip():
                        print(f"✅ Final fetch successful - Student Number: {student_num}")
                        # Update student_info with the complete data
                        student_info.update(final_student_info)
                        student_info['rfid'] = rfid  # Preserve RFID
                    else:
                        print(f"⚠️ Final fetch returned invalid Student Number: {student_num}")
                        student_num = None
                else:
                    print(f"⚠️ Final fetch returned None from Firebase for RFID: {rfid}")
            
            # Strip whitespace from course if it exists
            course_val = student_info.get('course', 'Unknown Course')
            if course_val:
                course_val = str(course_val).strip()
            
            # Only show 'Student Number Not Found' if we truly couldn't retrieve it
            final_student_id = student_num if student_num else 'Student Number Not Found'
            
            # Check if this is Event Mode entry (check if event_mode flag is set in student_info)
            is_event_mode = student_info.get('event_mode', False) or getattr(self, 'event_mode_active', False)
            
            # Set status based on activity type and event mode
            if activity_type == 'time_in':
                if is_event_mode:
                    status = 'TIME-IN (EVENT MODE)'
                else:
                    status = 'TIME-IN'
            else:  # time_out
                if is_event_mode:
                    status = 'TIME-OUT (EVENT MODE)'
                else:
                    status = 'TIME-OUT'
            
            person_info = {
                'id': final_student_id,  # Use Student Number as ID
                'student_id': final_student_id,  # Student Number from Firebase
                'student_number': final_student_id,  # Alternative field name
                'rfid': rfid,  # Keep RFID for reference (not displayed as ID)
                'name': student_info.get('name', 'Unknown Student'),
                'type': 'student',
                'course': course_val,  # Course with whitespace stripped
                'gender': student_info.get('gender', 'Unknown Gender'),
                'timestamp': timestamp,
                'status': status,  # Set based on activity type and event mode
                'event_mode': is_event_mode,  # Flag to indicate event mode
                # Note: 'guard_id' removed - not displayed anymore
                # NOTE: Gmail is NOT included for privacy
            }
            
            print(f"🔍 person_info created with id: {person_info.get('id')}, student_id: {person_info.get('student_id')}")
            
            # Update main screen
            self.update_main_screen_with_person(person_info)
            print(f"SUCCESS: Main screen updated with permanent student {activity_type}: {student_info['name']} - Status: {person_info['status']} - Student Number: {person_info.get('student_id')}")
            
        except Exception as e:
            print(f"ERROR: Error updating main screen with permanent student: {e}")
    
    def init_arduino_connection(self):
        """Initialize Arduino serial connection for gate control"""
        try:
            try:
                import serial
                import serial.tools.list_ports
            except ImportError:
                print("WARNING: pyserial not available - Arduino connection disabled")
                self.arduino_connected = False
                return
            
            # Find Arduino port
            arduino_port = None
            ports = serial.tools.list_ports.comports()
            
            for port in ports:
                if 'Arduino' in port.description or 'USB' in port.description:
                    arduino_port = port.device
                    break
            
            if not arduino_port:
                # Try common Arduino ports
                common_ports = ['/dev/ttyUSB0', '/dev/ttyACM0', '/dev/ttyACM1', 'COM3', 'COM4']
                for port in common_ports:
                    try:
                        test_serial = serial.Serial(port, 9600, timeout=1)
                        test_serial.close()
                        arduino_port = port
                        break
                    except:
                        continue
            
            if arduino_port:
                self.arduino_serial = serial.Serial(arduino_port, 9600, timeout=1)
                self.arduino_connected = True
                print(f"SUCCESS: Arduino connected on port: {arduino_port}")
                self.add_activity_log(f"Arduino connected on port: {arduino_port}")
                
                # Update Arduino connection status in UI (only if dashboard exists)
                if hasattr(self, 'arduino_status_label'):
                    self.root.after(0, self.update_arduino_connection_status, True)
                
                return True
            else:
                print("WARNING: Arduino not found - gate control disabled")
                self.arduino_connected = False
                self.add_activity_log("Arduino not found - gate control disabled")
                
                # Update Arduino connection status in UI (only if dashboard exists)
                if hasattr(self, 'arduino_status_label'):
                    self.root.after(0, self.update_arduino_connection_status, False)
                
                return False
                
        except ImportError:
            print("WARNING: PySerial not installed - Arduino communication disabled")
            print("💡 Install with: pip install pyserial")
            self.arduino_connected = False
            return False
        except Exception as e:
            print(f"ERROR: Error initializing Arduino: {e}")
            self.arduino_connected = False
            return False
    
    def send_arduino_command(self, command):
        """Send command to Arduino for gate control"""
        try:
            if self.arduino_connected and hasattr(self, 'arduino_serial'):
                self.arduino_serial.write(command.encode())
                print(f"📤 Sent to Arduino: {command}")
                self.add_activity_log(f"Arduino command sent: {command}")
                return True
            else:
                print("WARNING: Arduino not connected - command not sent")
                return False
        except Exception as e:
            print(f"ERROR: Error sending Arduino command: {e}")
            self.add_activity_log(f"Arduino command failed: {e}")
            return False
    
    def open_gate(self):
        """Open the gate via Arduino"""
        try:
            success = self.send_arduino_command("OPEN")
            if success:
                self.add_activity_log("Gate opened via Arduino")
                print("🚪 Gate opened")
            return success
        except Exception as e:
            print(f"ERROR: Error opening gate: {e}")
            return False
    
    def close_gate(self):
        """Close the gate via Arduino"""
        try:
            success = self.send_arduino_command("CLOSE")
            if success:
                self.add_activity_log("Gate closed via Arduino")
                print("🚪 Gate closed")
            return success
        except Exception as e:
            print(f"ERROR: Error closing gate: {e}")
            return False
    
    def handle_approve_button(self):
        """Handle approve button press from Arduino"""
        try:
            print("SUCCESS: Approve button pressed - Opening gate")
            self.add_activity_log("Approve button pressed - Opening gate")
            
            # CRITICAL: Use stored student info from incomplete uniform detection (similar to approve_complete_uniform)
            student_info = None
            current_rfid = None
            
            # First try to get from stored incomplete student info
            if hasattr(self, 'incomplete_student_info_for_approve') and self.incomplete_student_info_for_approve:
                student_info = self.incomplete_student_info_for_approve
                current_rfid = student_info.get('rfid')
                print(f"✅ Using stored incomplete student info for APPROVE: {student_info.get('name', 'Unknown')}")
            
            # Fallback: Get current student info from detection system
            if not current_rfid:
                if hasattr(self, 'detection_system') and self.detection_system:
                    detection_service = getattr(self.detection_system, 'detection_service', None)
                    if detection_service:
                        current_rfid = getattr(detection_service, 'current_rfid', None)
            
            # Try backup RFID storage
            if not current_rfid and hasattr(self, 'current_rfid_for_timer'):
                current_rfid = self.current_rfid_for_timer
            
            # If we don't have student_info yet, fetch it
            if not student_info and current_rfid:
                student_info = self.get_student_info_by_rfid(current_rfid)
            
            # Check if we have student info and RFID to proceed
            if not current_rfid or not student_info:
                print(f"⚠️ WARNING: Cannot process APPROVE - missing RFID or student info")
                return False
            
            # CRITICAL: Handle APPROVE button for BOTH incomplete AND complete uniform cases
            # Case 1: Incomplete uniform (has pending violations)
            if current_rfid in self.active_session_violations and len(self.active_session_violations[current_rfid]) > 0:
                    print(f"INFO: Approve pressed with incomplete uniform - finalizing violations and allowing entry")
                    # Finalize all pending violations (student enters with violation recorded)
                    self.finalize_session_violations(current_rfid)
                    
                    # Ensure we have complete student info
                    if not student_info:
                        student_info = self.get_student_info_by_rfid(current_rfid)
                    
                    if student_info:
                        student_id = student_info.get('student_id') or student_info.get('student_number')
                        if student_id:
                            current_time = self.get_current_timestamp()
                            
                            # CRITICAL: Save approved violation flag for 24-hour detection skip
                            try:
                                self.save_approved_violation_flag(student_id, current_rfid, current_time)
                                print(f"✅ Approved violation flag saved - detection will be skipped for 24 hours")
                            except Exception as e:
                                print(f"⚠️ WARNING: Could not save approved violation flag: {e}")
                                # Continue anyway - don't block entry if flag save fails
                            
                            # CRITICAL: Save entry to Firebase (same as record_complete_uniform_entry)
                            try:
                                # Ensure RFID is in student_info
                                if not student_info.get('rfid'):
                                    student_info['rfid'] = current_rfid
                                
                                # Save entry to Firebase
                                self.save_permanent_student_activity_to_firebase(
                                    current_rfid, 
                                    'time_in', 
                                    current_time, 
                                    student_info, 
                                    event_mode=False
                                )
                                print(f"✅ SUCCESS: Entry saved to Firebase for {student_info.get('name', 'Unknown')}")
                                
                                # CRITICAL: Add to local cache IMMEDIATELY after saving (before verification)
                                # Entry was saved, so we should cache it right away to ensure next tap is recognized as EXIT
                                # This prevents race condition where Firebase query might be slow
                                import time
                                if not hasattr(self, 'recent_entry_cache'):
                                    self.recent_entry_cache = {}
                                self.recent_entry_cache[current_rfid] = time.time()
                                print(f"✅ Added RFID {current_rfid} to recent entry cache immediately after save")
                                print(f"   - Cache timestamp: {self.recent_entry_cache[current_rfid]}")
                                print(f"   - Next tap will be recognized as EXIT")
                                
                                # CRITICAL: Force delay and retry logic to ensure Firebase write is committed
                                # Try up to 3 times to verify entry was saved (with increasing delays)
                                verify_entry = False
                                for attempt in range(3):
                                    delay = 0.5 + (attempt * 0.3)  # 0.5s, 0.8s, 1.1s
                                    time.sleep(delay)
                                    verify_entry = self.check_student_has_entry_record(current_rfid)
                                    if verify_entry:
                                        print(f"✅ VERIFIED: Entry record confirmed in Firebase (attempt {attempt + 1})")
                                        break
                                    else:
                                        print(f"⚠️ Attempt {attempt + 1}: Entry record not yet found, retrying...")
                                
                                if not verify_entry:
                                    print(f"⚠️ WARNING: Entry record not found after 3 attempts - may need more time or check manually")
                                    print(f"   - Cache is still active, so next tap will be recognized as EXIT")
                                else:
                                    print(f"✅ SUCCESS: Entry verified in Firebase - safe to proceed")
                            except Exception as e:
                                print(f"❌ ERROR: Failed to save entry to Firebase: {e}")
                                import traceback
                                traceback.print_exc()
                            
                            # CRITICAL: Reset detection flags BEFORE updating main screen
                            # This ensures main screen shows ENTRY status, not DETECTING
                            try:
                                # Reset detection flags FIRST (before main screen update)
                                self.detection_active = False
                                self.uniform_detection_complete = True  # Mark as complete since approved
                                print(f"✅ Detection flags reset BEFORE main screen update: detection_active=False, uniform_detection_complete=True")
                                
                                # Clear detection session tracking
                                if current_rfid in self.active_detection_sessions:
                                    del self.active_detection_sessions[current_rfid]
                                    print(f"✅ Cleared detection session for RFID {current_rfid}")
                                
                                # Stop any running detection
                                if hasattr(self, 'detection_system') and self.detection_system:
                                    self.detection_system.stop_detection()
                                    print(f"✅ Stopped detection system after approval")
                                
                                # Hide requirements section
                                try:
                                    self.hide_requirements_section()
                                except Exception:
                                    pass
                                
                                print(f"✅ Detection session cleared - next tap will be recognized as EXIT")
                            except Exception as e:
                                print(f"⚠️ WARNING: Could not clear detection session: {e}")
                            
                            # Update main screen with student entry info (AFTER flags are reset)
                            try:
                                if self.main_screen_window and self.main_screen_window.winfo_exists():
                                    self.update_main_screen_with_permanent_student(
                                        student_info, 
                                        'time_in', 
                                        current_time, 
                                        rfid=current_rfid
                                    )
                                    print(f"✅ SUCCESS: Main screen updated with student entry info (detection flags already reset)")
                            except Exception as e:
                                print(f"❌ ERROR: Failed to update main screen: {e}")
                                import traceback
                                traceback.print_exc()
                            
                            # Add to recent entries
                            try:
                                student_number = student_info.get('student_id') or student_info.get('student_number') or current_rfid
                                person_data = {
                                    'id': student_number,
                                    'student_id': student_number,
                                    'name': student_info.get('name', f'Student {current_rfid}'),
                                    'type': 'student',
                                    'course': student_info.get('course', 'Unknown'),
                                    'gender': student_info.get('gender', 'Unknown'),
                                    'timestamp': current_time,
                                    'status': 'TIME-IN (WITH VIOLATION)',
                                    'action': 'ENTRY',
                                    'guard_id': self.current_guard_id or 'Unknown'
                                }
                                self.add_to_recent_entries(person_data)
                                print(f"✅ SUCCESS: Added to recent entries")
                            except Exception as e:
                                print(f"❌ ERROR: Failed to add to recent entries: {e}")
                                import traceback
                                traceback.print_exc()
                            
                            # Record entry even with violation (student is entering with violation)
                            self.add_activity_log(f"Student Entry (With Violation): {student_info.get('name', 'Unknown')} - {current_time}")
                            print(f"INFO: Student entering with violation - entry recorded and saved to Firebase")
                            
                            # CRITICAL: Disable buttons after processing
                            if hasattr(self, 'approve_button'):
                                self.approve_button.config(state=tk.DISABLED)
                                print(f"✅ APPROVE button disabled after processing")
                            
                            if hasattr(self, 'cancel_button'):
                                self.cancel_button.config(state=tk.DISABLED)
                                print(f"✅ CANCEL button disabled after processing")
                            
                            # Removed approve_complete_uniform_btn - no longer needed
                            
                            # Clear stored incomplete student info
                            self.incomplete_student_info_for_approve = None
                            print(f"✅ Cleared incomplete student info for APPROVE")
                            
                            # Open gate and show success (entry already saved above)
                            self.open_gate()
                            
                            # Update main screen (already updated above, but ensure gate status is shown)
                            if self.main_screen_window and self.main_screen_window.winfo_exists():
                                # Don't overwrite entry status - it's already set by update_main_screen_with_permanent_student
                                pass
                            
                            # Show success message
                            self.show_green_success_message("Entry Approved (With Violation)", 
                                              "SUCCESS: Entry recorded with violation\n"
                                              "🚪 Gate is opening\n"
                                              "👤 Student may proceed\n"
                                              "⚠️ Violation recorded")
                            
                            # Return True to indicate success
                            return True
                        else:
                            print(f"⚠️ WARNING: Student info missing student_id - cannot save entry")
                            return False
                    else:
                        print(f"⚠️ WARNING: Student info is None - cannot save entry")
                        return False
            else:
                # No pending violations - APPROVE button shouldn't be enabled for this case
                # But if somehow clicked, log warning and don't process
                print(f"⚠️ WARNING: APPROVE button clicked but no pending violations found")
                print(f"   Current RFID: {current_rfid}")
                print(f"   active_session_violations keys: {list(getattr(self, 'active_session_violations', {}).keys())}")
                if current_rfid:
                    print(f"   Violations for {current_rfid}: {len(getattr(self, 'active_session_violations', {}).get(current_rfid, []))}")
                print(f"   This should not happen - APPROVE button should be disabled when no incomplete uniform")
                # Don't process - just log and return
                return False
            
        except Exception as e:
            print(f"ERROR: Error handling approve button: {e}")
            return False
    
    def handle_deny_button(self):
        """Handle deny button press from Arduino - Only lock gate and log, no violation finalization"""
        try:
            print("ERROR: Deny button pressed - Keeping gate closed")
            self.add_activity_log("Guard denied entry - gate locked")
            
            # Keep gate closed
            self.close_gate()
            
            # Do NOT finalize violations - just log the denial
            # Violations remain pending (student may still use "Approve Complete Uniform" button later)
            print(f"INFO: Deny button pressed - gate locked, violations remain pending")
            
            # Update main screen
            if self.main_screen_window and self.main_screen_window.winfo_exists():
                self.update_main_screen_with_gate_status("CLOSED", "DENIED")
            
            # Show warning message
            self.show_red_warning_message("Gate Denied", 
                              "ERROR: Uniform violation detected\n"
                              "🚪 Gate remains closed\n"
                              "👤 Person must correct uniform")
            
            return True
            
        except Exception as e:
            print(f"ERROR: Error handling deny button: {e}")
            return False
    
    def update_main_screen_with_gate_status(self, gate_status, decision):
        """Update main screen with gate status"""
        try:
            if not self.main_screen_window or not self.main_screen_window.winfo_exists():
                return
            
            print(f"📺 Main screen updated - Gate: {gate_status}, Decision: {decision}")
            
        except Exception as e:
            print(f"ERROR: Error updating main screen with gate status: {e}")
    
    def show_red_warning_message(self, title, message):
        """Show a warning message with red color"""
        try:
            # Create a custom messagebox window
            warning_window = tk.Toplevel(self.root)
            warning_window.title(title)
            warning_window.geometry("400x150")
            warning_window.configure(bg='#fef2f2')
            warning_window.resizable(False, False)
            
            # Ensure it's a child of the Guard UI window only
            warning_window.transient(self.root)
            warning_window.grab_set()
            
            # Center on the Guard UI window
            self.root.update_idletasks()
            guard_x = self.root.winfo_x()
            guard_y = self.root.winfo_y()
            guard_width = self.root.winfo_width()
            guard_height = self.root.winfo_height()
            
            x = guard_x + (guard_width // 2) - (400 // 2)
            y = guard_y + (guard_height // 2) - (150 // 2)
            warning_window.geometry(f"400x150+{x}+{y}")
            
            warning_window.lift(self.root)
            warning_window.attributes('-topmost', True)
            
            # Main frame
            main_frame = tk.Frame(warning_window, bg='#fef2f2')
            main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
            
            # Warning icon (red X)
            icon_label = tk.Label(
                main_frame,
                text="ERROR:",
                font=('Arial', 24),
                bg='#fef2f2',
                fg='#dc2626'
            )
            icon_label.pack(pady=(0, 10))
            
            # Title
            title_label = tk.Label(
                main_frame,
                text=title,
                font=('Arial', 16, 'bold'),
                bg='#fef2f2',
                fg='#dc2626'
            )
            title_label.pack(pady=(0, 5))
            
            # Message
            message_label = tk.Label(
                main_frame,
                text=message,
                font=('Arial', 12),
                bg='#fef2f2',
                fg='#1f2937',
                wraplength=350,
                justify=tk.CENTER
            )
            message_label.pack(pady=(0, 15))
            
            # OK button
            ok_button = tk.Button(
                main_frame,
                text="OK",
                font=('Arial', 12, 'bold'),
                bg='#dc2626',
                fg='white',
                relief='raised',
                bd=2,
                padx=20,
                pady=5,
                cursor='hand2',
                activebackground='#b91c1c',
                activeforeground='white',
                command=warning_window.destroy
            )
            ok_button.pack()
            
            # Auto-close after 5 seconds
            warning_window.after(5000, warning_window.destroy)
            
            # Focus on the window
            warning_window.focus_set()
            
        except Exception as e:
            print(f"ERROR: Error showing red warning message: {e}")
            # Fallback to regular messagebox
            messagebox.showwarning(title, message)
    
    def listen_for_arduino_buttons(self):
        """Listen for button presses from Arduino"""
        try:
            if self.arduino_connected and hasattr(self, 'arduino_serial'):
                # Check for incoming data
                if self.arduino_serial.in_waiting > 0:
                    data = self.arduino_serial.readline().decode().strip()
                    
                    if data == "APPROVE":
                        self.handle_approve_button()
                    elif data == "DENY":
                        self.handle_deny_button()
                    elif data:
                        print(f"📥 Arduino data: {data}")
                        self.add_activity_log(f"Arduino data received: {data}")
                
                # Schedule next check
                self.root.after(100, self.listen_for_arduino_buttons)
                
        except Exception as e:
            print(f"ERROR: Error listening for Arduino buttons: {e}")
            # Retry after delay
            self.root.after(1000, self.listen_for_arduino_buttons)
    
    def process_detection_result_for_gate(self, detection_result):
        """Process detection result and prepare for gate control"""
        try:
            if not detection_result:
                return
            
            # Extract detection information
            compliant = detection_result.get('compliant', False)
            violations = detection_result.get('violations', [])
            confidence = detection_result.get('confidence', 0)
            
            # Log detection result
            if compliant:
                self.add_activity_log(f"SUCCESS: Uniform compliant - Ready for approval (Confidence: {confidence:.2f})")
                print("SUCCESS: Uniform compliant - Waiting for guard approval")
            else:
                violation_text = ", ".join(violations) if violations else "Unknown violation"
                self.add_activity_log(f"ERROR: Uniform violation detected: {violation_text} (Confidence: {confidence:.2f})")
                print(f"ERROR: Uniform violation: {violation_text}")
                
                # Save uniform violation to Firebase when detected (as pending - will finalize when student exits/gives up)
                try:
                    # Get current student info from detection system
                    current_rfid = None
                    if hasattr(self, 'detection_system') and self.detection_system:
                        detection_service = getattr(self.detection_system, 'detection_service', None)
                        if detection_service:
                            current_rfid = getattr(detection_service, 'current_rfid', None)
                    
                    # Try backup RFID storage
                    if not current_rfid and hasattr(self, 'current_rfid_for_timer'):
                        current_rfid = self.current_rfid_for_timer
                    
                    if current_rfid:
                        # Get student info
                        student_info = self.get_student_info_by_rfid(current_rfid)
                        if student_info:
                            student_id = student_info.get('student_id')
                            if student_id:
                                current_time = self.get_current_timestamp()
                                
                                # Extract missing items from violations list if available
                                # CRITICAL: Only use violations list for missing_items - this contains actual missing items
                                # Do NOT use tracker's get_missing_components() or checkboxes as they may list all
                                # unconfirmed required items, not just actual missing items
                                missing_items = violations if violations and len(violations) > 0 else []
                                
                                # Note: We do NOT fall back to checkboxes or tracker here because:
                                # 1. Checkboxes may be auto-populated from tracker state (showing all unconfirmed items)
                                # 2. Tracker's get_missing_components() lists all unconfirmed required parts
                                # 3. Only the violations list from detection_result contains actual missing items
                                
                                if missing_items:
                                    print(f"✅ Using violations list for missing_items: {missing_items}")
                                    print(f"   - Only actual missing items will be stored in violation history")
                                else:
                                    print(f"⚠️ No violations list available - missing_items will be empty")
                                    print(f"   - This is correct: we only store actual missing items, not all unconfirmed items")
                                
                                # Check if no violations detected at all (empty detection)
                                if not violations or len(violations) == 0:
                                    if missing_items and len(missing_items) > 0:
                                        violation_details = f"Uniform incomplete - Missing: {', '.join(missing_items)}"
                                    else:
                                        violation_details = "No uniform parts detected - empty detection result"
                                else:
                                    violation_details = f"Uniform violation detected: {violation_text} (Confidence: {confidence:.2f})"
                                    # Add missing items to details if not already included
                                    if missing_items and len(missing_items) > 0 and violation_text:
                                        if ', '.join(missing_items) not in violation_text:
                                            violation_details += f" - Missing: {', '.join(missing_items)}"
                                
                                # Save as pending (is_final=False) - will be finalized when student exits/gives up
                                self.save_student_violation_to_firebase(
                                    student_id=student_id,
                                    violation_type='uniform_violation',
                                    timestamp=current_time,
                                    student_info=student_info,
                                    violation_details=violation_details,
                                    is_final=False,  # Pending until student gives up
                                    rfid=current_rfid,
                                    missing_items=missing_items  # Pass missing items list
                                )
                except Exception as e:
                    print(f"WARNING: Failed to save uniform violation to Firebase: {e}")
            
            # Update main screen with detection result
            if self.main_screen_window and self.main_screen_window.winfo_exists():
                self.update_main_screen_with_detection_result(detection_result)
            
            # Show detection result to guard
            if compliant:
                self.show_detection_result_message("Uniform Compliant", 
                                    f"SUCCESS: Uniform compliance verified\n"
                                    f"Confidence: {confidence:.2f}\n\n"
                                    f"Press APPROVE button to open gate\n"
                                    f"Press DENY button to keep gate closed")
            else:
                violation_text = ", ".join(violations) if violations else "Unknown violation"
                self.show_detection_result_message("Uniform Violation", 
                                    f"ERROR: Uniform violation detected\n"
                                    f"Violations: {violation_text}\n"
                                    f"Confidence: {confidence:.2f}\n\n"
                                    f"Press APPROVE button to open gate anyway\n"
                                    f"Press DENY button to keep gate closed")
            
        except Exception as e:
            print(f"ERROR: Error processing detection result for gate: {e}")
    
    def show_detection_result_message(self, title, message):
        """Show detection result message"""
        try:
            # Create a custom messagebox window
            result_window = tk.Toplevel(self.root)
            result_window.title(title)
            result_window.geometry("450x200")
            result_window.configure(bg='#f8fafc')
            result_window.resizable(False, False)
            
            # Ensure it's a child of the Guard UI window only
            result_window.transient(self.root)
            result_window.grab_set()
            
            # Center on the Guard UI window
            self.root.update_idletasks()
            guard_x = self.root.winfo_x()
            guard_y = self.root.winfo_y()
            guard_width = self.root.winfo_width()
            guard_height = self.root.winfo_height()
            
            x = guard_x + (guard_width // 2) - (450 // 2)
            y = guard_y + (guard_height // 2) - (200 // 2)
            result_window.geometry(f"450x200+{x}+{y}")
            
            result_window.lift(self.root)
            result_window.attributes('-topmost', True)
            
            # Main frame
            main_frame = tk.Frame(result_window, bg='#f8fafc')
            main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
            
            # Title
            title_label = tk.Label(
                main_frame,
                text=title,
                font=('Arial', 16, 'bold'),
                bg='#f8fafc',
                fg='#1f2937'
            )
            title_label.pack(pady=(0, 10))
            
            # Message
            message_label = tk.Label(
                main_frame,
                text=message,
                font=('Arial', 12),
                bg='#f8fafc',
                fg='#374151',
                wraplength=400,
                justify=tk.CENTER
            )
            message_label.pack(pady=(0, 15))
            
            # Instructions
            instruction_label = tk.Label(
                main_frame,
                text="Use Arduino APPROVE/DENY buttons to control gate",
                font=('Arial', 10, 'italic'),
                bg='#f8fafc',
                fg='#6b7280'
            )
            instruction_label.pack(pady=(0, 15))
            
            # OK button
            ok_button = tk.Button(
                main_frame,
                text="OK",
                font=('Arial', 12, 'bold'),
                bg='#3b82f6',
                fg='white',
                relief='raised',
                bd=2,
                padx=20,
                pady=5,
                cursor='hand2',
                activebackground='#2563eb',
                activeforeground='white',
                command=result_window.destroy
            )
            ok_button.pack()
            
            # Auto-close after 10 seconds
            result_window.after(10000, result_window.destroy)
            
            # Focus on the window
            result_window.focus_set()
            
        except Exception as e:
            print(f"ERROR: Error showing detection result message: {e}")
            # Fallback to regular messagebox
            messagebox.showinfo(title, message)
    
    def save_guard_login_to_firebase(self, guard_id):
        """Save guard login activity to Firebase"""
        try:
            if self.firebase_initialized and self.db:
                login_data = {
                    'guard_id': guard_id,
                    'session_id': self.session_id,
                    'login_time': self.get_current_timestamp(),
                    'login_type': 'manual_login',
                    'status': 'active'
                }
                
                # Save to guard_logins collection
                login_ref = self.db.collection('guard_logins').document(self.session_id)
                login_ref.set(login_data)
                
                print(f"SUCCESS: Guard login saved to Firebase: {guard_id}")
                
                # Add to activity log
                self.add_activity_log(f"Guard Login: {guard_id} - {login_data['login_time']}")
                
            else:
                print("WARNING: Firebase not available - guard login not saved to cloud")
                # Still add to activity log for local tracking
                self.add_activity_log(f"Guard Login: {guard_id} - {self.get_current_timestamp()}")
                
        except Exception as e:
            print(f"ERROR: Failed to save guard login to Firebase: {e}")
            # Still add to activity log for local tracking
            self.add_activity_log(f"Guard Login: {guard_id} - {self.get_current_timestamp()}")
    
    def save_guard_logout_to_firebase(self, guard_id):
        """Save guard logout activity to Firebase"""
        try:
            if self.firebase_initialized and self.db and hasattr(self, 'session_id'):
                logout_data = {
                    'guard_id': guard_id,
                    'session_id': self.session_id,
                    'logout_time': self.get_current_timestamp(),
                    'logout_type': 'manual_logout',
                    'status': 'logged_out'
                }
                
                # Update the existing login record with logout info
                login_ref = self.db.collection('guard_logins').document(self.session_id)
                login_ref.update(logout_data)
                
                print(f"SUCCESS: Guard logout saved to Firebase: {guard_id}")
                
                # Add to activity log
                self.add_activity_log(f"Guard Logout: {guard_id} - {logout_data['logout_time']}")
                
            else:
                print("WARNING: Firebase not available - guard logout not saved to cloud")
                # Still add to activity log for local tracking
                self.add_activity_log(f"Guard Logout: {guard_id} - {self.get_current_timestamp()}")
                
        except Exception as e:
            print(f"ERROR: Failed to save guard logout to Firebase: {e}")
            # Still add to activity log for local tracking
            self.add_activity_log(f"Guard Logout: {guard_id} - {self.get_current_timestamp()}")
    
    def save_teacher_activity_to_firebase(self, teacher_id, activity_type, timestamp, teacher_info=None):
        """Save teacher activity to Firebase - Updates teachers collection and saves to teacher_activities"""
        try:
            if self.firebase_initialized and self.db:
                # Get teacher info from Firebase if not provided
                if not teacher_info:
                    teacher_ref = self.db.collection('teachers').document(teacher_id)
                    teacher_doc = teacher_ref.get()
                    if teacher_doc.exists:
                        teacher_info = teacher_doc.to_dict()
                    else:
                        print(f"WARNING: Teacher {teacher_id} not found in Firebase")
                        teacher_info = {
                            'teacher_id': teacher_id,
                            'name': f'Teacher {teacher_id}',
                            'department': 'Unknown',
                            'position': 'Unknown'
                        }
                
                # Save to teacher_activities collection for historical log
                activity_data = {
                    'teacher_id': teacher_id,
                    'name': teacher_info.get('name', f'Teacher {teacher_id}'),
                    'activity_type': activity_type,  # 'time_in' or 'time_out'
                    'timestamp': timestamp,
                    'department': teacher_info.get('department', 'Unknown')
                }
                
                # Save to teacher_activities collection (using teacher_id_timestamp format)
                formatted_timestamp = self.format_document_id(timestamp)
                doc_id = f"{teacher_id}_{formatted_timestamp}"
                activity_ref = self.db.collection('teacher_activities').document(doc_id)
                activity_ref.set(activity_data)
                
                # Update/maintain teacher document in teachers collection
                teacher_ref = self.db.collection('teachers').document(teacher_id)
                teacher_doc = teacher_ref.get()
                
                teacher_update_data = {
                    'teacher_id': teacher_id,
                    'name': teacher_info.get('name', f'Teacher {teacher_id}'),
                    'department': teacher_info.get('department', 'Unknown'),
                    'position': teacher_info.get('position', 'Unknown'),
                    'last_updated': timestamp
                }
                
                # Update time_in or time_out based on activity type
                if activity_type == 'time_in':
                    teacher_update_data['time_in'] = timestamp
                    teacher_update_data['status'] = 'active'
                elif activity_type == 'time_out':
                    teacher_update_data['time_out'] = timestamp
                    teacher_update_data['status'] = 'exited'
                
                # Set created_at only if document doesn't exist
                if not teacher_doc.exists:
                    teacher_update_data['created_at'] = timestamp
                
                # Update or create teacher document
                teacher_ref.set(teacher_update_data, merge=True)
                
                print(f"SUCCESS: Teacher activity saved to Firebase: {activity_type} for {teacher_info.get('name', teacher_id)}")
            else:
                print("WARNING: Firebase not available - teacher activity not saved to cloud")
                
        except Exception as e:
            print(f"ERROR: Failed to save teacher activity to Firebase: {e}")
    
    def save_student_violation_to_firebase(self, student_id, violation_type, timestamp, student_info=None, violation_details=None, is_final=True, rfid=None, missing_items=None):
        """Save student violation to Firebase - Updates student_violations collection
        Args:
            student_id: Student ID
            violation_type: 'uniform_violation' or 'forgot_id'
            timestamp: Violation timestamp
            student_info: Student information dict
            violation_details: Detailed description of violation
            is_final: If True, increment violation_count. If False, save to history only (pending)
            rfid: RFID for session tracking (optional)
            missing_items: List of missing uniform items/parts (e.g., ["shirt", "pants", "shoes"])
        
        NOTE: This function does NOT clear or modify approved_violations flags.
        Approved violations persist for 24 hours regardless of new violations.
        """
        try:
            # CRITICAL: Check if student has active approved violation - but don't block saving
            # The approved violation flag should remain active even when new violations are saved
            if student_id:
                try:
                    has_approved = self.check_approved_violation_active(student_id, rfid)
                    if has_approved:
                        print(f"ℹ️ INFO: Student {student_id} has active approved violation - violation will be saved but detection skip remains active")
                except Exception as e:
                    print(f"⚠️ Warning: Could not check approved violation status: {e}")
            
            if self.firebase_initialized and self.db:
                # Get student info from Firebase if not provided
                if not student_info:
                    student_ref = self.db.collection('students').document(student_id)
                    student_doc = student_ref.get()
                    if student_doc.exists:
                        student_info = student_doc.to_dict()
                    else:
                        print(f"WARNING: Student {student_id} not found in Firebase")
                        student_info = {
                            'student_id': student_id,
                            'name': f'Student {student_id}',
                            'course': 'Unknown'
                        }
                
                # Always save to violation_history (for tracking all attempts)
                violation_ref = self.db.collection('student_violations').document(student_id)
                violation_doc = violation_ref.get()
                
                # Add violation details to history (always save, even if pending)
                if violation_details:
                    # Use student_id_timestamp format for violation_history document ID
                    formatted_timestamp = self.format_document_id(timestamp)
                    doc_id = f"{student_id}_{formatted_timestamp}"
                    details_ref = violation_ref.collection('violation_history').document(doc_id)
                    history_entry = {
                        'violation_type': violation_type,
                        'timestamp': timestamp,
                        'details': violation_details,
                        'student_id': student_id,
                        'status': 'pending' if not is_final else 'finalized',
                        'rfid': rfid or 'unknown'
                    }
                    # Always add missing_items field (even if empty) for uniform violations
                    # This ensures the field exists and can be updated later during finalization
                    if missing_items and isinstance(missing_items, list) and len(missing_items) > 0:
                        history_entry['missing_items'] = missing_items
                        print(f"✅ Storing missing items in violation history: {missing_items}")
                        print(f"   - Document ID: {doc_id}")
                        print(f"   - Violation type: {violation_type}")
                    elif violation_type == 'uniform_violation':
                        # For uniform violations, always include missing_items field (even if empty)
                        # It will be updated during finalization with correct checkbox state
                        history_entry['missing_items'] = []
                        print(f"⚠️ No missing_items provided for uniform violation - setting to empty list")
                        print(f"   - Will be updated during finalization with checkbox state")
                        print(f"   - Document ID: {doc_id}")
                    else:
                        print(f"⚠️ WARNING: No missing_items provided or empty list for violation history")
                        print(f"   - missing_items value: {missing_items}")
                        print(f"   - Document ID: {doc_id}")
                    # Use set with merge=True to ensure fields can be updated later
                    details_ref.set(history_entry, merge=True)
                    print(f"✅ Saved violation history document: {doc_id}")
                
                # If pending (not final), store in session tracking but don't increment count yet
                if not is_final:
                    if rfid:
                        if rfid not in self.active_session_violations:
                            self.active_session_violations[rfid] = []
                        violation_entry = {
                            'student_id': student_id,
                            'violation_type': violation_type,
                            'timestamp': timestamp,
                            'details': violation_details or f'{violation_type} violation',
                            'student_info': student_info
                        }
                        # Add missing_items to session tracking
                        if missing_items and isinstance(missing_items, list) and len(missing_items) > 0:
                            violation_entry['missing_items'] = missing_items
                        self.active_session_violations[rfid].append(violation_entry)
                        print(f"INFO: Violation saved as PENDING (session tracking) - will finalize when student exits/gives up")
                    else:
                        print(f"WARNING: RFID not provided for pending violation - cannot track in session")
                    return  # Don't update count or main document yet
                
                # If final, update violation count and main document
                violation_count = 1
                if violation_doc.exists:
                    # Update existing violation document
                    existing_data = violation_doc.to_dict()
                    violation_count = existing_data.get('violation_count', 0) + 1
                    
                    violation_update_data = {
                        'student_id': student_id,
                        'name': student_info.get('name', f'Student {student_id}'),
                        'course': student_info.get('course', 'Unknown'),
                        'violation_count': violation_count,
                        'last_violation_type': violation_type,  # 'uniform_violation' or 'forgot_id'
                        'last_violation_timestamp': timestamp,
                        'status': 'active',  # Set status to active when new violation occurs
                        'last_updated': timestamp
                    }
                    # Add missing_items if provided (for uniform violations)
                    if missing_items and isinstance(missing_items, list) and len(missing_items) > 0:
                        violation_update_data['last_missing_items'] = missing_items
                        print(f"✅ Storing missing items in violation document (is_final=True): {missing_items}")
                        print(f"   - Document will be updated with merge=True")
                    else:
                        print(f"⚠️ WARNING: No missing_items provided for is_final=True violation")
                        print(f"   - missing_items value: {missing_items}")
                    
                    # Preserve created_at from existing document
                    if 'created_at' in existing_data:
                        violation_update_data['created_at'] = existing_data['created_at']
                    
                    print(f"🔍 DEBUG: violation_update_data keys before save (is_final=True): {list(violation_update_data.keys())}")
                    print(f"🔍 DEBUG: violation_update_data has last_missing_items: {'last_missing_items' in violation_update_data}")
                    
                    violation_ref.set(violation_update_data, merge=True)
                    
                    # Verify the field was saved
                    try:
                        verify_doc = violation_ref.get()
                        if verify_doc.exists:
                            verify_data = verify_doc.to_dict()
                            if 'last_missing_items' in verify_data:
                                print(f"✅ VERIFIED: last_missing_items saved successfully (is_final=True): {verify_data.get('last_missing_items')}")
                            else:
                                print(f"⚠️ WARNING: last_missing_items NOT found in saved document after merge (is_final=True)!")
                    except Exception as e:
                        print(f"⚠️ Warning: Could not verify saved document: {e}")
                else:
                    # Create new violation document
                    violation_data = {
                        'student_id': student_id,
                        'name': student_info.get('name', f'Student {student_id}'),
                        'course': student_info.get('course', 'Unknown'),
                        'violation_count': 1,
                        'last_violation_type': violation_type,  # 'uniform_violation' or 'forgot_id'
                        'last_violation_timestamp': timestamp,
                        'status': 'active',
                        'created_at': timestamp,
                        'last_updated': timestamp
                    }
                    # Add missing_items if provided (for uniform violations)
                    if missing_items and isinstance(missing_items, list) and len(missing_items) > 0:
                        violation_data['last_missing_items'] = missing_items
                        print(f"✅ Storing missing items in new violation document (is_final=True): {missing_items}")
                    else:
                        print(f"⚠️ WARNING: No missing_items provided for new violation document (is_final=True)")
                        print(f"   - missing_items value: {missing_items}")
                    
                    print(f"🔍 DEBUG: violation_data keys before save (new document, is_final=True): {list(violation_data.keys())}")
                    print(f"🔍 DEBUG: violation_data has last_missing_items: {'last_missing_items' in violation_data}")
                    
                    violation_ref.set(violation_data)
                    
                    # Verify the field was saved
                    try:
                        verify_doc = violation_ref.get()
                        if verify_doc.exists:
                            verify_data = verify_doc.to_dict()
                            if 'last_missing_items' in verify_data:
                                print(f"✅ VERIFIED: last_missing_items saved successfully in new document (is_final=True): {verify_data.get('last_missing_items')}")
                            else:
                                print(f"⚠️ WARNING: last_missing_items NOT found in new saved document (is_final=True)!")
                    except Exception as e:
                        print(f"⚠️ Warning: Could not verify saved document: {e}")
                
                print(f"SUCCESS: Student violation saved to Firebase: {violation_type} for {student_info.get('name', student_id)}")
                print(f"INFO: Violation document location: student_violations/{student_id}")
                print(f"INFO: Violation data - Type: {violation_type}, Count: {violation_count}, Timestamp: {timestamp}")
            else:
                print("WARNING: Firebase not available - student violation not saved to cloud")
                
        except Exception as e:
            print(f"ERROR: Failed to save student violation to Firebase: {e}")
    
    def finalize_session_violations(self, rfid):
        """Finalize all pending violations for a session when student exits or gives up.
        Increments violation_count only once per session, regardless of how many violations occurred.
        """
        try:
            if rfid and rfid in self.active_session_violations:
                pending_violations = self.active_session_violations[rfid]
                if pending_violations:
                    print(f"INFO: Finalizing {len(pending_violations)} pending violations for session (RFID: {rfid})")
                    
                    # Get student info from first violation (all should be same student)
                    first_violation = pending_violations[0]
                    student_id = first_violation.get('student_id')
                    student_info = first_violation.get('student_info')
                    
                    if student_id:
                        # Only increment count once per session (not once per violation)
                        # Use the most recent violation for the main document update
                        latest_violation = pending_violations[-1]  # Most recent
                        violation_type = latest_violation.get('violation_type')
                        timestamp = latest_violation.get('timestamp')
                        
                        # CRITICAL: Get missing_items from current checkbox state (most accurate - guard's current review)
                        # Only fall back to stored violation data if checkboxes aren't available
                        missing_items = []
                        
                        # First, try to get from current requirement checkboxes (guard's manual review)
                        if hasattr(self, 'requirement_checkboxes') and self.requirement_checkboxes:
                            checked_count = 0
                            for req_name, checkbox_data in self.requirement_checkboxes.items():
                                if isinstance(checkbox_data, dict) and 'var' in checkbox_data:
                                    var = checkbox_data['var']
                                    if isinstance(var, tk.BooleanVar):
                                        if var.get():  # Checked = present
                                            checked_count += 1
                            
                            # Only use checkboxes if guard has manually reviewed (checked at least one)
                            if checked_count > 0:
                                for req_name, checkbox_data in self.requirement_checkboxes.items():
                                    if isinstance(checkbox_data, dict) and 'var' in checkbox_data:
                                        var = checkbox_data['var']
                                        if isinstance(var, tk.BooleanVar):
                                            if not var.get():  # Unchecked = missing
                                                part_name = checkbox_data.get('part', req_name)
                                                missing_items.append(part_name)
                                if missing_items:
                                    print(f"✅ Using current checkbox state for missing_items: {missing_items}")
                        
                        # Fallback: If no checkbox data, use stored violation data
                        if not missing_items or len(missing_items) == 0:
                            missing_items = latest_violation.get('missing_items', [])
                            print(f"🔍 DEBUG: Using stored missing_items from latest violation: {missing_items}")
                            
                            # If still empty, try to get from any violation
                            if not missing_items or len(missing_items) == 0:
                                print(f"🔍 DEBUG: No missing_items in latest violation, searching all pending violations...")
                                for i, violation in enumerate(pending_violations):
                                    violation_missing = violation.get('missing_items')
                                    print(f"🔍 DEBUG: Violation {i} missing_items: {violation_missing}")
                                    if violation_missing:
                                        missing_items = violation_missing
                                        print(f"✅ Found missing_items in violation {i}: {missing_items}")
                                        break
                        
                        print(f"🔍 DEBUG: Final missing_items to save: {missing_items}")
                        print(f"🔍 DEBUG: missing_items type: {type(missing_items)}, length: {len(missing_items) if isinstance(missing_items, list) else 'N/A'}")
                        
                        # Update violation count only once for this session
                        violation_ref = self.db.collection('student_violations').document(student_id)
                        violation_doc = violation_ref.get()
                        
                        violation_count = 1
                        if violation_doc.exists:
                            existing_data = violation_doc.to_dict()
                            violation_count = existing_data.get('violation_count', 0) + 1  # Increment once
                        else:
                            # Create new document
                            violation_data = {
                                'student_id': student_id,
                                'name': student_info.get('name', f'Student {student_id}') if student_info else f'Student {student_id}',
                                'course': student_info.get('course', 'Unknown') if student_info else 'Unknown',
                                'violation_count': 1,
                                'last_violation_type': violation_type,
                                'last_violation_timestamp': timestamp,
                                'status': 'active',
                                'created_at': timestamp,
                                'last_updated': timestamp
                            }
                            # Add missing_items if available
                            if missing_items and isinstance(missing_items, list) and len(missing_items) > 0:
                                violation_data['last_missing_items'] = missing_items
                                print(f"✅ Storing missing items in new violation document (finalized): {missing_items}")
                            else:
                                print(f"⚠️ WARNING: No missing_items to store in new violation document")
                                print(f"   - missing_items value: {missing_items}")
                                print(f"   - missing_items type: {type(missing_items)}")
                            violation_ref.set(violation_data)
                            
                            # Verify the field was saved
                            try:
                                verify_doc = violation_ref.get()
                                if verify_doc.exists:
                                    verify_data = verify_doc.to_dict()
                                    if 'last_missing_items' in verify_data:
                                        print(f"✅ VERIFIED: last_missing_items saved successfully: {verify_data.get('last_missing_items')}")
                                    else:
                                        print(f"⚠️ WARNING: last_missing_items NOT found in saved document!")
                            except Exception as e:
                                print(f"⚠️ Warning: Could not verify saved document: {e}")
                            print(f"SUCCESS: Created violation document and incremented count: {violation_count}")
                            # Clear pending violations for this session
                            del self.active_session_violations[rfid]
                            if rfid in self.active_detection_sessions:
                                del self.active_detection_sessions[rfid]
                            return
                        
                        # Update existing document
                        violation_update_data = {
                            'student_id': student_id,
                            'name': student_info.get('name', f'Student {student_id}') if student_info else f'Student {student_id}',
                            'course': student_info.get('course', 'Unknown') if student_info else 'Unknown',
                            'violation_count': violation_count,  # Incremented once
                            'last_violation_type': violation_type,
                            'last_violation_timestamp': timestamp,
                            'status': 'active',
                            'last_updated': timestamp
                        }
                        # Add missing_items if available
                        if missing_items and isinstance(missing_items, list) and len(missing_items) > 0:
                            violation_update_data['last_missing_items'] = missing_items
                            print(f"✅ Storing missing items in violation document (finalized): {missing_items}")
                            print(f"   - Document will be updated with merge=True")
                        else:
                            print(f"⚠️ WARNING: No missing_items to store in violation document update")
                            print(f"   - missing_items value: {missing_items}")
                            print(f"   - missing_items type: {type(missing_items)}")
                        
                        if violation_doc.exists:
                            existing_data = violation_doc.to_dict()
                            if 'created_at' in existing_data:
                                violation_update_data['created_at'] = existing_data['created_at']
                            print(f"🔍 DEBUG: Existing document has last_missing_items: {'last_missing_items' in existing_data}")
                            if 'last_missing_items' in existing_data:
                                print(f"🔍 DEBUG: Existing last_missing_items: {existing_data.get('last_missing_items')}")
                        
                        print(f"🔍 DEBUG: violation_update_data keys before save: {list(violation_update_data.keys())}")
                        print(f"🔍 DEBUG: violation_update_data has last_missing_items: {'last_missing_items' in violation_update_data}")
                        
                        violation_ref.set(violation_update_data, merge=True)
                        
                        # Verify the field was saved
                        try:
                            verify_doc = violation_ref.get()
                            if verify_doc.exists:
                                verify_data = verify_doc.to_dict()
                                if 'last_missing_items' in verify_data:
                                    print(f"✅ VERIFIED: last_missing_items saved successfully: {verify_data.get('last_missing_items')}")
                                else:
                                    print(f"⚠️ WARNING: last_missing_items NOT found in saved document after merge!")
                                    print(f"   - Document keys: {list(verify_data.keys())}")
                        except Exception as e:
                            print(f"⚠️ Warning: Could not verify saved document: {e}")
                        
                        # Update all violation history entries to "finalized" status and update missing_items
                        for violation in pending_violations:
                            # Update history entry status to finalized and missing_items (use current checkbox state)
                            violation_timestamp = violation.get('timestamp')
                            
                            # Use document ID directly (more reliable than timestamp query)
                            # Document ID format: {student_id}_{formatted_timestamp}
                            formatted_timestamp = self.format_document_id(violation_timestamp)
                            doc_id = f"{student_id}_{formatted_timestamp}"
                            history_ref = violation_ref.collection('violation_history')
                            history_doc_ref = history_ref.document(doc_id)
                            
                            # Always update status to finalized and missing_items
                            update_data = {'status': 'finalized'}
                            
                            # CRITICAL: Always update missing_items with the accurate value from checkbox state
                            # Even if it was empty initially, we should update it now with the correct value
                            if missing_items and isinstance(missing_items, list) and len(missing_items) > 0:
                                update_data['missing_items'] = missing_items
                                print(f"✅ Updating violation history missing_items: {missing_items}")
                                print(f"   - Document ID: {doc_id}")
                                print(f"   - Timestamp: {violation_timestamp}")
                            else:
                                # Even if empty, set it to empty list to ensure field exists
                                update_data['missing_items'] = []
                                print(f"⚠️ No missing_items to update - setting to empty list")
                                print(f"   - Document ID: {doc_id}")
                                print(f"   - Timestamp: {violation_timestamp}")
                            
                            try:
                                # Check if document exists first
                                existing_doc = history_doc_ref.get()
                                if not existing_doc.exists:
                                    print(f"⚠️ WARNING: Violation history document {doc_id} does not exist!")
                                    print(f"   - Creating it now with missing_items: {missing_items}")
                                    # Create the document with all necessary fields
                                    create_data = {
                                        'violation_type': violation.get('violation_type', 'uniform_violation'),
                                        'timestamp': violation_timestamp,
                                        'details': violation.get('details', 'Uniform violation'),
                                        'student_id': student_id,
                                        'status': 'finalized',
                                        'rfid': violation.get('rfid', 'unknown'),
                                        'missing_items': missing_items if missing_items else []
                                    }
                                    history_doc_ref.set(create_data)
                                    print(f"✅ Created violation history document: {doc_id}")
                                else:
                                    # Document exists, update it with merge=True
                                    history_doc_ref.set(update_data, merge=True)
                                    print(f"✅ Updated violation history document: {doc_id}")
                                    print(f"   - Update data: {update_data}")
                                
                                # Verify the update worked
                                verify_doc = history_doc_ref.get()
                                if verify_doc.exists:
                                    verify_data = verify_doc.to_dict()
                                    if 'missing_items' in verify_data:
                                        print(f"✅ VERIFIED: missing_items in violation history: {verify_data.get('missing_items')}")
                                    else:
                                        print(f"⚠️ WARNING: missing_items NOT found after update!")
                                        print(f"   - Document keys: {list(verify_data.keys())}")
                                else:
                                    print(f"⚠️ WARNING: Document does not exist after update!")
                            except Exception as e:
                                print(f"⚠️ Warning: Could not update violation history document {doc_id}: {e}")
                                import traceback
                                traceback.print_exc()
                                # Fallback: Try querying by timestamp
                                try:
                                    history_docs = history_ref.where('timestamp', '==', violation_timestamp).limit(1).get()
                                    if history_docs:
                                        for history_doc in history_docs:
                                            history_doc.reference.set(update_data, merge=True)
                                            print(f"✅ Updated violation history document (fallback): {history_doc.id}")
                                    else:
                                        print(f"⚠️ WARNING: No violation history document found with timestamp: {violation_timestamp}")
                                except Exception as e2:
                                    print(f"❌ ERROR: Could not update violation history document: {e2}")
                                    import traceback
                                    traceback.print_exc()
                        
                        print(f"SUCCESS: Session violations finalized - count incremented to {violation_count} for {len(pending_violations)} violations")
                    
                    # Clear pending violations for this session
                    del self.active_session_violations[rfid]
                    
                # Also clear session tracking
                if rfid in self.active_detection_sessions:
                    del self.active_detection_sessions[rfid]
                    
        except Exception as e:
            print(f"ERROR: Failed to finalize session violations: {e}")
            import traceback
            traceback.print_exc()
    
    def save_student_rfid_assignment_to_firebase(self, assignment_info):
        """Save student RFID assignment to Firebase"""
        try:
            if self.firebase_initialized and self.db:
                # Save to student_rfid_assignments collection
                assignment_ref = self.db.collection('student_rfid_assignments').document()
                assignment_data = {
                    'student_id': assignment_info['student_id'],
                    'name': assignment_info['name'],
                    'course': assignment_info['course'],
                    'gender': assignment_info['gender'],
                    'rfid': assignment_info['rfid'],
                    'assignment_time': assignment_info['assignment_time'],
                    'status': assignment_info['status'],
                    'assignment_type': 'forgot_id'
                }
                # Add expiry_time if it exists in assignment_info
                if 'expiry_time' in assignment_info:
                    assignment_data['expiry_time'] = assignment_info['expiry_time']
                assignment_ref.set(assignment_data)
                
                print(f"SUCCESS: Student RFID assignment saved to Firebase: {assignment_info['name']} - {assignment_info['rfid']}")
            else:
                print("WARNING: Firebase not available - student RFID assignment not saved to cloud")
                
        except Exception as e:
            print(f"ERROR: Failed to save student RFID assignment to Firebase: {e}")
    
    def save_permanent_students_to_firebase(self):
        """Save permanent student records to Firebase students collection"""
        try:
            if self.firebase_initialized and self.db:
                # Permanent student data
                students_data = [
                    {
                        'student_id': '02000289900',
                        'rfid': '0095365253',
                        'name': 'John Jason Domingo',
                        'course': 'ICT',
                        'gender': 'Male',
                        'status': 'active',
                        'student_type': 'permanent',
                        'created_at': self.get_current_timestamp(),
                        'last_login': None,
                        'total_logins': 0
                    }
                ]
                
                # Save each student to Firebase
                for student_data in students_data:
                    student_ref = self.db.collection('students').document(student_data['student_id'])
                    student_ref.set(student_data)
                    print(f"SUCCESS: Student {student_data['name']} saved to Firebase students collection")
                
                print("SUCCESS: All permanent students saved to Firebase students collection")
                
            else:
                print("WARNING: Firebase not available - permanent students not saved to cloud")
                
        except Exception as e:
            print(f"ERROR: Failed to save permanent students to Firebase: {e}")
    
    def save_guards_to_firebase(self):
        """Save guard RFID numbers to Firebase guards collection"""
        try:
            if self.firebase_initialized and self.db:
                # Guard data to save
                guards_data = [
                    {
                        'guard_id': '0095081841',
                        'rfid': '0095081841',
                        'name': 'Guard 1',
                        'role': 'Security Guard',
                        'status': 'active',
                        'created_at': self.get_current_timestamp(),
                        'last_login': None,
                        'total_logins': 0
                    },
                    {
                        'guard_id': '0095339862',
                        'rfid': '0095339862',
                        'name': 'Guard 2',
                        'role': 'Security Guard',
                        'status': 'active',
                        'created_at': self.get_current_timestamp(),
                        'last_login': None,
                        'total_logins': 0
                    }
                ]
                
                # Save each guard to Firebase
                for guard_data in guards_data:
                    guard_ref = self.db.collection('guards').document(guard_data['guard_id'])
                    guard_doc = guard_ref.get()
                    
                    if guard_doc.exists:
                        # Document exists - preserve existing name and other fields
                        existing_data = guard_doc.to_dict()
                        # Only update fields that should be updated, preserve name and login info
                        update_data = {
                            'guard_id': guard_data['guard_id'],
                            'rfid': guard_data['rfid'],
                            'role': guard_data['role'],
                            'status': guard_data['status']
                        }
                        # Preserve existing name if it exists, otherwise use default
                        if 'name' in existing_data and existing_data['name']:
                            update_data['name'] = existing_data['name']
                        else:
                            update_data['name'] = guard_data['name']
                        # Preserve existing login info
                        if 'last_login' in existing_data:
                            update_data['last_login'] = existing_data['last_login']
                        if 'total_logins' in existing_data:
                            update_data['total_logins'] = existing_data['total_logins']
                        if 'created_at' in existing_data:
                            update_data['created_at'] = existing_data['created_at']
                        else:
                            update_data['created_at'] = guard_data['created_at']
                        
                        guard_ref.update(update_data)
                        print(f"SUCCESS: Guard {guard_data['guard_id']} updated in Firebase (name preserved)")
                    else:
                        # Document doesn't exist - create new with default data
                        guard_ref.set(guard_data)
                        print(f"SUCCESS: Guard {guard_data['guard_id']} created in Firebase guards collection")
                
                print("SUCCESS: All guards saved to Firebase guards collection")
                
            else:
                print("WARNING: Firebase not available - guards not saved to cloud")
                
        except Exception as e:
            print(f"ERROR: Failed to save guards to Firebase: {e}")
    
    def update_guard_login_info(self, guard_id):
        """Update guard's last login and total logins count in Firebase"""
        try:
            if self.firebase_initialized and self.db:
                guard_ref = self.db.collection('guards').document(guard_id)
                
                # Get current guard data
                guard_doc = guard_ref.get()
                if guard_doc.exists:
                    guard_data = guard_doc.to_dict()
                    current_logins = guard_data.get('total_logins', 0)
                    
                    # Update guard login information
                    guard_ref.update({
                        'last_login': self.get_current_timestamp(),
                        'total_logins': current_logins + 1,
                        'status': 'active'
                    })
                    
                    print(f"SUCCESS: Guard {guard_id} login info updated in Firebase")
                else:
                    print(f"WARNING: Guard {guard_id} not found in Firebase guards collection")
                    
            else:
                print("WARNING: Firebase not available - guard login info not updated")
                
        except Exception as e:
            print(f"ERROR: Failed to update guard login info: {e}")
    
    def approve_gate(self):
        """Approve and open the gate"""
        try:
            # Add to activity log
            self.add_activity_log("GATE APPROVED - Gate opened by guard")
            
            # Show confirmation
            messagebox.showinfo("Gate Approved", 
                              "SUCCESS: Gate has been OPENED\n\n"
                              "Access granted - Gate is now open for entry/exit.\n\n"
                              "Remember to close the gate when appropriate.")
            
            # Update button states to show gate is open (if buttons exist)
            if hasattr(self, 'approve_button'):
                self.approve_button.config(
                    text="SUCCESS: OPEN",
                    bg='#059669',
                    state='disabled'
                )
            if hasattr(self, 'deny_button'):
                self.deny_button.config(
                    text="ACCESS DENIED",
                    bg='#dc2626',
                    state='normal'
                )
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open gate: {e}")
    
    def deny_gate(self):
        """Deny and close the gate"""
        try:
            # Add to activity log
            self.add_activity_log("GATE DENIED - Gate closed by guard")
            
            # Show confirmation
            messagebox.showinfo("Gate Denied", 
                              "ERROR: Gate has been CLOSED\n\n"
                              "Access denied - Gate is now closed.\n\n"
                              "No entry/exit allowed until approved.")
            
            # Update button states to show gate is closed (if buttons exist)
            if hasattr(self, 'deny_button'):
                self.deny_button.config(
                    text="ERROR: CLOSED",
                    bg='#b91c1c',
                    state='disabled'
                )
            if hasattr(self, 'approve_button'):
                self.approve_button.config(
                    text="ACCESS GRANTED",
                bg='#10b981',
                state='normal'
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to close gate: {e}")
    
    def update_button_states(self, selected_type):
        """Update button visual states"""
        # No person type buttons to update since they were removed
        pass
    
    def create_camera_feed_section(self, parent):
        """Create camera feed section - simplified, no right panel"""
        camera_frame = tk.LabelFrame(
            parent,
            text="LIVE CAMERA FEED",
            font=('Arial', 14, 'bold'),
            fg='#374151',
            bg='#ffffff',
            relief='groove',
            bd=2
        )
        camera_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Camera feed label with box/portrait orientation - STANDBY MODE
        self.camera_label = tk.Label(
            camera_frame,
            text="📷 CAMERA FEED (STANDBY MODE)\n\n🔒 Camera is CLOSED\n\nCamera will ONLY open when:\n• Student taps their ID\n• Detection process starts\n\nNo camera access during guard login\n\n💡 Camera preview will show here\nwhen detection is active",
            font=('Arial', 12),
            fg='#374151',
            bg='#dbeafe',
            justify=tk.CENTER,
            relief='sunken',
            bd=3
        )
        self.camera_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        
        # Set minimum height for portrait orientation
        camera_frame.configure(height=520)

    def update_detected_classes_list(self, detected_classes):
        """Update detected classes - now adds to activity log instead of separate listbox.
        Accepts either a list of dicts with keys 'class_name' and 'confidence',
        or a list of strings.
        """
        try:
            # Normalize names in the worker thread first
            names = []
            for item in (detected_classes or []):
                if isinstance(item, dict):
                    name = item.get('class_name')
                    conf = item.get('confidence')
                    if name:
                        if isinstance(conf, (int, float)):
                            names.append(f"{name} ({conf:.2f})")
                        else:
                            names.append(name)
                else:
                    names.append(str(item))

            # Save latest non-empty detection set with timestamp
            try:
                if names:
                    self._last_detected_classes = names
                    self._last_detect_time = time.time()
            except Exception:
                pass

            # Marshal UI update to main thread
            def _apply(names_list):
                try:
                    # If empty, use last known detections (do not clear immediately)
                    effective = names_list if names_list else getattr(self, '_last_detected_classes', [])
                    
                    # Add detected classes to activity log if there are new detections
                    if effective and effective != getattr(self, '_last_logged_classes', []):
                        detected_str = ", ".join(effective)
                        self.add_activity_log(f"Detected: {detected_str}")
                        self._last_logged_classes = effective.copy()

                    # Update requirements panel against current effective detections
                    try:
                        self._update_requirements_status([n.lower() for n in effective])
                    except Exception:
                        pass
                except Exception:
                    pass

            try:
                self.root.after(0, _apply, names)
            except Exception:
                _apply(names)
        except Exception:
            pass

    def set_uniform_requirements_by_course(self, course, gender=None):
        """Set uniform requirements based on course and gender."""
        try:
            course_upper = (course or "").upper()
            gender_lower = (gender or "").lower()
            
            # Determine which preset to use
            if "SHS" in course_upper or "SENIOR HIGH" in course_upper:
                self.current_uniform_key = 'SHS'
            elif "BSHM" in course_upper or "HOSPITALITY" in course_upper or "HOSPITALITY MANAGEMENT" in course_upper:
                # BSHM has different requirements for male and female
                if gender_lower.startswith('f'):
                    self.current_uniform_key = 'BSHM_FEMALE'
                else:
                    self.current_uniform_key = 'BSHM_MALE'
            elif "BSCPE" in course_upper:
                # BSCPE has different requirements for male and female
                if gender_lower.startswith('f'):
                    # BSCPE_FEMALE not defined yet, use BSCPE_MALE for now
                    self.current_uniform_key = 'BSCPE_MALE'
                else:
                    self.current_uniform_key = 'BSCPE_MALE'
            elif "ARTS" in course_upper and "SCIENCE" in course_upper:
                # Arts and Science has its own requirements
                self.current_uniform_key = 'ARTS_AND_SCIENCE'
            elif "TOURISM" in course_upper:
                self.current_uniform_key = 'TOURISM'
            else:
                # Default to SHS if unknown
                self.current_uniform_key = 'SHS'
            
            # Get requirements from preset
            self.current_uniform_requirements = list(self.uniform_requirements_presets.get(self.current_uniform_key, []))
            
            # Update requirements frame title
            self.update_requirements_frame_title(course)
            
            # Re-render the requirements list
            self._render_uniform_requirements()
            
            print(f"✅ Uniform requirements set to: {self.current_uniform_key} ({len(self.current_uniform_requirements)} items)")
        except Exception as e:
            print(f"⚠️ Error setting uniform requirements: {e}")
    
    def update_requirements_frame_title(self, course):
        """Update the requirements frame title based on course."""
        try:
            if hasattr(self, 'requirements_frame') and self.requirements_frame:
                course_upper = (course or "").upper()
                if "SHS" in course_upper or "SENIOR HIGH" in course_upper:
                    title = "UNIFORM REQUIREMENTS (SHS)"
                elif "BSHM" in course_upper or "HOSPITALITY" in course_upper:
                    title = "UNIFORM REQUIREMENTS (BSHM)"
                elif "BSCPE" in course_upper:
                    title = "UNIFORM REQUIREMENTS (BSCPE)"
                elif "ARTS" in course_upper and "SCIENCE" in course_upper:
                    title = "UNIFORM REQUIREMENTS (ARTS AND SCIENCE)"
                elif "TOURISM" in course_upper:
                    title = "UNIFORM REQUIREMENTS (TOURISM)"
                else:
                    title = "UNIFORM REQUIREMENTS"
                
                # Update the box frame title (not requirements_frame which is now just a container)
                if hasattr(self, 'requirements_box_frame') and self.requirements_box_frame:
                    self.requirements_box_frame.config(text=title)
        except Exception:
            pass
    
    def update_requirements_student_info(self, student_info=None):
        """Update student name and course in requirements section"""
        try:
            if hasattr(self, 'requirements_student_info_label') and self.requirements_student_info_label:
                if student_info:
                    name = student_info.get('name', 'Unknown')
                    course = student_info.get('course', 'Unknown Course')
                    # Format: "Name - Course/Senior High School"
                    self.requirements_student_info_label.config(text=f"{name} - {course}")
                else:
                    # Clear if no student info
                    self.requirements_student_info_label.config(text="")
        except Exception as e:
            print(f"⚠️ Error updating requirements student info: {e}")
    
    def show_requirements_section(self, student_info=None):
        """Show the uniform requirements section when student ID is tapped."""
        try:
            if hasattr(self, 'requirements_frame') and self.requirements_frame:
                # Update student info if provided
                if student_info:
                    self.update_requirements_student_info(student_info)
                
                # Hide the spacer when showing requirements
                if hasattr(self, 'requirements_spacer') and self.requirements_spacer:
                    try:
                        self.requirements_spacer.pack_forget()
                    except:
                        pass
                
                # Check if already visible, if not then pack it
                try:
                    self.requirements_frame.pack_info()
                    # Already visible, no need to pack again
                except tk.TclError:
                    # Not visible, pack it INSIDE requirements_box_frame
                    # The box is always visible, we just show the contents when student ID is tapped
                    self.requirements_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(10, 15))
        except Exception:
            pass
    
    def hide_requirements_section(self):
        """Hide the uniform requirements section when no student is detected."""
        try:
            if hasattr(self, 'requirements_frame') and self.requirements_frame:
                self.requirements_frame.pack_forget()
                # Reset the scheduled flag when actually hiding
                self.requirements_hide_scheduled = False
            
            # Show spacer again when requirements are hidden (to keep box visible)
            if hasattr(self, 'requirements_spacer') and self.requirements_spacer:
                try:
                    self.requirements_spacer.pack(fill=tk.X)
                except:
                    pass
            
            # Clear student info label when hiding
            if hasattr(self, 'requirements_student_info_label') and self.requirements_student_info_label:
                self.requirements_student_info_label.config(text="")
            
            # CRITICAL: Don't clear main screen if entry was just saved
            # Only clear if we're not in the middle of processing an entry
            # Check if entry was recently saved by checking if detection_active is False and uniform_detection_complete is True
            should_clear = True
            if hasattr(self, 'detection_active') and not self.detection_active:
                if hasattr(self, 'uniform_detection_complete') and self.uniform_detection_complete:
                    # Entry was just processed - don't clear main screen yet
                    should_clear = False
                    print(f"⏸️ Skipping main screen clear - entry was just saved, keep entry status visible")
            
            if should_clear:
                # Also return main screen to standby when requirements section is hidden
                # (This happens after the 5-second delay for showing status)
                if self.main_screen_window and self.main_screen_window.winfo_exists():
                    self.show_standby_message()
                    print(f"✅ Main screen returned to standby")
        except Exception:
            pass

    def _render_uniform_requirements(self):
        """Render the static list of required parts for current uniform as checkboxes."""
        if not hasattr(self, 'requirements_container') or not self.requirements_container:
            return
        try:
            # Clear existing checkboxes
            for widget in self.requirements_container.winfo_children():
                widget.destroy()
            self.requirement_checkboxes = {}
            self.requirement_checkbox_widgets = {}
            
            # Create checkboxes for each requirement
            for part in self.current_uniform_requirements:
                key = part.lower()
                
                # Create BooleanVar for checkbox state (False = unchecked = missing = shows X)
                var = tk.BooleanVar(value=False)  # Initially unchecked (missing)
                self.requirement_checkboxes[key] = {'var': var, 'part': part}
                
                # Create checkbox with label showing "✗ {part}" when unchecked, "{part}" when checked
                # Use lambda with default arguments to capture current values
                def make_update_callback(part_name, var_ref, key_ref):
                    def update_label():
                        checkbox_widget = self.requirement_checkbox_widgets.get(key_ref)
                        if checkbox_widget:
                            if var_ref.get():
                                # Checked = detected = no X mark
                                checkbox_widget.config(text=part_name)
                            else:
                                # Unchecked = missing = X mark
                                checkbox_widget.config(text=f"✗ {part_name}")
                    return update_label
                
                checkbox = tk.Checkbutton(
                    self.requirements_container,
                    text=f"✗ {part}",  # Initially unchecked (missing)
                    variable=var,
                    font=('Arial', 10),
                    bg='#ffffff',
                    fg='#111827',
                    anchor='w',
                    command=make_update_callback(part, var, key),
                    state=tk.DISABLED  # Initially read-only, will be enabled after DENIED is clicked
                )
                checkbox.pack(fill=tk.X, pady=2)
                self.requirement_checkbox_widgets[key] = checkbox
            
            self.requirements_status_label.config(text="IN/COMPLETE UNIFORM", fg='#dc2626')
        except Exception as e:
            print(f"ERROR: Error rendering uniform requirements: {e}")
            import traceback
            traceback.print_exc()
    
    def _make_requirement_checkboxes_editable(self, editable=True):
        """Make requirement checkboxes editable or read-only"""
        try:
            if hasattr(self, 'requirement_checkbox_widgets') and self.requirement_checkbox_widgets:
                for key, checkbox_widget in self.requirement_checkbox_widgets.items():
                    if checkbox_widget:
                        if editable:
                            checkbox_widget.config(state=tk.NORMAL)
                        else:
                            checkbox_widget.config(state=tk.DISABLED)
                if editable:
                    print(f"✅ Requirement checkboxes are now editable")
                else:
                    print(f"✅ Requirement checkboxes are now read-only")
        except Exception as e:
            print(f"⚠️ Error making checkboxes editable: {e}")

    def _update_requirements_status(self, detected_names_lower):
        """Compare detected classes to requirements and update UI labels.
        Tracks detection counts and permanently checks requirements after 2 detections.
        Permanent checks remain even when detection stops.
        Handles special cases like BSHM female (rtw pants OR rtw skirt).
        """
        try:
            # If uniform detection is complete, preserve permanently checked requirements
            # and don't update based on current detections (to prevent reverting checkboxes)
            if getattr(self, 'uniform_detection_complete', False):
                # Uniform is complete - preserve permanently checked requirements
                if not hasattr(self, '_permanently_checked_requirements'):
                    self._permanently_checked_requirements = set()
                
                # Update checkboxes based on permanently checked requirements only
                if hasattr(self, 'requirement_checkboxes') and self.requirement_checkboxes:
                    required = [p.lower() for p in self.current_uniform_requirements]
                    for part in self.current_uniform_requirements:
                        key = part.lower()
                        if key in self.requirement_checkboxes:
                            var = self.requirement_checkboxes[key]['var']
                            checkbox_widget = self.requirement_checkbox_widgets.get(key)
                            
                            # Check if requirement is permanently checked
                            if key in self._permanently_checked_requirements:
                                var.set(True)
                                if checkbox_widget:
                                    checkbox_widget.config(text=part)
                            else:
                                var.set(False)
                                if checkbox_widget:
                                    checkbox_widget.config(text=f"✗ {part}")
                
                # Keep status as "Complete uniform"
                if hasattr(self, 'requirements_status_label'):
                    self.requirements_status_label.config(text="Complete uniform", fg='#059669')
                
                return  # Exit early - don't process current detections
            
            required = [p.lower() for p in self.current_uniform_requirements]
            present = set()
            
            # Initialize detection counts if not exists
            if not hasattr(self, '_requirement_detection_counts'):
                self._requirement_detection_counts = {}
            if not hasattr(self, '_permanently_checked_requirements'):
                self._permanently_checked_requirements = set()
            
            # Filter out "no detection" message
            filtered_detections = [dn for dn in detected_names_lower if dn and "no detection" not in dn.lower()]
            
            # Special handling for BSHM female: either rtw pants OR rtw skirt is acceptable
            is_bshm_female = (self.current_uniform_key == 'BSHM_FEMALE')
            is_bshm_male = (self.current_uniform_key == 'BSHM_MALE')
            pants_or_skirt_required = is_bshm_female and ('rtw pants' in required or 'rtw skirt' in required)
            
            # Validation: Check for incorrect shoe types for BSHM
            # Male should NOT have closed shoes, Female should NOT have black shoes
            if is_bshm_male:
                # Check if closed shoes is detected (incorrect for male)
                closed_shoes_detected = any('closed shoes' in dn for dn in filtered_detections)
                if closed_shoes_detected:
                    print("⚠️ BSHM Male: Incorrect shoe type detected (closed shoes). Required: black shoes")
            elif is_bshm_female:
                # Check if black shoes is detected (incorrect for female)
                black_shoes_detected = any('black shoes' in dn for dn in filtered_detections)
                if black_shoes_detected:
                    print("⚠️ BSHM Female: Incorrect shoe type detected (black shoes). Required: closed shoes")
            
            # Check current detections and update counts
            # First, determine which requirements are detected in this frame (once per frame)
            detected_this_frame = set()
            for req in required:
                # Special case: For BSHM female, if rtw pants or rtw skirt is detected, mark both as satisfied
                if pants_or_skirt_required and ('rtw pants' in req or 'rtw skirt' in req):
                    # Check if either pants or skirt is detected
                    pants_detected = any('rtw pants' in dn or 'pants' in dn for dn in filtered_detections)
                    skirt_detected = any('rtw skirt' in dn or 'skirt' in dn for dn in filtered_detections)
                    if pants_detected or skirt_detected:
                        # Mark both as detected (even though only one may actually be detected)
                        if 'rtw pants' in required:
                            detected_this_frame.add('rtw pants')
                        if 'rtw skirt' in required:
                            detected_this_frame.add('rtw skirt')
                else:
                    # Normal detection check (substring match)
                    # Check if requirement name appears in any detected class name
                    is_detected = any(req in dn for dn in filtered_detections)
                    if is_detected:
                        detected_this_frame.add(req)
                        # Debug: show what was detected
                        matching_detections = [dn for dn in filtered_detections if req in dn]
                        if matching_detections:
                            print(f"🔍 Requirement '{req}' detected in: {matching_detections}")
            
            # Increment counts for requirements detected in this frame (once per frame)
            for req in detected_this_frame:
                current_count = self._requirement_detection_counts.get(req, 0)
                self._requirement_detection_counts[req] = current_count + 1
                
                # If detected twice, mark as permanently checked
                if self._requirement_detection_counts[req] >= 2:
                    self._permanently_checked_requirements.add(req)
                    print(f"✅ Requirement '{req}' permanently checked (detected {self._requirement_detection_counts[req]} times)")
                    
                # Special case: For BSHM female, if one is permanently checked, mark both
                if pants_or_skirt_required:
                    if req == 'rtw pants' and req in self._permanently_checked_requirements:
                        if 'rtw skirt' in required:
                            self._permanently_checked_requirements.add('rtw skirt')
                    elif req == 'rtw skirt' and req in self._permanently_checked_requirements:
                        if 'rtw pants' in required:
                            self._permanently_checked_requirements.add('rtw pants')
            
            # Include in present set if currently detected OR permanently checked
            # For BSHM female pants/skirt, check if either is detected/permanent
            for req in required:
                is_detected = req in detected_this_frame
                is_permanent = req in self._permanently_checked_requirements
                
                # Special handling for BSHM female pants/skirt
                if pants_or_skirt_required and ('rtw pants' in req or 'rtw skirt' in req):
                    # Check if either pants or skirt is detected/permanent
                    pants_det = 'rtw pants' in detected_this_frame or 'rtw pants' in self._permanently_checked_requirements
                    skirt_det = 'rtw skirt' in detected_this_frame or 'rtw skirt' in self._permanently_checked_requirements
                    if pants_det or skirt_det:
                        present.add(req)
                elif is_detected or is_permanent:
                    present.add(req)

            # Update checkboxes - check detected items (no X), uncheck missing items (X)
            if hasattr(self, 'requirement_checkboxes') and self.requirement_checkboxes:
                for part in self.current_uniform_requirements:
                    key = part.lower()
                    if key in self.requirement_checkboxes:
                        var = self.requirement_checkboxes[key]['var']
                        checkbox_widget = self.requirement_checkbox_widgets.get(key)
                        
                        # Update checkbox state based on detection
                        if key in present:
                            # Detected - check the checkbox (no X mark)
                            var.set(True)
                            if checkbox_widget:
                                checkbox_widget.config(text=part)
                        else:
                            # Not detected - uncheck the checkbox (X mark)
                            var.set(False)
                            if checkbox_widget:
                                checkbox_widget.config(text=f"✗ {part}")

            # Update status label
            if len(present) == len(required) and required:
                self.requirements_status_label.config(text="Complete uniform", fg='#059669')
                # Check for early completion - all requirements detected
                if not self.uniform_detection_complete:
                    self.uniform_detection_complete = True
                    print(f"✅ All uniform requirements detected - timer will stop")
                    # CRITICAL: Mark ALL requirements that have been detected at least once as permanently checked
                    # This ensures all detected requirements stay checked even after detection stops
                    if not hasattr(self, '_permanently_checked_requirements'):
                        self._permanently_checked_requirements = set()
                    # Add all requirements that have been detected (count > 0) to permanently checked
                    for req in required:
                        # Check if requirement was detected at least once
                        detection_count = self._requirement_detection_counts.get(req, 0)
                        if detection_count > 0 or req in present:
                            self._permanently_checked_requirements.add(req)
                            print(f"✅ Requirement '{req}' marked as permanently checked (complete uniform, detected {detection_count} times)")
                    # Handle complete uniform - record entry and stop detection
                    self._handle_complete_uniform_detected()
            else:
                self.requirements_status_label.config(text="Incomplete uniform", fg='#dc2626')
        except Exception:
            pass

    def _monitor_uniform_detection_timer(self):
        """Background thread that monitors the 15-second timer for uniform detection.
        If timer expires without complete uniform, handles incomplete uniform timeout.
        """
        try:
            import time
            while True:
                # Check if timer is still active
                if self.uniform_detection_timer_start is None:
                    break
                
                # Check if uniform already complete
                if self.uniform_detection_complete:
                    print(f"✅ Uniform detection completed early - timer stopped")
                    break
                
                # Check if detection stream has stopped
                if (hasattr(self, 'external_detection_stop_event') and 
                    self.external_detection_stop_event and 
                    self.external_detection_stop_event.is_set()):
                    break
                
                # Calculate elapsed time
                elapsed = time.time() - self.uniform_detection_timer_start
                
                # If 15 seconds elapsed and uniform not complete
                if elapsed >= self.uniform_detection_timer_duration:
                    print(f"⏱️ 15-second timer expired - uniform incomplete")
                    # Handle incomplete uniform timeout
                    if self.current_student_info_for_timer:
                        self._handle_incomplete_uniform_timeout(self.current_student_info_for_timer)
                    # Clear timer variables after timeout handler completes
                    self.current_student_info_for_timer = None
                    self.current_rfid_for_timer = None
                    break
                
                # Sleep briefly before next check
                time.sleep(0.5)
                
        except Exception as e:
            print(f"⚠️ Error in uniform detection timer monitor: {e}")

    def _handle_incomplete_uniform_timeout(self, student_info):
        """Handle incomplete uniform after 15-second timer expires."""
        try:
            print(f"⛔ Handling incomplete uniform timeout for: {student_info.get('name', 'Unknown')}")
            print(f"🔍 DEBUG: student_info keys: {list(student_info.keys()) if student_info else 'None'}")
            print(f"🔍 DEBUG: student_info rfid: {student_info.get('rfid', 'NOT FOUND')}")
            print(f"🔍 DEBUG: student_info id: {student_info.get('id', 'NOT FOUND')}")
            
            # CRITICAL: Get RFID from multiple sources with priority order
            # 1. First check student_info dictionary (should have it after our fixes)
            # 2. Fallback to self.current_rfid_for_timer (backup safety)
            rfid_to_use = None
            if student_info:
                rfid_to_use = student_info.get('rfid')
            
            if not rfid_to_use and hasattr(self, 'current_rfid_for_timer') and self.current_rfid_for_timer:
                rfid_to_use = self.current_rfid_for_timer
                print(f"🔍 DEBUG: Using RFID from current_rfid_for_timer fallback: {rfid_to_use}")
                # Add it to student_info for future use
                student_info['rfid'] = rfid_to_use
            
            if rfid_to_use:
                print(f"🔍 ========== TIMEOUT HANDLER: FETCHING STUDENT NUMBER ==========")
                print(f"🔍 RFID to use for Firebase lookup: {rfid_to_use}")
                print(f"🔍 Current student_info keys: {list(student_info.keys()) if student_info else 'None'}")
                print(f"🔍 Current student_info rfid field: {student_info.get('rfid') if student_info else 'N/A'}")
                print(f"🔍 Ensuring Student Number is present in student_info using RFID: {rfid_to_use}")
                
                # Ensure RFID is in student_info before calling helper function
                if not student_info.get('rfid'):
                    student_info['rfid'] = rfid_to_use
                    print(f"🔍 Added RFID to student_info: {rfid_to_use}")
                
                # Use helper function to ensure Student Number is present and valid
                # This function now guarantees RFID is preserved
                print(f"🔍 Calling _ensure_student_number_from_firebase with RFID: {rfid_to_use}")
                student_info = self._ensure_student_number_from_firebase(student_info, rfid_to_use)
                print(f"🔍 After _ensure_student_number_from_firebase:")
                print(f"   - student_info keys: {list(student_info.keys()) if student_info else 'None'}")
                print(f"   - student_id: {student_info.get('student_id') if student_info else 'N/A'}")
                print(f"   - rfid: {student_info.get('rfid') if student_info else 'N/A'}")
                
                # Final validation: ensure Student Number is not RFID
                student_num = student_info.get('student_id') or student_info.get('student_number')
                if not student_num or str(student_num).strip() == str(rfid_to_use).strip():
                    print(f"⚠️ WARNING: Student Number still equals RFID after validation. Re-fetching from Firebase...")
                    print(f"🔍 Calling get_student_info_by_rfid directly with RFID: {rfid_to_use}")
                    full_student_info = self.get_student_info_by_rfid(rfid_to_use)
                    print(f"🔍 get_student_info_by_rfid returned: {full_student_info is not None}")
                    if full_student_info:
                        print(f"🔍 Full student_info from Firebase: {list(full_student_info.keys()) if full_student_info else 'None'}")
                        print(f"🔍 Student Number from Firebase: {full_student_info.get('student_id') if full_student_info else 'None'}")
                        # Replace student_info completely with fresh Firebase data
                        student_info.clear()
                        student_info.update(full_student_info)
                        # CRITICAL: Preserve RFID after replacing
                        student_info['rfid'] = rfid_to_use
                        print(f"✅ Replaced student_info with complete Firebase data. Keys: {list(student_info.keys())}")
                        print(f"✅ Student Number: {student_info.get('student_id')}")
                        print(f"✅ RFID preserved: {student_info.get('rfid')}")
                    else:
                        print(f"❌ ERROR: get_student_info_by_rfid returned None for RFID: {rfid_to_use}")
                else:
                    print(f"✅ Student Number validated: {student_num}")
                    print(f"✅ RFID present: {student_info.get('rfid')}")
            else:
                print(f"⚠️ CRITICAL ERROR: No RFID available to fetch Student Number!")
                print(f"⚠️ Available student_info keys: {list(student_info.keys()) if student_info else 'None'}")
                print(f"⚠️ current_rfid_for_timer value: {getattr(self, 'current_rfid_for_timer', 'NOT SET')}")
            
            # Guard UI updates
            # requirements_status_label already shows "Incomplete uniform" (no change needed)
            # Keep requirements section visible - only hide when gate control button is clicked
            # Don't schedule hiding - requirements stay visible until guard clicks a button
            print(f"✅ Requirements section remains visible until gate control button is clicked")
            
            # Schedule violation finalization if no retry happens within 30 seconds
            # This ensures violations are finalized even if student doesn't tap again
            if rfid_to_use:
                def finalize_if_no_retry():
                    # Check if violations are still pending (student didn't retry or pass)
                    if rfid_to_use in self.active_session_violations:
                        if len(self.active_session_violations[rfid_to_use]) > 0:
                            print(f"INFO: No retry detected after timeout - finalizing pending violations")
                            self.finalize_session_violations(rfid_to_use)
                
                # Finalize after 30 seconds if no retry (student gave up)
                self.root.after(30000, finalize_if_no_retry)
            
            # Reset detection tracking variables
            self._requirement_detection_counts = {}
            self._permanently_checked_requirements = set()
            
            # CRITICAL: Enable all buttons when incomplete uniform is detected (first entry)
            # This is the first entry attempt, so student hasn't entered yet
            # All buttons should be enabled regardless of previous violations
            # Store student info for button handlers
            self.incomplete_student_info_for_approve = student_info
            
            # Enable all buttons (APPROVE, CANCEL, DENIED) for first entry
            if hasattr(self, 'approve_button'):
                try:
                    self.approve_button.config(state=tk.NORMAL)  # Enable APPROVE button
                    print(f"✅ APPROVE button enabled - incomplete uniform detected for {student_info.get('name', 'Unknown')} (first entry)")
                except Exception as e:
                    print(f"⚠️ WARNING: Could not enable APPROVE button: {e}")
            
            if hasattr(self, 'cancel_button'):
                try:
                    self.cancel_button.config(state=tk.NORMAL)  # Enable CANCEL button
                    print(f"✅ CANCEL button enabled - incomplete uniform detected for {student_info.get('name', 'Unknown')} (first entry)")
                except Exception as e:
                    print(f"⚠️ WARNING: Could not enable CANCEL button: {e}")
            
            if hasattr(self, 'deny_button'):
                try:
                    self.deny_button.config(state=tk.NORMAL)  # Enable DENIED button
                    print(f"✅ DENIED button enabled - incomplete uniform detected for {student_info.get('name', 'Unknown')} (first entry)")
                except Exception as e:
                    print(f"⚠️ WARNING: Could not enable DENIED button: {e}")
            
            # Enable checkboxes immediately when incomplete uniform is detected
            # This allows guard to manually check off items they can see are present
            self._make_requirement_checkboxes_editable(True)
            print(f"✅ Requirement checkboxes enabled - guard can now manually check items")
            
            # Reset RFID status tracker on timeout (timeout = exit, so clear entry status)
            # This ensures tapping ID again after timeout will be treated as fresh start
            if hasattr(self, 'rfid_status_tracker') and rfid_to_use:
                try:
                    self.rfid_status_tracker[rfid_to_use] = 'EXIT'  # Timeout = exit
                    print(f"🔄 Reset RFID status tracker for {rfid_to_use} to EXIT (timeout)")
                except Exception:
                    pass
            
            # Store student info for retry button
            self.incomplete_student_info_for_retry = student_info.copy() if student_info else None
            
            # Removed approve_complete_uniform_btn - use approve_button instead for incomplete uniform
            
            # Main screen updates - show incomplete uniform indicator
            self.update_main_screen_with_incomplete_uniform(student_info)
            
            # Stop detection stream
            if hasattr(self, 'external_detection_stop_event') and self.external_detection_stop_event:
                self.external_detection_stop_event.set()
                print(f"🛑 Stopping detection stream due to timeout")
            
            # Re-enable ID input field - detection completed (incomplete uniform timeout)
            if hasattr(self, 'person_id_entry'):
                self.person_id_entry.config(state=tk.NORMAL)
                print(f"🔓 ID input field enabled - detection completed (timeout)")
            
            # Mark detection as inactive
            self.detection_active = False
            
            # Log activity
            current_time = self.get_current_timestamp()
            self.add_activity_log(f"Incomplete uniform after 15 seconds - Student: {student_info.get('name', 'Unknown')} - {current_time}")
            
            # Save uniform violation to Firebase (timeout = incomplete uniform)
            try:
                student_id = student_info.get('student_id') or student_info.get('student_number')
                if student_id:
                    # Get missing items ONLY from requirement checkboxes if guard has manually interacted with them
                    # CRITICAL: Do NOT use tracker's get_missing_components() - it lists all unconfirmed required parts,
                    # not just actual missing items. Checkboxes start unchecked, so we can only trust them if guard
                    # has manually checked at least one item (indicating they've reviewed and confirmed what's present).
                    missing_items = []
                    try:
                        # Only use requirement checkboxes if guard has manually interacted (checked at least one item)
                        if hasattr(self, 'requirement_checkboxes') and self.requirement_checkboxes:
                            checked_count = 0
                            for req_name, checkbox_data in self.requirement_checkboxes.items():
                                # checkbox_data is a dict with 'var' (BooleanVar) and 'part' (string)
                                if isinstance(checkbox_data, dict) and 'var' in checkbox_data:
                                    var = checkbox_data['var']
                                    if isinstance(var, tk.BooleanVar):
                                        if var.get():  # Checked = present
                                            checked_count += 1
                            
                            # Only use checkboxes if guard has manually checked at least one item
                            # This indicates they've reviewed and we can trust unchecked items as actually missing
                            if checked_count > 0:
                                for req_name, checkbox_data in self.requirement_checkboxes.items():
                                    if isinstance(checkbox_data, dict) and 'var' in checkbox_data:
                                        var = checkbox_data['var']
                                        if isinstance(var, tk.BooleanVar):
                                            if not var.get():  # Unchecked = missing (only if guard has reviewed)
                                                part_name = checkbox_data.get('part', req_name)
                                                missing_items.append(part_name)
                                if missing_items:
                                    print(f"✅ Retrieved missing items from requirement checkboxes (guard reviewed): {missing_items}")
                            else:
                                print(f"⚠️ Guard has not manually reviewed checkboxes - cannot determine missing items accurately")
                                print(f"   - Leaving missing_items empty to avoid storing incorrect data")
                        else:
                            print(f"⚠️ Requirement checkboxes not available - cannot determine missing items")
                        
                        if missing_items:
                            print(f"✅ Using missing_items for violation history: {missing_items}")
                            print(f"   - Only actual missing items (from guard's manual review) will be stored")
                        else:
                            print(f"⚠️ No missing_items to store - better to leave empty than store incorrect data")
                    except Exception as e:
                        print(f"⚠️ Warning: Could not get missing items from checkboxes: {e}")
                        missing_items = []
                    
                    violation_details = "Uniform incomplete after 15-second timeout - required uniform parts not detected"
                    if missing_items and len(missing_items) > 0:
                        violation_details += f" - Missing: {', '.join(missing_items)}"
                    
                    # Save as pending (is_final=False) - will be finalized when student exits/gives up
                    self.save_student_violation_to_firebase(
                        student_id=student_id,
                        violation_type='uniform_violation',
                        timestamp=current_time,
                        student_info=student_info,
                        violation_details=violation_details,
                        is_final=False,  # Pending until student gives up
                        rfid=rfid_to_use,
                        missing_items=missing_items  # Pass missing items list
                    )
                    print(f"SUCCESS: Uniform violation (timeout) saved as PENDING for {student_info.get('name', 'Unknown')}")
                else:
                    print(f"WARNING: Cannot save violation - Student ID not found in student_info")
            except Exception as e:
                print(f"WARNING: Failed to save uniform violation (timeout) to Firebase: {e}")
            
            # Restore guard camera feed (will happen in finally block of _run_stream)
            
        except Exception as e:
            print(f"❌ Error handling incomplete uniform timeout: {e}")

    def _schedule_complete_uniform_auto_entry(self, rfid, student_info):
        """Schedule automatic entry after 8 seconds if guard doesn't click DENIED"""
        try:
            # Cancel any existing timer
            if hasattr(self, 'complete_uniform_auto_entry_timer') and self.complete_uniform_auto_entry_timer:
                try:
                    self.root.after_cancel(self.complete_uniform_auto_entry_timer)
                except Exception:
                    pass
            
            # Schedule auto-entry after 8 seconds
            def auto_entry_callback():
                try:
                    # Check if DENIED was clicked (would cancel auto-entry)
                    if hasattr(self, 'denied_button_clicked') and self.denied_button_clicked:
                        print("⏭️ Auto-entry cancelled - DENIED button was clicked")
                        return
                    
                    # Check if buttons are still enabled (guard hasn't made decision)
                    if hasattr(self, 'approve_button') and self.approve_button:
                        button_state = str(self.approve_button.cget('state'))
                        if button_state == 'normal':
                            print("✅ Auto-entry triggered - 8 seconds elapsed, guard didn't click DENIED")
                            self._execute_complete_uniform_auto_entry(rfid, student_info)
                        else:
                            print("⏭️ Auto-entry skipped - guard already made decision")
                except Exception as e:
                    print(f"⚠️ Error in auto-entry callback: {e}")
            
            self.complete_uniform_auto_entry_timer = self.root.after(8000, auto_entry_callback)  # 8 seconds
            print(f"⏱️ Auto-entry scheduled for 8 seconds - guard can click DENIED to cancel")
            self.add_activity_log("Complete uniform detected - 8 second countdown started, guard can review")
        except Exception as e:
            print(f"⚠️ Error scheduling auto-entry: {e}")
    
    def _execute_complete_uniform_auto_entry(self, rfid, student_info):
        """Execute automatic entry for complete uniform (after 8 seconds)"""
        try:
            print(f"✅ Executing auto-entry for complete uniform: {student_info.get('name', 'Unknown')}")
            
            # Record complete uniform entry
            self.record_complete_uniform_entry(rfid, student_info)
            
            # Open gate
            self.open_gate()
            
            # Update gate status
            self.gate_status_label.config(text="🔓 Gate: Unlocked", fg='#10b981')
            
            # Update main screen
            from datetime import datetime
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            person_info = {
                'id': rfid,
                'name': student_info.get('name', 'Unknown Student'),
                'type': 'student',
                'course': student_info.get('course', 'Unknown'),
                'gender': student_info.get('gender', 'Unknown'),
                'timestamp': current_time,
                'status': 'COMPLETE UNIFORM',
                'action': 'ENTRY',
                'guard_id': self.current_guard_id or 'Unknown'
            }
            
            if self.main_screen_window and self.main_screen_window.winfo_exists():
                self.update_main_screen_with_person(person_info)
                self.add_to_recent_entries(person_info)
            
            # Show success message
            self.show_green_success_message("Entry Approved (Auto)", 
                              "SUCCESS: Complete uniform detected\n"
                              "🚪 Gate is opening\n"
                              "👤 Student may proceed\n"
                              "✅ Auto-entry after 8 seconds")
            
            # Reset all button states (ACCESS GRANTED and CANCEL disabled, DENIED enabled)
            self._reset_denied_button_state()
            self.complete_uniform_student_info = None
            
            # Hide requirements section
            self.hide_requirements_section()
            
            # Clear main screen after gate action
            self.root.after(3500, self.clear_main_screen_after_gate_action)
            
            # Schedule status update after unlock duration
            self.root.after(3500, self.update_gate_status_locked)
            
        except Exception as e:
            print(f"⚠️ Error executing auto-entry: {e}")
            import traceback
            traceback.print_exc()
    
    def show_complete_uniform_popup_with_countdown(self, student_info):
        """Show complete uniform popup with countdown information"""
        try:
            # Use existing popup method but with countdown message
            if hasattr(self, 'show_complete_uniform_popup'):
                self.show_complete_uniform_popup(student_info)
            else:
                # Fallback: show message
                self.show_green_success_message("Complete Uniform Detected", 
                                  "✅ Complete uniform detected\n"
                                  "⏱️ Auto-entry in 8 seconds\n"
                                  "🔘 Guard can click DENIED to review\n"
                                  "🚪 Gate will open automatically")
        except Exception as e:
            print(f"⚠️ Error showing complete uniform popup: {e}")
    
    def _handle_complete_uniform_detected(self):
        """Handle complete uniform detection - record entry and stop detection."""
        try:
            if not self.current_student_info_for_timer:
                print("⚠️ No student info available for complete uniform handling")
                return
            
            student_info = self.current_student_info_for_timer
            rfid = student_info.get('rfid') or student_info.get('id')
            
            if not rfid:
                print("⚠️ No RFID available for complete uniform entry recording")
                return
            
            print(f"✅ Handling complete uniform for: {student_info.get('name', 'Unknown')}")
            
            # Store student info for potential auto-entry
            self.complete_uniform_student_info = student_info.copy() if student_info else None
            
            # CRITICAL: Enable all buttons for complete uniform - guard can review and decide
            # Enable ACCESS GRANTED, CANCEL, and DENIED buttons
            if hasattr(self, 'approve_button'):
                try:
                    self.approve_button.config(state=tk.NORMAL)
                    print(f"✅ APPROVE button enabled - complete uniform detected, guard can review")
                except Exception as e:
                    print(f"⚠️ WARNING: Could not enable APPROVE button: {e}")
            
            if hasattr(self, 'cancel_button'):
                try:
                    self.cancel_button.config(state=tk.NORMAL)
                    print(f"✅ CANCEL button enabled - complete uniform detected, guard can review")
                except Exception as e:
                    print(f"⚠️ WARNING: Could not enable CANCEL button: {e}")
            
            # Reset denied button state flags first (but don't disable - will enable after reset)
            self.denied_button_clicked = False
            self.original_detection_state = None
            self.original_uniform_status = None
            
            # Reset DENIED button text and command to original
            if hasattr(self, 'deny_button') and self.deny_button:
                self.deny_button.config(
                    text="DENIED",
                    command=self.handle_interface_deny
                )
            
            # Enable DENIED button when complete uniform is detected (new RFID tap)
            if hasattr(self, 'deny_button'):
                try:
                    self.deny_button.config(state=tk.NORMAL)
                    print(f"✅ DENIED button enabled - complete uniform detected, guard can review")
                except Exception as e:
                    print(f"⚠️ WARNING: Could not enable DENIED button: {e}")
            
            # Make checkboxes read-only initially (will become editable if DENIED clicked)
            self._make_requirement_checkboxes_editable(False)
            
            # Schedule auto-entry after 8 seconds (if guard doesn't click DENIED)
            self._schedule_complete_uniform_auto_entry(rfid, student_info)
            
            # Update main screen to show COMPLETE UNIFORM status (waiting for guard decision)
            from datetime import datetime
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            person_info = {
                'id': rfid,
                'name': student_info.get('name', 'Unknown Student'),
                'type': 'student',
                'course': student_info.get('course', 'Unknown'),
                'gender': student_info.get('gender', 'Unknown'),
                'timestamp': current_time,
                'status': 'COMPLETE UNIFORM',  # Will show as complete uniform, waiting for guard
                'action': 'DETECTING',  # Not yet entry - waiting for guard decision
                'guard_id': self.current_guard_id or 'Unknown'
            }
            # Mark uniform detection as complete
            self.uniform_detection_complete = True
            if self.main_screen_window and self.main_screen_window.winfo_exists():
                self.update_main_screen_with_person(person_info)
            
            # Show complete uniform popup with countdown info
            self.show_complete_uniform_popup_with_countdown(student_info)
            
            # Stop detection stream
            if hasattr(self, 'external_detection_stop_event') and self.external_detection_stop_event:
                self.external_detection_stop_event.set()
                print(f"🛑 Stopping detection stream - uniform complete")
            
            # Keep requirements section visible - only hide when gate control button is clicked
            # Don't schedule hiding - requirements stay visible until guard clicks a button
            print(f"✅ Requirements section remains visible until gate control button is clicked")
            
            # Reset detection tracking counts but PRESERVE permanently checked requirements
            # This ensures checkboxes stay checked even after detection completes
            self._requirement_detection_counts = {}
            # DO NOT reset _permanently_checked_requirements - preserve checked state
            # self._permanently_checked_requirements = set()  # REMOVED - preserve checked requirements
            
            # Clear incomplete student info and disable approve complete uniform button since uniform is now complete
            self.incomplete_student_info_for_retry = None
            # Removed approve_complete_uniform_btn - no longer needed
            
            # Re-enable ID input field - detection completed (complete uniform)
            if hasattr(self, 'person_id_entry'):
                self.person_id_entry.config(state=tk.NORMAL)
                print(f"🔓 ID input field enabled - detection completed (complete uniform)")
            
            # Mark detection as inactive
            self.detection_active = False
            
        except Exception as e:
            print(f"❌ Error handling complete uniform detection: {e}")
    
    def approve_complete_uniform(self):
        """Approve complete uniform manually - guard confirms uniform is complete"""
        try:
            if not self.incomplete_student_info_for_retry:
                messagebox.showwarning("No Student", "No incomplete uniform detection to approve.")
                return
            
            student_info = self.incomplete_student_info_for_retry
            student_name = student_info.get('name', 'Unknown')
            
            print(f"✅ Guard approving complete uniform for: {student_name}")
            self.add_activity_log(f"Guard approved complete uniform for {student_name}")
            
            # Get RFID
            rfid = student_info.get('rfid')
            if not rfid and hasattr(self, 'current_rfid_for_timer') and self.current_rfid_for_timer:
                rfid = self.current_rfid_for_timer
                student_info['rfid'] = rfid
            elif rfid:
                self.current_rfid_for_timer = rfid
            
            if not rfid:
                messagebox.showerror("Error", "RFID not found. Cannot approve entry.")
                return
            
            # Clear pending violations (guard confirms uniform is complete, so violations don't count)
            if rfid and rfid in self.active_session_violations:
                pending_count = len(self.active_session_violations[rfid])
                if pending_count > 0:
                    print(f"INFO: Guard approved complete uniform - clearing {pending_count} pending violations (violations don't count when guard approves)")
                    del self.active_session_violations[rfid]
            
            # Clear session tracking
            if rfid in self.active_detection_sessions:
                del self.active_detection_sessions[rfid]
            
            # Record complete uniform entry (same as automatic complete uniform detection)
            self.record_complete_uniform_entry(rfid, student_info)
            
            # Open gate (same behavior as automatic complete uniform)
            self.open_gate()
            
            # Update main screen
            if self.main_screen_window and self.main_screen_window.winfo_exists():
                self.update_main_screen_with_gate_status("OPEN", "APPROVED")
            
            # Show success message
            self.show_green_success_message("Uniform Approved", 
                              "SUCCESS: Guard confirmed uniform is complete\n"
                              "🚪 Gate is opening\n"
                              "👤 Student may proceed")
            
            # CRITICAL: Disable buttons after processing
            # Removed approve_complete_uniform_btn - no longer needed
            if hasattr(self, 'approve_button'):
                self.approve_button.config(state=tk.DISABLED)
                print(f"✅ APPROVE button disabled after processing")
            
            if hasattr(self, 'cancel_button'):
                self.cancel_button.config(state=tk.DISABLED)
                print(f"✅ CANCEL button disabled after processing")
            
            # Clear incomplete student info
            self.incomplete_student_info_for_retry = None
            
            # Hide requirements section
            self.hide_requirements_section()
            
            # Stop detection if still running
            if hasattr(self, 'detection_system') and self.detection_system:
                self.detection_system.stop_detection()
            
            if hasattr(self, 'external_detection_stop_event') and self.external_detection_stop_event:
                self.external_detection_stop_event.set()
            
            self.detection_active = False
            
            # Re-enable ID input field
            if hasattr(self, 'person_id_entry'):
                self.person_id_entry.config(state=tk.NORMAL)
            
            print(f"✅ Complete uniform approval processed for: {student_name}")
                
        except Exception as e:
            print(f"❌ Error approving complete uniform: {e}")
            self.add_activity_log(f"❌ Error approving complete uniform: {e}")
            messagebox.showerror("Error", f"Failed to approve complete uniform: {e}")

    def update_main_screen_with_incomplete_uniform(self, student_info):
        """Display student info with incomplete uniform indicator using the same 3-panel layout"""
        try:
            if not self.main_screen_window or not self.main_screen_window.winfo_exists():
                print("WARNING: Main screen window not available for incomplete uniform display")
                return
            
            # CRITICAL: Ensure Student Number is present before displaying
            # Get RFID to fetch Student Number if missing
            rfid_to_use = student_info.get('rfid') or self.current_rfid_for_timer
            
            if rfid_to_use:
                # Use helper function to ensure Student Number is present and valid
                student_info = self._ensure_student_number_from_firebase(student_info, rfid_to_use)
                
                # Final validation: ensure Student Number is not RFID
                student_num_check = student_info.get('student_id') or student_info.get('student_number')
                if not student_num_check or str(student_num_check).strip() == str(rfid_to_use).strip():
                    print(f"⚠️ Student Number still missing in incomplete uniform display, fetching fresh from Firebase...")
                    full_student_info = self.get_student_info_by_rfid(rfid_to_use)
                    if full_student_info:
                        # Replace student_info completely with fresh Firebase data
                        student_info.clear()
                        student_info.update(full_student_info)
                        print(f"✅ Replaced student_info with complete Firebase data in incomplete uniform display")
                        print(f"✅ Student Number: {student_info.get('student_id')}")
                else:
                    print(f"✅ Student Number validated in incomplete uniform display: {student_num_check}")
            
            # Get Student Number - should now be guaranteed from validation above
            student_number = (student_info.get('student_id') or 
                            student_info.get('student_number') or 
                            student_info.get('Student Number'))
            
            # Validate: Student Number should not be RFID, 'Unknown', None, or empty
            rfid_value = student_info.get('rfid') or self.current_rfid_for_timer
            is_valid = (student_number and 
                       student_number != 'Unknown' and 
                       student_number != 'Student Number Not Found' and
                       str(student_number).strip() != '' and
                       (not rfid_value or str(student_number).strip() != str(rfid_value).strip()))
            
            if not is_valid:
                # Last resort: fetch directly from Firebase
                if rfid_value:
                    print(f"⚠️ Student Number still invalid in incomplete uniform display, fetching directly from Firebase for RFID: {rfid_value}")
                    fresh_student_info = self.get_student_info_by_rfid(rfid_value)
                    if fresh_student_info:
                        student_number = fresh_student_info.get('student_id') or fresh_student_info.get('student_number')
                        # Final check: make sure it's valid (not None, not empty, not RFID)
                        if student_number and str(student_number).strip() != '' and str(student_number).strip() != str(rfid_value).strip():
                            print(f"✅ Fetched Student Number directly from Firebase: {student_number}")
                            # Update student_info with the correct Student Number
                            student_info['student_id'] = student_number
                            student_info['student_number'] = student_number
                            student_info['id'] = student_number
                        else:
                            print(f"⚠️ WARNING: Fetched Student Number from Firebase is invalid (None, empty, or equals RFID): {student_number}")
                            student_number = 'Student Number Not Found'
                    else:
                        print(f"⚠️ WARNING: Could not fetch Student Number from Firebase")
                        student_number = 'Student Number Not Found'
                else:
                    print(f"⚠️ WARNING: No RFID available for final fetch")
                    student_number = 'Student Number Not Found'
            
            # Strip whitespace from student number if it exists
            if student_number and student_number != 'Student Number Not Found':
                student_number = str(student_number).strip()
            
            # Get current time
            current_time = self.get_current_timestamp()
            
            # Strip whitespace from course
            course_val = student_info.get('course', 'Unknown Course')
            if course_val:
                course_val = str(course_val).strip()
            
            # Strip whitespace from gender
            gender_val = student_info.get('gender', 'Unknown Gender')
            if gender_val:
                gender_val = str(gender_val).strip()
            
            # Create person_info dict with INCOMPLETE UNIFORM status - use same format as update_main_screen_with_person()
            person_info = {
                'id': student_number,
                'student_id': student_number,
                'student_number': student_number,
                'rfid': rfid_value,  # Ensure RFID is included for image loading
                'name': student_info.get('name', 'Unknown Student'),
                'type': 'student',
                'course': course_val,
                'gender': gender_val,
                'timestamp': current_time,
                'status': 'INCOMPLETE UNIFORM',  # Use INCOMPLETE UNIFORM status
                'action': 'DETECTING'  # Keep as DETECTING to show incomplete state
            }
            
            # Use the same 3-panel layout function for consistency
            self.update_main_screen_with_person(person_info)
            
            print(f"✅ Main screen updated with incomplete uniform indicator for {student_info.get('name', 'Unknown')} using consistent 3-panel layout")
            
            # Note: The 15-second timer is already scheduled in update_main_screen_with_person()
            # No need for separate timer - it will return to standby after 15 seconds
            print(f"⏱️ Main screen will return to standby in 15 seconds (via _schedule_main_screen_clear)")
            
        except Exception as e:
            print(f"ERROR: Error updating main screen with incomplete uniform: {e}")
    
    def _return_to_standby_after_incomplete(self):
        """Return main screen to standby after incomplete uniform display (called after 5 seconds)"""
        try:
            if not self.main_screen_window or not self.main_screen_window.winfo_exists():
                return
            
            self.show_standby_message()
            print(f"✅ Main screen returned to standby after incomplete uniform timeout")
        except Exception as e:
            print(f"ERROR: Error returning to standby: {e}")

    def _refresh_detected_classes_view(self):
        """Periodically clear the list if detections are stale (>1.5s)."""
        try:
            stale_after = 1.5
            now_ts = time.time()
            is_stale = (now_ts - getattr(self, '_last_detect_time', 0.0)) > stale_after
            # Removed detected_classes_listbox - detections now go to activity log
            if is_stale:
                self._last_detected_classes = []
        except Exception:
            pass
        finally:
            try:
                self.root.after(700, self._refresh_detected_classes_view)
            except Exception:
                pass
    
    def create_gate_control_section(self, parent):
        """Create gate control section with interface buttons"""
        gate_frame = tk.LabelFrame(
            parent,
            text="🚪 Gate Control",
            font=('Arial', 12, 'bold'),
            fg='#1f2937',
            bg='#ffffff',
            relief='solid',
            bd=1
        )
        gate_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Gate status display
        status_frame = tk.Frame(gate_frame, bg='#ffffff')
        status_frame.pack(fill=tk.X, padx=10, pady=(10, 5))
        
        self.gate_status_label = tk.Label(
            status_frame,
            text="🔒 Gate: Locked",
            font=('Arial', 11, 'bold'),
            fg='#dc2626',
            bg='#ffffff'
        )
        self.gate_status_label.pack(side=tk.LEFT)
        
        # Arduino connection status
        self.arduino_status_label = tk.Label(
            status_frame,
            text="🔌 Arduino: Disconnected",
            font=('Arial', 10),
            fg='#6b7280',
            bg='#ffffff'
        )
        self.arduino_status_label.pack(side=tk.RIGHT)
        
        # Update Arduino status if already connected
        if hasattr(self, 'arduino_connected') and self.arduino_connected:
            self.update_arduino_connection_status(True)
        
        # Control buttons frame - all 4 buttons in one row
        # Increased padding to give more breathing room
        buttons_frame = tk.Frame(gate_frame, bg='#ffffff')
        buttons_frame.pack(fill=tk.X, padx=5, pady=(5, 5))
        
        # Approve button - DISABLED by default, only enabled when incomplete uniform is detected
        self.approve_button = tk.Button(
            buttons_frame,
            text="ACCESS GRANTED",
            font=('Arial', 9, 'bold'),
            bg='#10b981',
            fg='white',
            relief='raised',
            bd=2,
            padx=5,
            pady=6,
            cursor='hand2',
            activebackground='#059669',
            activeforeground='white',
            command=self.handle_interface_approve,
            state=tk.DISABLED  # Disabled by default - only enabled when incomplete uniform detected
        )
        self.approve_button.pack(side=tk.LEFT, padx=(0, 2), fill=tk.X, expand=True)
        
        # Store incomplete student info for APPROVE button (similar to approve_complete_uniform)
        self.incomplete_student_info_for_approve = None
        
        # Cancel button - DISABLED by default, only enabled after incomplete uniform or DENIED clicked
        self.cancel_button = tk.Button(
            buttons_frame,
            text="CANCEL",
            font=('Arial', 9, 'bold'),
            bg='#6b7280',
            fg='white',
            disabledforeground='#d1d5db',  # Light gray text when disabled - ensures text is visible
            relief='raised',
            bd=2,
            padx=5,
            pady=6,
            cursor='hand2',
            activebackground='#4b5563',
            activeforeground='white',
            command=self.handle_interface_cancel,
            state=tk.DISABLED  # Disabled by default - only enabled after incomplete uniform or DENIED clicked
        )
        self.cancel_button.pack(side=tk.LEFT, padx=(2, 2), fill=tk.X, expand=True)
        
        # Deny button
        self.deny_button = tk.Button(
            buttons_frame,
            text="DENIED",
            font=('Arial', 9, 'bold'),
            bg='#dc2626',
            fg='white',
            relief='raised',
            bd=2,
            padx=5,
            pady=6,
            cursor='hand2',
            activebackground='#b91c1c',
            activeforeground='white',
            command=self.handle_interface_deny
        )
        self.deny_button.pack(side=tk.LEFT, padx=(2, 0), fill=tk.X, expand=True)
        
        # Instructions
        instructions_label = tk.Label(
            gate_frame,
            text="💡 Use buttons to control gate. Event Mode bypasses uniform detection.",
            font=('Arial', 9, 'italic'),
            fg='#6b7280',
            bg='#ffffff',
            wraplength=400
        )
        instructions_label.pack(pady=(0, 10))
    
    def clear_main_screen_after_gate_action(self):
        """Clear main screen after gate control button is clicked"""
        try:
            # CRITICAL: If entry was just saved, schedule a 15-second timer to clear main screen
            # Check if entry was recently saved by checking if detection_active is False and uniform_detection_complete is True
            if hasattr(self, 'detection_active') and not self.detection_active:
                if hasattr(self, 'uniform_detection_complete') and self.uniform_detection_complete:
                    # Entry was just processed - schedule 15-second timer to clear main screen
                    print(f"✅ Entry was just saved - scheduling 15-second timer to return to standby")
                    
                    # Cancel any pending auto-clear timer
                    if hasattr(self, 'main_screen_clear_timer') and self.main_screen_clear_timer is not None:
                        try:
                            self.root.after_cancel(self.main_screen_clear_timer)
                        except Exception:
                            pass
                        self.main_screen_clear_timer = None
                    
                    # Clear violation flags
                    self.incomplete_student_info_for_approve = None
                    
                    # CRITICAL: Clear active_session_violations to prevent blocking the clear timer
                    # Violations have already been finalized and saved to Firebase
                    if hasattr(self, 'active_session_violations') and self.active_session_violations:
                        # Get current RFID if available
                        current_rfid = getattr(self, 'current_rfid_for_timer', None)
                        if current_rfid and current_rfid in self.active_session_violations:
                            del self.active_session_violations[current_rfid]
                            print(f"✅ Cleared active_session_violations for RFID {current_rfid} (violations already finalized)")
                        # Also clear any remaining violations (defensive)
                        self.active_session_violations.clear()
                        print(f"✅ Cleared all active_session_violations (entry was saved)")
                    
                    # CRITICAL: Check if this is an approved violation entry (re-entry scenario)
                    # For approved violation entries, DENIED button should stay enabled for at least 10 seconds
                    # Don't reset denied button state immediately - it will be disabled by the 10-second timer
                    # Check if DENIED button is currently enabled (indicates approved violation re-entry)
                    is_approved_violation_entry = False
                    if hasattr(self, 'deny_button') and self.deny_button:
                        try:
                            current_state = self.deny_button.cget('state')
                            if current_state == tk.NORMAL:
                                # DENIED is enabled - this might be an approved violation entry
                                # Check if we have an approved violation flag (student_id check)
                                is_approved_violation_entry = True
                                print(f"⏸️ DENIED button is enabled - skipping immediate reset (will be disabled after 10 seconds)")
                        except Exception:
                            pass
                    
                    # Reset denied button state only if NOT an approved violation entry
                    # For approved violation entries, DENIED will be disabled by the 10-second timer
                    if not is_approved_violation_entry:
                        self._reset_denied_button_state()
                    else:
                        # For approved violation entries, only reset the state flags, don't disable buttons
                        self.denied_button_clicked = False
                        self.original_detection_state = None
                        self.original_uniform_status = None
                        print(f"✅ Reset denied button state flags (DENIED button will stay enabled for 10 seconds)")
                    
                    # Make checkboxes read-only
                    self._make_requirement_checkboxes_editable(False)
                    
            # Schedule 15-second timer to clear main screen and return to standby
            # CRITICAL: Cancel any existing timer first to avoid conflicts
            if hasattr(self, 'main_screen_clear_timer') and self.main_screen_clear_timer is not None:
                try:
                    self.root.after_cancel(self.main_screen_clear_timer)
                    print(f"🔍 Cancelled existing main screen clear timer before scheduling new one")
                except Exception:
                    pass
            
            self.main_screen_clear_timer = self.root.after(15000, self._clear_main_screen_after_delay)
            print(f"✅ Scheduled main screen clear after 15 seconds (entry was saved) - timer ID: {self.main_screen_clear_timer}")
            
            return  # Exit early - don't clear main screen immediately
            
            # Normal clear behavior (for CANCEL, DENIED without entry, etc.)
            # Cancel any pending auto-clear timer
            if hasattr(self, 'main_screen_clear_timer') and self.main_screen_clear_timer is not None:
                try:
                    self.root.after_cancel(self.main_screen_clear_timer)
                except Exception:
                    pass
                self.main_screen_clear_timer = None
            
            # Clear violation flags so main screen can be cleared
            self.incomplete_student_info_for_approve = None
            
            # Reset denied button state
            self._reset_denied_button_state()
            
            # Make checkboxes read-only
            self._make_requirement_checkboxes_editable(False)
            
            # Clear all widgets from person display frame
            if hasattr(self, 'person_display_frame'):
                for widget in self.person_display_frame.winfo_children():
                    try:
                        widget.destroy()
                    except Exception:
                        pass
            
            # Return to standby mode
            if self.main_screen_window and self.main_screen_window.winfo_exists():
                self.show_standby_message()
            
            print(f"✅ Main screen cleared after gate control action")
        except Exception as e:
            print(f"⚠️ Error clearing main screen after gate action: {e}")
    
    def handle_interface_approve(self):
        """Handle approve button press from interface"""
        try:
            # Check if DENIED was clicked (either complete or incomplete uniform scenario)
            is_denied_clicked_scenario = (
                hasattr(self, 'denied_button_clicked') and 
                self.denied_button_clicked
            )
            
            # Check if this is complete uniform scenario (after DENIED clicked)
            is_complete_uniform_scenario = (
                is_denied_clicked_scenario and
                hasattr(self, 'original_uniform_status') and 
                self.original_uniform_status == 'complete'
            )
            
            # Check if this is incomplete uniform scenario (after DENIED clicked)
            is_incomplete_uniform_scenario = (
                is_denied_clicked_scenario and
                hasattr(self, 'original_uniform_status') and 
                self.original_uniform_status == 'incomplete'
            )
            
            # Check if this is approved violation re-entry scenario (after DENIED clicked)
            is_approved_violation_scenario = (
                is_denied_clicked_scenario and
                hasattr(self, 'approved_violation_student_info') and 
                self.approved_violation_student_info is not None
            )
            
            # Handle approved violation re-entry scenario first
            if is_approved_violation_scenario:
                # Approved violation re-entry: Entry without new violation (violation already stored in Firebase)
                print("SUCCESS: Interface Approve button pressed - Entry without new violation (approved violation re-entry)")
                self.add_activity_log("Interface Approve button pressed - Entry without new violation (approved violation re-entry)")
                
                # Get student info
                student_info = self.approved_violation_student_info
                current_rfid = self.approved_violation_rfid
                
                if not current_rfid or not student_info:
                    print(f"⚠️ WARNING: Cannot process APPROVE - missing RFID or student info")
                    messagebox.showerror("Error", "Cannot process approve - missing student information")
                    return
                
                # CRITICAL: Reset detection flags BEFORE recording entry
                try:
                    self.detection_active = False
                    self.uniform_detection_complete = True
                    print(f"✅ Detection flags reset BEFORE recording entry: detection_active=False, uniform_detection_complete=True")
                    
                    # Clear detection session tracking
                    if current_rfid in self.active_detection_sessions:
                        del self.active_detection_sessions[current_rfid]
                        print(f"✅ Cleared detection session for RFID {current_rfid}")
                    
                    # Stop any running detection
                    if hasattr(self, 'detection_system') and self.detection_system:
                        self.detection_system.stop_detection()
                        print(f"✅ Stopped detection system after approved violation approval")
                except Exception as e:
                    print(f"⚠️ WARNING: Could not reset detection flags: {e}")
                
                # Record entry without new violation (re-entry, violation already stored)
                self.record_complete_uniform_entry(current_rfid, student_info)
                
                # Open gate
                self.open_gate()
                
                # Update gate status
                self.gate_status_label.config(text="🔓 Gate: Unlocked", fg='#10b981')
                
                # Show success message
                self.show_green_success_message("Entry Approved (No New Violation)", 
                                  "SUCCESS: Entry recorded WITHOUT new violation\n"
                                  "🚪 Gate is opening\n"
                                  "👤 Student may proceed\n"
                                  "✅ Re-entry allowed (violation already stored)")
                
                # Reset GRANTED button back to DENIED
                if hasattr(self, 'deny_button') and self.deny_button:
                    try:
                        self.deny_button.config(
                            text="DENIED",
                            command=self.handle_interface_deny,
                            state=tk.DISABLED
                        )
                        print(f"✅ GRANTED button reset to DENIED (approved violation re-entry granted)")
                    except Exception as e:
                        print(f"⚠️ WARNING: Could not reset GRANTED button: {e}")
                
                # Disable all buttons
                if hasattr(self, 'approve_button') and self.approve_button:
                    self.approve_button.config(state=tk.DISABLED)
                if hasattr(self, 'cancel_button') and self.cancel_button:
                    self.cancel_button.config(state=tk.DISABLED)
                
                # Reset denied button state flags
                self.denied_button_clicked = False
                self.original_detection_state = None
                self.original_uniform_status = None
                
                # Clear approved violation info
                self.approved_violation_student_info = None
                self.approved_violation_rfid = None
                
                # Make checkboxes read-only again
                self._make_requirement_checkboxes_editable(False)
                
                # Hide requirements section
                self.hide_requirements_section()
                
                # Schedule 15-second timer to clear main screen
                self.clear_main_screen_after_gate_action()
                
                # Schedule status update after unlock duration
                self.root.after(3500, self.update_gate_status_locked)
                
                return
            
            if is_complete_uniform_scenario:
                # Complete uniform scenario: Entry with violation (incorrect uniform)
                print("SUCCESS: Interface Approve button pressed - Entry with violation (complete uniform scenario)")
                self.add_activity_log("Interface Approve button pressed - Entry with violation (complete uniform)")
                
                # Get student info
                student_info = None
                current_rfid = None
                
                if hasattr(self, 'complete_uniform_student_info') and self.complete_uniform_student_info:
                    student_info = self.complete_uniform_student_info
                    current_rfid = student_info.get('rfid')
                
                if not current_rfid and hasattr(self, 'current_rfid_for_timer'):
                    current_rfid = self.current_rfid_for_timer
                
                if not student_info and current_rfid:
                    student_info = self.get_student_info_by_rfid(current_rfid)
                
                if not current_rfid or not student_info:
                    print(f"⚠️ WARNING: Cannot process APPROVE - missing RFID or student info")
                    messagebox.showerror("Error", "Cannot process approve - missing student information")
                    return
                
                # CRITICAL: Stop detection before processing entry
                try:
                    if hasattr(self, 'detection_system') and self.detection_system:
                        self.detection_system.stop_detection()
                        print(f"🛑 Stopped detection system before processing ACCESS GRANTED")
                    self.detection_active = False
                    # Don't set uniform_detection_complete yet - it will be set to True after entry is saved
                except Exception as e:
                    print(f"⚠️ Warning: Could not stop detection: {e}")
                
                # CRITICAL: Ensure violations exist in active_session_violations before calling handle_approve_button
                # If DENIED was clicked early during detection, violations might not be in active_session_violations yet
                if current_rfid not in getattr(self, 'active_session_violations', {}) or len(getattr(self, 'active_session_violations', {}).get(current_rfid, [])) == 0:
                    print(f"⚠️ WARNING: No violations found in active_session_violations - creating violation entry manually")
                    # Create a violation entry based on manual checkbox selections or generic violation
                    if current_rfid not in getattr(self, 'active_session_violations', {}):
                        self.active_session_violations[current_rfid] = []
                    
                    # Get missing items from requirement checkboxes if available
                    missing_items = []
                    if hasattr(self, 'requirement_checkboxes') and self.requirement_checkboxes:
                        for req_name, checkbox_data in self.requirement_checkboxes.items():
                            # checkbox_data is a dict with 'var' (BooleanVar) and 'part' (string)
                            if isinstance(checkbox_data, dict) and 'var' in checkbox_data:
                                var = checkbox_data['var']
                                if isinstance(var, tk.BooleanVar):
                                    if not var.get():  # Unchecked = missing
                                        part_name = checkbox_data.get('part', req_name)
                                        missing_items.append(part_name)
                    
                    # Create violation entry
                    violation_entry = {
                        'student_id': student_info.get('student_id') or student_info.get('student_number'),
                        'violation_type': 'incomplete_uniform',
                        'timestamp': self.get_current_timestamp(),
                        'details': 'Incomplete uniform violation (guard approved with violation after DENIED)',
                        'student_info': student_info,
                        'missing_items': missing_items if missing_items else ['Uniform violation detected']
                    }
                    self.active_session_violations[current_rfid].append(violation_entry)
                    print(f"✅ Created violation entry manually for ACCESS GRANTED after DENIED (complete uniform scenario)")
                
                # Entry with violation (guard sees problem)
                entry_saved = self.handle_approve_button()
                
                if entry_saved:
                    print(f"✅ Entry with violation saved successfully")
                    
                    # CRITICAL: Verify entry record is saved before allowing next tap
                    import time
                    verify_entry = False
                    for attempt in range(3):
                        delay = 0.5 + (attempt * 0.3)  # 0.5s, 0.8s, 1.1s
                        time.sleep(delay)
                        verify_entry = self.check_student_has_entry_record(current_rfid)
                        if verify_entry:
                            print(f"✅ VERIFIED: Entry record confirmed in Firebase (attempt {attempt + 1}) - next tap will be EXIT")
                            break
                        else:
                            print(f"⚠️ Attempt {attempt + 1}: Entry record not yet found, retrying...")
                    
                    if not verify_entry:
                        print(f"⚠️ WARNING: Entry record not found after 3 attempts - may need more time")
                    else:
                        print(f"✅ SUCCESS: Entry verified in Firebase - next tap will timeout (EXIT)")
                else:
                    print(f"⚠️ WARNING: Entry may not have been saved")
                
                # CRITICAL: Clear detection session to prevent restart on next tap
                try:
                    if current_rfid in getattr(self, 'active_detection_sessions', {}):
                        del self.active_detection_sessions[current_rfid]
                        print(f"✅ Cleared detection session for RFID {current_rfid}")
                except Exception as e:
                    print(f"⚠️ Warning: Could not clear detection session: {e}")
                
                # Send approve command to Arduino
                self.send_arduino_command("APPROVE")
                
                # Open gate
                self.open_gate()
                
                # Update gate status
                self.gate_status_label.config(text="🔓 Gate: Unlocked", fg='#10b981')
                
                # Show success message
                self.show_green_success_message("Entry Approved (With Violation)", 
                                  "SUCCESS: Entry recorded with violation\n"
                                  "🚪 Gate is opening\n"
                                  "👤 Student may proceed\n"
                                  "⚠️ Violation recorded")
                
                # CRITICAL: Schedule 15-second timer to clear main screen and return to standby
                # This ensures the entry status is displayed for 15 seconds, then returns to standby
                self.clear_main_screen_after_gate_action()
                
                # NOTE: handle_approve_button() already updates main screen with entry status
                # Don't overwrite it with gate status - entry status is more important
            elif is_incomplete_uniform_scenario:
                # Incomplete uniform scenario after DENIED clicked: Entry with violation
                print("SUCCESS: Interface Approve button pressed - Entry with violation (incomplete uniform after DENIED)")
                self.add_activity_log("Interface Approve button pressed - Entry with violation (incomplete uniform after DENIED)")
                
                # Get student info
                student_info = None
                current_rfid = None
                
                if hasattr(self, 'incomplete_student_info_for_approve') and self.incomplete_student_info_for_approve:
                    student_info = self.incomplete_student_info_for_approve
                    current_rfid = student_info.get('rfid')
                
                if not current_rfid and hasattr(self, 'current_rfid_for_timer'):
                    current_rfid = self.current_rfid_for_timer
                
                if not student_info and current_rfid:
                    student_info = self.get_student_info_by_rfid(current_rfid)
                
                if not current_rfid or not student_info:
                    print(f"⚠️ WARNING: Cannot process APPROVE - missing RFID or student info")
                    messagebox.showerror("Error", "Cannot process approve - missing student information")
                    return
                
                # CRITICAL: Stop detection before processing entry
                try:
                    if hasattr(self, 'detection_system') and self.detection_system:
                        self.detection_system.stop_detection()
                        print(f"🛑 Stopped detection system before processing ACCESS GRANTED")
                    self.detection_active = False
                    # Don't set uniform_detection_complete yet - it will be set to True after entry is saved
                except Exception as e:
                    print(f"⚠️ Warning: Could not stop detection: {e}")
                
                # CRITICAL: Ensure violations exist in active_session_violations before calling handle_approve_button
                # If DENIED was clicked early during detection, violations might not be in active_session_violations yet
                if current_rfid not in getattr(self, 'active_session_violations', {}) or len(getattr(self, 'active_session_violations', {}).get(current_rfid, [])) == 0:
                    print(f"⚠️ WARNING: No violations found in active_session_violations - creating violation entry manually")
                    # Create a violation entry based on manual checkbox selections or generic violation
                    if current_rfid not in getattr(self, 'active_session_violations', {}):
                        self.active_session_violations[current_rfid] = []
                    
                    # Get missing items from requirement checkboxes if available
                    missing_items = []
                    if hasattr(self, 'requirement_checkboxes') and self.requirement_checkboxes:
                        for req_name, checkbox_data in self.requirement_checkboxes.items():
                            # checkbox_data is a dict with 'var' (BooleanVar) and 'part' (string)
                            if isinstance(checkbox_data, dict) and 'var' in checkbox_data:
                                var = checkbox_data['var']
                                if isinstance(var, tk.BooleanVar):
                                    if not var.get():  # Unchecked = missing
                                        part_name = checkbox_data.get('part', req_name)
                                        missing_items.append(part_name)
                    
                    # Create violation entry
                    violation_entry = {
                        'student_id': student_info.get('student_id') or student_info.get('student_number'),
                        'violation_type': 'incomplete_uniform',
                        'timestamp': self.get_current_timestamp(),
                        'details': 'Incomplete uniform violation (guard approved with violation)',
                        'student_info': student_info,
                        'missing_items': missing_items if missing_items else ['Uniform violation detected']
                    }
                    self.active_session_violations[current_rfid].append(violation_entry)
                    print(f"✅ Created violation entry manually for ACCESS GRANTED after DENIED")
                
                # Debug: Check if violations were created
                print(f"🔍 DEBUG: Checking violations before handle_approve_button")
                print(f"   Current RFID: {current_rfid}")
                print(f"   active_session_violations keys: {list(getattr(self, 'active_session_violations', {}).keys())}")
                if current_rfid in getattr(self, 'active_session_violations', {}):
                    print(f"   Violations for {current_rfid}: {len(getattr(self, 'active_session_violations', {}).get(current_rfid, []))}")
                    for i, v in enumerate(getattr(self, 'active_session_violations', {}).get(current_rfid, [])):
                        print(f"     Violation {i+1}: {v.get('violation_type', 'unknown')} - {v.get('details', 'no details')}")
                else:
                    print(f"   ⚠️ WARNING: RFID {current_rfid} not found in active_session_violations!")
                
                # Entry with violation (guard sees problem with incomplete uniform)
                try:
                    entry_saved = self.handle_approve_button()
                    print(f"🔍 DEBUG: handle_approve_button returned: {entry_saved}")
                except Exception as e:
                    print(f"❌ ERROR: Exception in handle_approve_button: {e}")
                    import traceback
                    traceback.print_exc()
                    messagebox.showerror("Error", f"Failed to process entry: {e}")
                    entry_saved = False
                
                if entry_saved:
                    print(f"✅ Entry with violation saved successfully")
                    
                    # CRITICAL: Verify entry record is saved before allowing next tap
                    import time
                    verify_entry = False
                    for attempt in range(3):
                        delay = 0.5 + (attempt * 0.3)  # 0.5s, 0.8s, 1.1s
                        time.sleep(delay)
                        verify_entry = self.check_student_has_entry_record(current_rfid)
                        if verify_entry:
                            print(f"✅ VERIFIED: Entry record confirmed in Firebase (attempt {attempt + 1}) - next tap will be EXIT")
                            break
                        else:
                            print(f"⚠️ Attempt {attempt + 1}: Entry record not yet found, retrying...")
                    
                    if not verify_entry:
                        print(f"⚠️ WARNING: Entry record not found after 3 attempts - may need more time")
                    else:
                        print(f"✅ SUCCESS: Entry verified in Firebase - next tap will timeout (EXIT)")
                else:
                    print(f"⚠️ WARNING: Entry may not have been saved - handle_approve_button returned False")
                
                # CRITICAL: Clear detection session to prevent restart on next tap
                try:
                    if current_rfid in getattr(self, 'active_detection_sessions', {}):
                        del self.active_detection_sessions[current_rfid]
                        print(f"✅ Cleared detection session for RFID {current_rfid}")
                except Exception as e:
                    print(f"⚠️ Warning: Could not clear detection session: {e}")
                
                # Send approve command to Arduino
                self.send_arduino_command("APPROVE")
                
                # Open gate
                self.open_gate()
                
                # Update gate status
                self.gate_status_label.config(text="🔓 Gate: Unlocked", fg='#10b981')
                
                # Show success message
                self.show_green_success_message("Entry Approved (With Violation)", 
                                  "SUCCESS: Entry recorded with violation\n"
                                  "🚪 Gate is opening\n"
                                  "👤 Student may proceed\n"
                                  "⚠️ Violation recorded")
                
                # CRITICAL: Schedule 15-second timer to clear main screen and return to standby
                # This ensures the entry status is displayed for 15 seconds, then returns to standby
                self.clear_main_screen_after_gate_action()
                
                # NOTE: handle_approve_button() already updates main screen with entry status
                # Don't overwrite it with gate status - entry status is more important
                # Only update gate status if main screen wasn't updated by handle_approve_button
                if self.main_screen_window and self.main_screen_window.winfo_exists():
                    # Check if main screen was already updated by handle_approve_button
                    # If not, update with gate status as fallback
                    try:
                        # handle_approve_button should have called update_main_screen_with_permanent_student
                        # which shows entry status - don't overwrite it
                        pass  # Don't call update_main_screen_with_gate_status - it would overwrite entry status
                    except Exception as e:
                        print(f"⚠️ Warning: Could not check main screen update: {e}")
            else:
                # Normal incomplete uniform scenario (DENIED not clicked)
                print("SUCCESS: Interface Approve button pressed - Unlocking gate")
                self.add_activity_log("Interface Approve button pressed - Unlocking gate")
                
                # Get student info for verification
                student_info = None
                current_rfid = None
                
                if hasattr(self, 'incomplete_student_info_for_approve') and self.incomplete_student_info_for_approve:
                    student_info = self.incomplete_student_info_for_approve
                    current_rfid = student_info.get('rfid')
                
                if not current_rfid and hasattr(self, 'current_rfid_for_timer'):
                    current_rfid = self.current_rfid_for_timer
                
                # CRITICAL: Stop detection before processing entry
                try:
                    if hasattr(self, 'detection_system') and self.detection_system:
                        self.detection_system.stop_detection()
                        print(f"🛑 Stopped detection system before processing ACCESS GRANTED")
                    self.detection_active = False
                    # Don't set uniform_detection_complete yet - it will be set to True after entry is saved
                except Exception as e:
                    print(f"⚠️ Warning: Could not stop detection: {e}")
                
                # CRITICAL: Ensure violations exist in active_session_violations before calling handle_approve_button
                # If detection completed but violations weren't stored, create them now
                if current_rfid and (current_rfid not in getattr(self, 'active_session_violations', {}) or len(getattr(self, 'active_session_violations', {}).get(current_rfid, [])) == 0):
                    print(f"⚠️ WARNING: No violations found in active_session_violations - creating violation entry manually")
                    # Create a violation entry
                    if current_rfid not in getattr(self, 'active_session_violations', {}):
                        self.active_session_violations[current_rfid] = []
                    
                    # Get missing items from requirement checkboxes if available
                    missing_items = []
                    if hasattr(self, 'requirement_checkboxes') and self.requirement_checkboxes:
                        for req_name, checkbox_data in self.requirement_checkboxes.items():
                            # checkbox_data is a dict with 'var' (BooleanVar) and 'part' (string)
                            if isinstance(checkbox_data, dict) and 'var' in checkbox_data:
                                var = checkbox_data['var']
                                if isinstance(var, tk.BooleanVar):
                                    if not var.get():  # Unchecked = missing
                                        part_name = checkbox_data.get('part', req_name)
                                        missing_items.append(part_name)
                    
                    # Create violation entry
                    violation_entry = {
                        'student_id': student_info.get('student_id') or student_info.get('student_number') if student_info else None,
                        'violation_type': 'incomplete_uniform',
                        'timestamp': self.get_current_timestamp(),
                        'details': 'Incomplete uniform violation (guard approved with violation)',
                        'student_info': student_info,
                        'missing_items': missing_items if missing_items else ['Uniform violation detected']
                    }
                    self.active_session_violations[current_rfid].append(violation_entry)
                    print(f"✅ Created violation entry manually for ACCESS GRANTED")
                
                # CRITICAL: Call handle_approve_button() to save entry to Firebase, update main screen, and add to recent entries
                entry_saved = self.handle_approve_button()
                
                if entry_saved:
                    print(f"✅ Entry saved successfully via handle_approve_button()")
                    
                    # CRITICAL: Verify entry record is saved before allowing next tap
                    if current_rfid:
                        import time
                        verify_entry = False
                        for attempt in range(3):
                            delay = 0.5 + (attempt * 0.3)  # 0.5s, 0.8s, 1.1s
                            time.sleep(delay)
                            verify_entry = self.check_student_has_entry_record(current_rfid)
                            if verify_entry:
                                print(f"✅ VERIFIED: Entry record confirmed in Firebase (attempt {attempt + 1}) - next tap will be EXIT")
                                break
                            else:
                                print(f"⚠️ Attempt {attempt + 1}: Entry record not yet found, retrying...")
                        
                        if not verify_entry:
                            print(f"⚠️ WARNING: Entry record not found after 3 attempts - may need more time")
                        else:
                            print(f"✅ SUCCESS: Entry verified in Firebase - next tap will timeout (EXIT)")
                else:
                    print(f"⚠️ WARNING: handle_approve_button() returned False - entry may not have been saved")
                
                # CRITICAL: Clear detection session to prevent restart on next tap
                if current_rfid:
                    try:
                        if current_rfid in getattr(self, 'active_detection_sessions', {}):
                            del self.active_detection_sessions[current_rfid]
                            print(f"✅ Cleared detection session for RFID {current_rfid}")
                    except Exception as e:
                        print(f"⚠️ Warning: Could not clear detection session: {e}")
                
                # Send approve command to Arduino
                self.send_arduino_command("APPROVE")
                
                # Open gate
                self.open_gate()
                
                # Update gate status
                self.gate_status_label.config(text="🔓 Gate: Unlocked", fg='#10b981')
                
                # Show success message
                self.show_green_success_message("Gate Approved", 
                                  "SUCCESS: Gate is unlocking\n"
                                  "🚪 Person may proceed\n"
                                  "✅ Entry recorded with violation")
                
                # CRITICAL: Schedule 15-second timer to clear main screen and return to standby
                # This ensures the entry status is displayed for 15 seconds, then returns to standby
                self.clear_main_screen_after_gate_action()
                
                # NOTE: handle_approve_button() already updates main screen with entry status
                # Don't overwrite it with gate status - entry status is more important
            
            # Reset all button states after processing
            self._reset_denied_button_state()
            
            # Make checkboxes read-only again
            self._make_requirement_checkboxes_editable(False)
            
            # Cancel auto-entry timer if exists
            if hasattr(self, 'complete_uniform_auto_entry_timer') and self.complete_uniform_auto_entry_timer:
                try:
                    self.root.after_cancel(self.complete_uniform_auto_entry_timer)
                    self.complete_uniform_auto_entry_timer = None
                except Exception:
                    pass
            
            # Reset pending operations
            self.incomplete_student_info_for_approve = None
            self.complete_uniform_student_info = None
            
            # CRITICAL: Only hide requirements section and clear main screen if entry was NOT saved
            # If entry was saved, keep the entry status visible on main screen
            # Check if entry was saved by checking detection flags (entry was saved if detection_active is False and uniform_detection_complete is True)
            entry_was_saved = (
                hasattr(self, 'detection_active') and not self.detection_active and
                hasattr(self, 'uniform_detection_complete') and self.uniform_detection_complete
            )
            
            if not entry_was_saved:
                # Entry was not saved - hide requirements and clear main screen
                self.hide_requirements_section()
                self.clear_main_screen_after_gate_action()
            else:
                # Entry was saved - just hide requirements section (don't clear main screen)
                # The main screen should show entry status
                try:
                    if hasattr(self, 'requirements_frame') and self.requirements_frame:
                        self.requirements_frame.pack_forget()
                        # Clear student info label when hiding
                        if hasattr(self, 'requirements_student_info_label') and self.requirements_student_info_label:
                            self.requirements_student_info_label.config(text="")
                        print(f"✅ Requirements section hidden (entry status remains visible on main screen)")
                except Exception as e:
                    print(f"⚠️ Warning: Could not hide requirements section: {e}")
            
            # Schedule status update after unlock duration
            self.root.after(3500, self.update_gate_status_locked)
            
        except Exception as e:
            print(f"ERROR: Error handling interface approve: {e}")
            self.add_activity_log(f"Error handling interface approve: {e}")
            import traceback
            traceback.print_exc()
    
    def handle_interface_cancel(self):
        """Handle cancel button press - Four functions:
        1. If denied_button_clicked (complete uniform): Permanently lock gate and return to standby
        2. If denied_button_clicked (incomplete uniform): Lock gate and return to standby
        3. If denied_button_clicked (approved violation re-entry): Lock gate quickly and return to standby
        4. If not: Entry without violation (bypass incomplete uniform)
        """
        try:
            # CRITICAL: Check if this is approved violation re-entry scenario
            is_approved_violation_scenario = (
                hasattr(self, 'approved_violation_student_info') and 
                self.approved_violation_student_info is not None
            )
            
            # Check if DENIED was clicked - if so, lock gate and return to standby
            if hasattr(self, 'denied_button_clicked') and self.denied_button_clicked:
                is_complete_uniform = (
                    hasattr(self, 'original_uniform_status') and 
                    self.original_uniform_status == 'complete'
                )
                
                # Handle approved violation re-entry scenario first
                if is_approved_violation_scenario:
                    print("INFO: Cancel button pressed after DENIED (approved violation re-entry) - Locking gate quickly and returning to standby")
                    self.add_activity_log("Cancel button pressed after DENIED (approved violation re-entry) - Locking gate, returning to standby")
                    
                    # Get RFID from approved violation info
                    cancel_rfid = self.approved_violation_rfid
                    
                    # CRITICAL: Delete the entry record that was just created
                    # Since guard cancelled, student didn't actually enter, so next tap should be ENTRY not EXIT
                    if cancel_rfid:
                        try:
                            self.delete_most_recent_entry_record(cancel_rfid)
                            print(f"✅ Deleted entry record for RFID {cancel_rfid} - next tap will be treated as ENTRY")
                        except Exception as e:
                            print(f"⚠️ WARNING: Could not delete entry record: {e}")
                    
                    # Send deny command to Arduino (lock gate quickly)
                    self.send_arduino_command("DENY")
                    
                    # Update gate status
                    self.gate_status_label.config(text="🔒 Gate: Locked", fg='#dc2626')
                    
                    # Reset GRANTED button back to DENIED
                    if hasattr(self, 'deny_button') and self.deny_button:
                        try:
                            self.deny_button.config(
                                text="DENIED",
                                command=self.handle_interface_deny,
                                state=tk.DISABLED
                            )
                            print(f"✅ GRANTED button reset to DENIED (approved violation re-entry cancelled)")
                        except Exception as e:
                            print(f"⚠️ WARNING: Could not reset GRANTED button: {e}")
                    
                    # Disable all buttons
                    if hasattr(self, 'approve_button') and self.approve_button:
                        self.approve_button.config(state=tk.DISABLED)
                    if hasattr(self, 'cancel_button') and self.cancel_button:
                        self.cancel_button.config(state=tk.DISABLED)
                    
                    # Reset denied button state flags
                    self.denied_button_clicked = False
                    self.original_detection_state = None
                    self.original_uniform_status = None
                    
                    # Clear approved violation info
                    self.approved_violation_student_info = None
                    self.approved_violation_rfid = None
                    
                    # Make checkboxes read-only again
                    self._make_requirement_checkboxes_editable(False)
                    
                    # Hide requirements section
                    self.hide_requirements_section()
                    
                    # Clear main screen and return to standby immediately (no delay)
                    if hasattr(self, 'person_display_frame'):
                        for widget in self.person_display_frame.winfo_children():
                            try:
                                widget.destroy()
                            except Exception:
                                pass
                    
                    if self.main_screen_window and self.main_screen_window.winfo_exists():
                        self.show_standby_message()
                    
                    # Show info message
                    self.show_green_success_message("Gate Locked", 
                                      "Gate is locked\n"
                                      "❌ Entry cancelled\n"
                                      "🔄 Returned to standby")
                    
                    return
                
                if is_complete_uniform:
                    print("INFO: Cancel button pressed after DENIED (complete uniform) - Permanently locking gate and returning to standby")
                    self.add_activity_log("Cancel button pressed after DENIED (complete uniform) - Permanently locking gate, returning to standby")
                else:
                    print("INFO: Cancel button pressed after DENIED - Locking gate and returning to standby")
                    self.add_activity_log("Cancel button pressed after DENIED - Locking gate, returning to standby")
                
                # Send deny command to Arduino (keep gate locked)
                self.send_arduino_command("DENY")
                
                # Update gate status
                self.gate_status_label.config(text="🔒 Gate: Locked", fg='#dc2626')
                
                # Cancel auto-entry timer if exists
                if hasattr(self, 'complete_uniform_auto_entry_timer') and self.complete_uniform_auto_entry_timer:
                    try:
                        self.root.after_cancel(self.complete_uniform_auto_entry_timer)
                        self.complete_uniform_auto_entry_timer = None
                    except Exception:
                        pass
                
                # Reset all button states
                self._reset_denied_button_state()
                
                # Make checkboxes read-only again
                self._make_requirement_checkboxes_editable(False)
                
                # Reset pending operations
                self.incomplete_student_info_for_approve = None
                self.complete_uniform_student_info = None
                
                # Hide requirements section
                self.hide_requirements_section()
                
                # Clear main screen and return to standby
                self.clear_main_screen_after_gate_action()
                
                # Show info message
                if is_complete_uniform:
                    self.show_green_success_message("Gate Permanently Locked", 
                                      "Gate is permanently locked\n"
                                      "❌ Entry cancelled\n"
                                      "🔄 Returned to standby")
                else:
                    self.show_green_success_message("Gate Locked", 
                                      "Gate is locked\n"
                                      "❌ Entry cancelled\n"
                                      "🔄 Returned to standby")
                
                return
            
            # Default behavior: Entry without violation (bypass incomplete uniform)
            print("SUCCESS: Cancel button pressed - Entry without violation (overriding incomplete uniform)")
            self.add_activity_log("Cancel button pressed - Entry without violation")
            
            # Get student info from stored incomplete student info
            student_info = None
            current_rfid = None
            
            if hasattr(self, 'incomplete_student_info_for_approve') and self.incomplete_student_info_for_approve:
                student_info = self.incomplete_student_info_for_approve
                current_rfid = student_info.get('rfid')
                print(f"✅ Using stored incomplete student info for CANCEL: {student_info.get('name', 'Unknown')}")
            
            # Fallback: Get current student info from detection system
            if not current_rfid:
                if hasattr(self, 'detection_system') and self.detection_system:
                    detection_service = getattr(self.detection_system, 'detection_service', None)
                    if detection_service:
                        current_rfid = getattr(detection_service, 'current_rfid', None)
            
            # Try backup RFID storage
            if not current_rfid and hasattr(self, 'current_rfid_for_timer'):
                current_rfid = self.current_rfid_for_timer
            
            # If we don't have student_info yet, fetch it
            if not student_info and current_rfid:
                student_info = self.get_student_info_by_rfid(current_rfid)
            
            # Check if we have student info and RFID to proceed
            if not current_rfid or not student_info:
                print(f"⚠️ WARNING: Cannot process CANCEL - missing RFID or student info")
                messagebox.showerror("Error", "Cannot process cancel - missing student information")
                return
            
            # CRITICAL: Reset detection flags BEFORE recording entry
            # This ensures update_main_screen_with_person shows ENTRY status, not DETECTING
            # record_complete_uniform_entry calls update_main_screen_with_permanent_student
            # which calls update_main_screen_with_person, and that function checks detection_active flag
            try:
                # Reset detection flags FIRST (before recording entry and updating main screen)
                self.detection_active = False
                self.uniform_detection_complete = True  # Mark as complete since approved
                print(f"✅ Detection flags reset BEFORE recording entry: detection_active=False, uniform_detection_complete=True")
                
                # Clear detection session tracking
                if current_rfid in self.active_detection_sessions:
                    del self.active_detection_sessions[current_rfid]
                    print(f"✅ Cleared detection session for RFID {current_rfid}")
                
                # Stop any running detection
                if hasattr(self, 'detection_system') and self.detection_system:
                    self.detection_system.stop_detection()
                    print(f"✅ Stopped detection system after cancel approval")
            except Exception as e:
                print(f"⚠️ WARNING: Could not reset detection flags: {e}")
            
            # CRITICAL: Record entry as COMPLETE UNIFORM (no violation)
            # This clears violations and allows entry without violation
            # This will update the main screen with ENTRY status (flags are already set above)
            self.record_complete_uniform_entry(current_rfid, student_info)
            
            # Open gate (same as complete uniform)
            self.open_gate()
            
            # Update gate status
            self.gate_status_label.config(text="🔓 Gate: Unlocked", fg='#10b981')
            
            # Show success message
            self.show_green_success_message("Entry Approved (No Violation)", 
                              "SUCCESS: Entry recorded without violation\n"
                              "🚪 Gate is opening\n"
                              "👤 Student may proceed\n"
                              "✅ Uniform violation overridden")
            
            # NOTE: record_complete_uniform_entry already updates main screen with entry status
            # Don't overwrite it with gate status - entry status is more important
            
            # Disable buttons after processing
            if hasattr(self, 'approve_button') and self.approve_button:
                self.approve_button.config(state=tk.DISABLED)
            if hasattr(self, 'cancel_button') and self.cancel_button:
                self.cancel_button.config(state=tk.DISABLED)
            
            # Reset pending operations
            self.incomplete_student_info_for_approve = None
            
            # Hide requirements section
            self.hide_requirements_section()
            
            # Clear main screen after gate control action
            # This will schedule a 15-second timer to clear the screen (entry status will be visible for 15 seconds)
            self.clear_main_screen_after_gate_action()
            
            # Schedule status update after unlock duration
            self.root.after(3500, self.update_gate_status_locked)
            
        except Exception as e:
            print(f"ERROR: Error handling interface cancel: {e}")
            self.add_activity_log(f"Error handling interface cancel: {e}")
            import traceback
            traceback.print_exc()
    
    def handle_interface_deny(self):
        """Handle deny button press - toggles to GRANTED and makes checkboxes editable"""
        try:
            # Check if this is first click (toggle to GRANTED) or second click (already GRANTED)
            if not self.denied_button_clicked:
                # First click - toggle to GRANTED
                print("INFO: Interface Deny button pressed - Toggling to GRANTED, waiting for guard decision")
                self.add_activity_log("Deny button pressed - Toggled to GRANTED, waiting for guard decision")
                
                # Cancel auto-entry timer if it exists (complete uniform scenario)
                if hasattr(self, 'complete_uniform_auto_entry_timer') and self.complete_uniform_auto_entry_timer:
                    try:
                        self.root.after_cancel(self.complete_uniform_auto_entry_timer)
                        self.complete_uniform_auto_entry_timer = None
                        print("✅ Auto-entry timer cancelled - guard clicked DENIED")
                    except Exception:
                        pass
                
                # Cancel any existing main screen auto-clear timer (prevent premature clearing)
                if hasattr(self, 'main_screen_clear_timer') and self.main_screen_clear_timer is not None:
                    try:
                        self.root.after_cancel(self.main_screen_clear_timer)
                        self.main_screen_clear_timer = None
                        print("✅ Cancelled main screen auto-clear timer - DENIED clicked, waiting for guard decision")
                    except Exception:
                        pass
                
                # Store original detection state
                self.denied_button_clicked = True
                self.original_detection_state = self.uniform_detection_complete
                
                # Determine original uniform status
                if self.uniform_detection_complete:
                    self.original_uniform_status = 'complete'
                elif hasattr(self, 'incomplete_student_info_for_approve') and self.incomplete_student_info_for_approve:
                    self.original_uniform_status = 'incomplete'
                else:
                    # Check if there are active violations
                    if hasattr(self, 'active_session_violations') and self.active_session_violations:
                        current_rfid = getattr(self, 'current_rfid_for_timer', None)
                        if current_rfid and current_rfid in self.active_session_violations:
                            self.original_uniform_status = 'incomplete'
                        else:
                            self.original_uniform_status = 'complete'
                    else:
                        self.original_uniform_status = 'complete'
                
                # Toggle button text from DENIED to GRANTED
                if hasattr(self, 'deny_button') and self.deny_button:
                    self.deny_button.config(
                        text="GRANTED",
                        command=self.handle_interface_granted_after_deny
                    )
                    print(f"✅ DENIED button toggled to GRANTED")
                
                # Make requirement checkboxes editable
                self._make_requirement_checkboxes_editable(True)
                
                # Stop detection immediately when DENIED is clicked during detection
                if hasattr(self, 'detection_active') and self.detection_active:
                    print("🛑 Stopping detection - DENIED button clicked during detection")
                    self.detection_active = False
                    
                    # Stop detection system
                    if hasattr(self, 'detection_system') and self.detection_system:
                        try:
                            self.detection_system.stop_detection()
                            print("✅ Detection system stopped")
                        except Exception as e:
                            print(f"⚠️ Error stopping detection system: {e}")
                    
                    # Stop external detection stop event
                    if hasattr(self, 'external_detection_stop_event') and self.external_detection_stop_event:
                        try:
                            self.external_detection_stop_event.set()
                            print("✅ External detection stop event set")
                        except Exception as e:
                            print(f"⚠️ Error setting stop event: {e}")
                
                # CRITICAL: Check if this is approved violation re-entry scenario
                is_approved_violation_scenario = (
                    hasattr(self, 'approved_violation_student_info') and 
                    self.approved_violation_student_info is not None
                )
                
                # Enable buttons based on scenario
                if is_approved_violation_scenario:
                    # Approved violation re-entry: Enable ACCESS GRANTED and CANCEL
                    if hasattr(self, 'approve_button') and self.approve_button:
                        self.approve_button.config(state=tk.NORMAL)
                        print(f"✅ ACCESS GRANTED button enabled - approved violation re-entry scenario")
                    if hasattr(self, 'cancel_button') and self.cancel_button:
                        self.cancel_button.config(state=tk.NORMAL)
                        print(f"✅ CANCEL button enabled - approved violation re-entry scenario")
                else:
                    # Normal scenario: Enable all buttons (for complete uniform scenario)
                    if hasattr(self, 'approve_button') and self.approve_button:
                        self.approve_button.config(state=tk.NORMAL)
                    if hasattr(self, 'cancel_button') and self.cancel_button:
                        self.cancel_button.config(state=tk.NORMAL)
                
                # Send deny command to Arduino (keep gate locked)
                self.send_arduino_command("DENY")
                
                # Update gate status
                self.gate_status_label.config(text="🔒 Gate: Locked (Pending Decision)", fg='#f59e0b')
                
                # Get current student info to display HOLD status
                student_info = None
                current_rfid = None
                
                # CRITICAL: Check for approved violation re-entry scenario first
                if hasattr(self, 'approved_violation_student_info') and self.approved_violation_student_info:
                    student_info = self.approved_violation_student_info
                    current_rfid = self.approved_violation_rfid
                    print(f"✅ Using approved violation student info for HOLD status: {student_info.get('name', 'Unknown')}")
                elif hasattr(self, 'current_student_info_for_timer') and self.current_student_info_for_timer:
                    student_info = self.current_student_info_for_timer
                    current_rfid = student_info.get('rfid') or student_info.get('id')
                elif hasattr(self, 'complete_uniform_student_info') and self.complete_uniform_student_info:
                    student_info = self.complete_uniform_student_info
                    current_rfid = student_info.get('rfid')
                elif hasattr(self, 'incomplete_student_info_for_approve') and self.incomplete_student_info_for_approve:
                    student_info = self.incomplete_student_info_for_approve
                    current_rfid = student_info.get('rfid')
                
                if not current_rfid and hasattr(self, 'current_rfid_for_timer'):
                    current_rfid = self.current_rfid_for_timer
                
                if not student_info and current_rfid:
                    student_info = self.get_student_info_by_rfid(current_rfid)
                
                # CRITICAL: Ensure Student Number is present (not RFID) for HOLD status display
                # Use the same logic as incomplete uniform to get student number from Firebase
                if student_info and current_rfid:
                    # Ensure student number is fetched and validated from Firebase
                    student_info = self._ensure_student_number_from_firebase(student_info, current_rfid)
                    
                    # Get Student Number - should now be guaranteed from validation above
                    student_number = (student_info.get('student_id') or 
                                    student_info.get('student_number') or 
                                    student_info.get('Student Number'))
                    
                    # Validate: Student Number should not be RFID, 'Unknown', None, or empty
                    is_valid = (student_number and 
                               student_number != 'Unknown' and 
                               student_number != 'Student Number Not Found' and
                               str(student_number).strip() != '' and
                               str(student_number).strip() != str(current_rfid).strip())
                    
                    if not is_valid:
                        # Last resort: fetch directly from Firebase
                        print(f"⚠️ Student Number still invalid in HOLD status display, fetching directly from Firebase for RFID: {current_rfid}")
                        fresh_student_info = self.get_student_info_by_rfid(current_rfid)
                        if fresh_student_info:
                            student_number = fresh_student_info.get('student_id') or fresh_student_info.get('student_number')
                            # Final check: make sure it's valid (not None, not empty, not RFID)
                            if student_number and str(student_number).strip() != '' and str(student_number).strip() != str(current_rfid).strip():
                                print(f"✅ Fetched Student Number directly from Firebase for HOLD status: {student_number}")
                                # Update student_info with the correct Student Number
                                student_info['student_id'] = student_number
                                student_info['student_number'] = student_number
                            else:
                                print(f"⚠️ WARNING: Fetched Student Number from Firebase is invalid (None, empty, or equals RFID): {student_number}")
                                student_number = 'Student Number Not Found'
                        else:
                            print(f"⚠️ WARNING: Could not fetch Student Number from Firebase for HOLD status")
                            student_number = 'Student Number Not Found'
                    else:
                        print(f"✅ Student Number validated for HOLD status: {student_number}")
                    
                    # Strip whitespace from student number if it exists
                    if student_number and student_number != 'Student Number Not Found':
                        student_number = str(student_number).strip()
                else:
                    student_number = 'Student Number Not Found'
                
                # Update main screen with HOLD status (keep student info visible)
                if student_info and self.main_screen_window and self.main_screen_window.winfo_exists():
                    from datetime import datetime
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Strip whitespace from course
                    course_val = student_info.get('course', 'Unknown Course')
                    if course_val:
                        course_val = str(course_val).strip()
                    
                    person_info = {
                        'id': student_number,  # Use Student Number, not RFID
                        'student_id': student_number,  # Student Number from Firebase
                        'student_number': student_number,  # Alternative field name
                        'rfid': current_rfid,  # Keep RFID for reference (for image loading)
                        'name': student_info.get('name', 'Unknown Student'),
                        'type': 'student',
                        'course': course_val,
                        'gender': student_info.get('gender', 'Unknown'),
                        'timestamp': current_time,
                        'status': 'HOLD',  # Display HOLD status
                        'action': 'HOLD',
                        'guard_id': self.current_guard_id or 'Unknown'
                    }
                    self.update_main_screen_with_person(person_info)
                    print(f"✅ Main screen updated with HOLD status for {student_info.get('name', 'Unknown')} - Student Number: {student_number}")
                
                # Keep requirements section visible (don't hide it)
                print(f"✅ Requirements section remains visible - waiting for guard decision")
                
                # Show info message
                self.show_green_success_message("Pending Guard Decision", 
                                  "Gate is locked - waiting for guard decision\n"
                                  "✅ Detection stopped\n"
                                  "✅ Checkboxes are now editable\n"
                                  "👤 Review uniform requirements\n"
                                  "🔘 Click GRANTED or ACCESS GRANTED to approve\n"
                                  "❌ Click CANCEL to lock and return to standby")
                
            else:
                # Already toggled - should not happen, but handle gracefully
                print("⚠️ DENIED button already toggled to GRANTED")
                
        except Exception as e:
            print(f"ERROR: Error handling interface deny: {e}")
            self.add_activity_log(f"Error handling interface deny: {e}")
            import traceback
            traceback.print_exc()
    
    def handle_interface_granted_after_deny(self):
        """Handle GRANTED button (formerly DENIED) - entry based on original detection state"""
        try:
            print("SUCCESS: GRANTED button pressed (after DENIED) - Processing entry based on original detection")
            self.add_activity_log("GRANTED button pressed (after DENIED) - Processing entry")
            
            # CRITICAL: Check if this is approved violation re-entry scenario
            is_approved_violation_scenario = (
                hasattr(self, 'approved_violation_student_info') and 
                self.approved_violation_student_info is not None
            )
            
            # Get student info
            student_info = None
            current_rfid = None
            
            # Check for approved violation scenario first
            if is_approved_violation_scenario:
                student_info = self.approved_violation_student_info
                current_rfid = self.approved_violation_rfid
                print(f"✅ Using approved violation student info for GRANTED: {student_info.get('name', 'Unknown')}")
            # Check for complete uniform scenario
            elif hasattr(self, 'complete_uniform_student_info') and self.complete_uniform_student_info:
                student_info = self.complete_uniform_student_info
                current_rfid = student_info.get('rfid')
            
            # Fallback: incomplete uniform scenario
            if not student_info:
                if hasattr(self, 'incomplete_student_info_for_approve') and self.incomplete_student_info_for_approve:
                    student_info = self.incomplete_student_info_for_approve
                    current_rfid = student_info.get('rfid')
            
            # Fallback: Get current student info
            if not current_rfid:
                if hasattr(self, 'detection_system') and self.detection_system:
                    detection_service = getattr(self.detection_system, 'detection_service', None)
                    if detection_service:
                        current_rfid = getattr(detection_service, 'current_rfid', None)
            
            if not current_rfid and hasattr(self, 'current_rfid_for_timer'):
                current_rfid = self.current_rfid_for_timer
            
            if not student_info and current_rfid:
                student_info = self.get_student_info_by_rfid(current_rfid)
            
            if not current_rfid or not student_info:
                print(f"⚠️ WARNING: Cannot process GRANTED - missing RFID or student info")
                messagebox.showerror("Error", "Cannot process granted - missing student information")
                return
            
            # CRITICAL: Reset detection flags BEFORE recording entry
            # This ensures update_main_screen_with_person shows ENTRY status, not DETECTING
            # record_complete_uniform_entry calls update_main_screen_with_permanent_student
            # which calls update_main_screen_with_person, and that function checks detection_active flag
            try:
                # Reset detection flags FIRST (before recording entry and updating main screen)
                self.detection_active = False
                self.uniform_detection_complete = True  # Mark as complete since granted
                print(f"✅ Detection flags reset BEFORE recording entry: detection_active=False, uniform_detection_complete=True")
                
                # Clear detection session tracking
                if current_rfid in self.active_detection_sessions:
                    del self.active_detection_sessions[current_rfid]
                    print(f"✅ Cleared detection session for RFID {current_rfid}")
                
                # Stop any running detection
                if hasattr(self, 'detection_system') and self.detection_system:
                    self.detection_system.stop_detection()
                    print(f"✅ Stopped detection system after granted approval")
            except Exception as e:
                print(f"⚠️ WARNING: Could not reset detection flags: {e}")
            
            # Handle approved violation re-entry scenario
            if is_approved_violation_scenario:
                # Approved violation re-entry: Entry without new violation (violation already stored in Firebase)
                print(f"✅ GRANTED button clicked - recording entry WITHOUT new violation (approved violation re-entry)")
                self.add_activity_log("GRANTED button pressed - Entry without new violation (approved violation re-entry)")
                
                # Record entry without violation (re-entry, violation already stored)
                self.record_complete_uniform_entry(current_rfid, student_info)
                
                # Open gate
                self.open_gate()
                
                # Update gate status
                self.gate_status_label.config(text="🔓 Gate: Unlocked", fg='#10b981')
                
                # Show success message
                self.show_green_success_message("Entry Approved (No New Violation)", 
                                  "SUCCESS: Entry recorded WITHOUT new violation\n"
                                  "🚪 Gate is opening\n"
                                  "👤 Student may proceed\n"
                                  "✅ Re-entry allowed (violation already stored)")
                
                # Reset GRANTED button back to DENIED
                if hasattr(self, 'deny_button') and self.deny_button:
                    try:
                        self.deny_button.config(
                            text="DENIED",
                            command=self.handle_interface_deny,
                            state=tk.DISABLED
                        )
                        print(f"✅ GRANTED button reset to DENIED (approved violation re-entry granted)")
                    except Exception as e:
                        print(f"⚠️ WARNING: Could not reset GRANTED button: {e}")
                
                # Disable all buttons
                if hasattr(self, 'approve_button') and self.approve_button:
                    self.approve_button.config(state=tk.DISABLED)
                if hasattr(self, 'cancel_button') and self.cancel_button:
                    self.cancel_button.config(state=tk.DISABLED)
                
                # Reset denied button state flags
                self.denied_button_clicked = False
                self.original_detection_state = None
                self.original_uniform_status = None
                
                # Clear approved violation info
                self.approved_violation_student_info = None
                self.approved_violation_rfid = None
                
                # Make checkboxes read-only again
                self._make_requirement_checkboxes_editable(False)
                
                # Hide requirements section
                self.hide_requirements_section()
                
                # Schedule 15-second timer to clear main screen
                self.clear_main_screen_after_gate_action()
                
                # Schedule status update after unlock duration
                self.root.after(3500, self.update_gate_status_locked)
                
                return
            
            # Normal scenario: GRANTED button (formerly DENIED) always records entry WITHOUT violation
            # Regardless of original detection state, GRANTED = entry without violation
            print(f"✅ GRANTED button clicked - recording entry WITHOUT violation (bypassing incomplete uniform)")
            # This will update the main screen with ENTRY status (flags are already set above)
            self.record_complete_uniform_entry(current_rfid, student_info)
            
            # Open gate
            self.open_gate()
            
            # Update gate status
            self.gate_status_label.config(text="🔓 Gate: Unlocked", fg='#10b981')
            
            # Show success message - GRANTED always means no violation
            self.show_green_success_message("Entry Approved (No Violation)", 
                              "SUCCESS: Entry recorded WITHOUT violation\n"
                              "🚪 Gate is opening\n"
                              "👤 Student may proceed\n"
                              "✅ Uniform violation bypassed")
            
            # NOTE: record_complete_uniform_entry already updates main screen with entry status
            # Don't overwrite it with gate status - entry status is more important
            
            # Cancel auto-entry timer if exists
            if hasattr(self, 'complete_uniform_auto_entry_timer') and self.complete_uniform_auto_entry_timer:
                try:
                    self.root.after_cancel(self.complete_uniform_auto_entry_timer)
                    self.complete_uniform_auto_entry_timer = None
                except Exception:
                    pass
            
            # Reset all button states (ACCESS GRANTED and CANCEL disabled, DENIED enabled)
            self._reset_denied_button_state()
            
            # Reset pending operations
            self.complete_uniform_student_info = None
            
            # Make checkboxes read-only again
            self._make_requirement_checkboxes_editable(False)
            
            # Hide requirements section
            self.hide_requirements_section()
            
            # Clear main screen after gate control action
            # This will schedule a 15-second timer to clear the screen (entry status will be visible for 15 seconds)
            self.clear_main_screen_after_gate_action()
            
            # Schedule status update after unlock duration
            self.root.after(3500, self.update_gate_status_locked)
            
        except Exception as e:
            print(f"ERROR: Error handling interface granted after deny: {e}")
            self.add_activity_log(f"Error handling interface granted after deny: {e}")
            import traceback
            traceback.print_exc()
    
    def _reset_denied_button_state(self):
        """Reset DENIED button to original state and disable all gate control buttons"""
        try:
            self.denied_button_clicked = False
            self.original_detection_state = None
            self.original_uniform_status = None
            
            # Reset DENIED button text and command, but DISABLE it (will be enabled only when new RFID is tapped)
            if hasattr(self, 'deny_button') and self.deny_button:
                self.deny_button.config(
                    text="DENIED",
                    command=self.handle_interface_deny,
                    state=tk.DISABLED  # Disabled - will be enabled only when new RFID is tapped
                )
                print(f"✅ DENIED button reset to original state (disabled - waiting for new RFID tap)")
            
            # Disable ACCESS GRANTED button
            if hasattr(self, 'approve_button') and self.approve_button:
                self.approve_button.config(state=tk.DISABLED)
                print(f"✅ ACCESS GRANTED button disabled")
            
            # Disable CANCEL button
            if hasattr(self, 'cancel_button') and self.cancel_button:
                self.cancel_button.config(state=tk.DISABLED)
                print(f"✅ CANCEL button disabled")
        except Exception as e:
            print(f"⚠️ Error resetting denied button state: {e}")
    
    def update_gate_status_locked(self):
        """Update gate status to locked after auto-lock"""
        try:
            self.gate_status_label.config(text="🔒 Gate: Locked", fg='#dc2626')
            self.add_activity_log("Gate auto-locked after 3 seconds")
        except Exception as e:
            print(f"ERROR: Error updating gate status: {e}")
    
    def update_arduino_connection_status(self, connected):
        """Update Arduino connection status display"""
        try:
            # Check if the dashboard has been created and arduino_status_label exists
            if hasattr(self, 'arduino_status_label') and self.arduino_status_label and hasattr(self.arduino_status_label, 'config'):
                if connected:
                    self.arduino_status_label.config(text="🔌 Arduino: Connected", fg='#10b981')
                else:
                    self.arduino_status_label.config(text="🔌 Arduino: Disconnected", fg='#dc2626')
            else:
                # Dashboard not created yet, just log the status
                status = "Connected" if connected else "Disconnected"
                print(f"🔌 Arduino: {status} (UI update pending)")
        except Exception as e:
            print(f"ERROR: Error updating Arduino status: {e}")
    
    def create_activity_logs_section(self, parent):
        """Create activity logs section"""
        logs_frame = tk.LabelFrame(
            parent,
            text="Activity Log",
            font=('Arial', 14, 'bold'),
            fg='#374151',
            bg='#ffffff',
            relief='groove',
            bd=2
        )
        logs_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Logs text widget - increased height to show more activity history
        self.logs_text = tk.Text(
            logs_frame,
            height=15,
            font=('Consolas', 10),
            bg='#f8fafc',
            fg='#374151',
            relief='flat',
            bd=1,
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        self.logs_text.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Add initial message
        self.add_activity_log("System initialized successfully")
        self.add_activity_log("Ready for guard operations")
    
    def set_person_type(self, person_type):
        """Set the selected person type"""
        self.person_type_var.set(person_type)
        
        # Reset all buttons
        self.student_btn.config(bg='#dbeafe', fg='#1e3a8a')
        self.teacher_btn.config(bg='#dbeafe', fg='#1e3a8a')
        self.visitor_btn.config(bg='#dbeafe', fg='#1e3a8a')
        
        # Highlight selected button
        if person_type == "student":
            self.student_btn.config(bg='#1e3a8a', fg='white')
        elif person_type == "teacher":
            self.teacher_btn.config(bg='#1e3a8a', fg='white')
        elif person_type == "visitor":
            self.visitor_btn.config(bg='#1e3a8a', fg='white')
    
    def log_person_entry(self):
        """Log person entry and start detection if needed"""
        person_id = self.person_id_var.get().strip().upper()
        person_type = self.person_type_var.get()
        
        if not person_id:
            messagebox.showwarning("Warning", "Please enter a Person ID")
            return
        
        # Ensure Firebase is ready before processing
        if not self.firebase_initialized:
            print("INFO: Firebase not ready, attempting to initialize...")
            # Show status message to user
            self.add_activity_log("INFO: Initializing Firebase connection...")
            
            # Check if we've already tried Firebase initialization multiple times
            if not hasattr(self, 'firebase_init_attempts'):
                self.firebase_init_attempts = 0
            
            self.firebase_init_attempts += 1
            
            # If we've tried too many times, go directly to offline mode
            if self.firebase_init_attempts > 2:
                print("WARNING: Firebase initialization failed multiple times - switching to offline mode")
                self.add_activity_log("WARNING: Firebase initialization failed multiple times - switching to offline mode")
                self.firebase_initialized = False
                self.root.after(1000, lambda: self.process_person_entry_offline())
                return
            
            # Try to initialize Firebase with better error handling
            try:
                success = self.init_firebase()
                time.sleep(1)  # Shorter wait time
                
                if not success:
                    print("WARNING: Firebase initialization failed, trying alternative method...")
                    self.add_activity_log("WARNING: Firebase initialization failed, trying alternative method...")
                    self.init_firebase_async()
                    time.sleep(2)  # Shorter wait time
                
                # Try again if still not initialized
                if not self.firebase_initialized:
                    print("WARNING: Firebase still not ready, retrying in 3 seconds...")
                    self.add_activity_log("WARNING: Firebase initialization delayed, retrying...")
                    self.root.after(3000, lambda: self.retry_person_entry())
                    return
                else:
                    print("SUCCESS: Firebase initialized successfully on retry")
                    self.add_activity_log("SUCCESS: Firebase initialized successfully on retry")
            except Exception as e:
                print(f"ERROR: Firebase initialization failed: {e}")
                self.add_activity_log(f"ERROR: Firebase initialization failed: {e}")
                print("WARNING: Switching to offline mode...")
                self.add_activity_log("WARNING: Switching to offline mode...")
                self.firebase_initialized = False
                self.root.after(1000, lambda: self.process_person_entry_offline())
            return
        
        # Recognize visitor RFID taps even if the guard selected the wrong person_type
        if (person_id in getattr(self, 'visitor_rfid_registry', {})) or (hasattr(self, 'visitor_rfid_assignments') and person_id in self.visitor_rfid_assignments):
            try:
                self.handle_visitor_rfid_tap(person_id)
            except Exception as e:
                print(f"ERROR: Failed to handle visitor RFID tap for {person_id}: {e}")
            self.person_id_var.set("")  # Clear input
            return
        
        # Recognize student temporary RFID taps regardless of selected type
        if hasattr(self, 'student_rfid_assignments') and person_id in self.student_rfid_assignments:
            try:
                self.handle_student_forgot_id_rfid_tap(person_id)
            except Exception as e:
                print(f"ERROR: Failed to handle student forgot-ID RFID tap for {person_id}: {e}")
            self.person_id_var.set("")  # Clear input
            return
        
        # Check if this is a permanent student RFID tap
        print(f"🔍 DEBUG log_person_entry: person_type = '{person_type}', person_id = '{person_id}'")
        is_permanent = self.is_permanent_student_rfid(person_id)
        print(f"🔍 DEBUG log_person_entry: is_permanent_student_rfid({person_id}) = {is_permanent}")
        
        # Fallback: If RFID check failed but student exists in Firebase (as student type), still treat as permanent
        if person_type == "student" and not is_permanent:
            # Try to verify student exists in Firebase
            try:
                student_info = self.get_student_info_by_rfid(person_id)
                if student_info:
                    print(f"🔍 DEBUG log_person_entry: Student found in Firebase but RFID check failed - treating as permanent student")
                    is_permanent = True
            except Exception as e:
                print(f"⚠️ DEBUG log_person_entry: Could not verify student in Firebase: {e}")
        
        if person_type == "student" and is_permanent:
            print(f"🔍 DEBUG log_person_entry: Calling handle_permanent_student_rfid_tap({person_id})")
            self.handle_permanent_student_rfid_tap(person_id)
            self.person_id_var.set("")  # Clear input
            return
        else:
            print(f"🔍 DEBUG log_person_entry: NOT calling handle_permanent_student_rfid_tap - person_type='{person_type}', is_permanent={is_permanent}")
        
        # Check if this person is already detected (for exit) - MUST CHECK BEFORE LOGGING ENTRY
        try:
            detected_in_main = hasattr(self, 'current_person_id') and self.current_person_id == person_id and bool(getattr(self, 'detection_active', False))
            detected_in_system = False
            if hasattr(self, 'detection_system') and self.detection_system:
                detected_in_system = getattr(self.detection_system, 'current_person_id', None) == person_id and getattr(self.detection_system, 'detection_active', False)

            # Also consult the RFID status tracker if available (explicit toggle)
            rfid_toggled_as_entry = False
            if hasattr(self, 'rfid_status_tracker'):
                rfid_toggled_as_entry = self.rfid_status_tracker.get(person_id, '') == 'ENTRY'

            if detected_in_main or detected_in_system or rfid_toggled_as_entry:
                # Person is detected (double-tap) - check if they have entry record before treating as exit
                has_entry_record = self.check_person_has_entry_record(person_id, person_type)
                
                if not has_entry_record:
                    # No entry record - person never completed entry, don't treat as exit
                    if person_type.lower() == 'student':
                        # For students: restart detection (retry) instead of exit
                        print(f"🔄 Student {person_id} tapped again but has no entry record - restarting detection")
                        try:
                            self.stop_detection()
                        except Exception:
                            pass
                        
                        # Check if this is a permanent student - use appropriate handler
                        if self.is_permanent_student_rfid(person_id):
                            # For permanent students, use the proper handler which will restart detection
                            student_info = self.get_student_info_by_rfid(person_id)
                            if student_info:
                                self.handle_permanent_student_timein(person_id, student_info)
                                self.add_activity_log(f"Restarting detection for {student_info.get('name', 'Unknown')} (student) - no entry record found")
                            else:
                                print(f"⚠️ Could not get student info for {person_id}")
                                self.add_activity_log(f"Failed to restart detection for {person_id}: student info not found")
                        else:
                            # For non-permanent students, use start_person_detection
                            person_name = self.get_person_name(person_id, person_type)
                            try:
                                self.start_person_detection(person_id, person_name, person_type)
                                self.add_activity_log(f"Restarting detection for {person_name} ({person_type}) - no entry record found")
                            except Exception as e:
                                print(f"⚠️ Failed to restart detection: {e}")
                                self.add_activity_log(f"Failed to restart detection for {person_id}: {e}")
                    else:
                        # For non-students: no entry record means they're outside - start entry normally
                        print(f"ℹ️ {person_type} {person_id} has no entry record - treating as new entry")
                        # Don't return - continue with normal entry processing below
                        # Reset detection state so entry can be processed
                        try:
                            self.stop_detection()
                        except Exception:
                            pass
                        # Clear input and continue to entry processing
                        self.person_id_var.set("")
                        # Don't return - fall through to entry processing below
                else:
                    # Person has entry record - this is a valid exit
                    print(f"✅ {person_type} {person_id} has entry record - processing exit")
                    try:
                        self.stop_detection()
                    except Exception:
                        pass
                    self.add_activity_log(f"Person Exit: {person_id} ({person_type})")
                    
                    # Save teacher time-out to Firebase
                    if person_type.lower() == 'teacher':
                        try:
                            current_time = self.get_current_timestamp()
                            self.save_teacher_activity_to_firebase(person_id, 'time_out', current_time)
                            self.add_activity_log(f"Teacher time-out saved: {person_id}")
                        except Exception as e:
                            print(f"WARNING: Failed to save teacher time-out: {e}")

                    # Update main screen with exit status
                    if self.main_screen_window and self.main_screen_window.winfo_exists():
                        self.update_main_screen_with_exit(person_id, person_type)

                    # Ensure the RFID toggle state is reset to EXIT
                    try:
                        if not hasattr(self, 'rfid_status_tracker'):
                            self.rfid_status_tracker = {}
                        self.rfid_status_tracker[person_id] = 'EXIT'
                    except Exception:
                        pass

                    self.person_id_var.set("")  # Clear input
                    return
        except Exception:
            # If any check fails, continue with normal processing
            pass
        
        # This is an ENTRY - DO NOT log here, entry will only be logged after uniform completion
        # (Entry is logged in record_complete_uniform_entry() after successful uniform detection)
        
        # Get person information for main screen display
        if person_type.lower() == 'student':
            # For students, try to get info by RFID first, then by student ID
            student_info = self.get_student_info_by_rfid(person_id)
            if not student_info:
                # Try get_student_info but avoid the offline fallback that creates "Student {person_id}"
                try:
                    if self.firebase_initialized and self.db:
                        # Query Firebase directly to avoid offline fallback
                        doc_ref = self.db.collection('students').document(person_id)
                        doc = doc_ref.get()
                        
                        if doc.exists:
                            data = doc.to_dict()
                            student_info = {
                                'name': data.get('Name', 'Unknown'),
                                'course': data.get('Course', data.get('Department', 'Unknown')),
                                'gender': data.get('Gender', 'Unknown')
                            }
                except Exception as e:
                    print(f"WARNING: Firebase query failed in fallback: {e}")
            
            # If no student found and the input looks like an RFID, treat it as unassigned
            if not student_info:
                is_rfid_like = person_id.isdigit() and len(person_id) >= 8
                if is_rfid_like:
                    # Unknown/unassigned RFID tapped - do not start detection
                    self.add_activity_log(f"Unassigned RFID tapped: {person_id}")
                    messagebox.showwarning("Unassigned RFID", f"RFID {person_id} is not assigned to any student.\nPlease assign a temporary RFID first or verify student number.")
                    # Clear input and return without starting detection
                    self.person_id_var.set("")
                    return

            if student_info:
                from datetime import datetime
                person_info = {
                    'id': person_id,
                    'name': student_info.get('name', 'Unknown Student'),
                    'type': 'student',
                    'course': student_info.get('course', 'Unknown'),
                    'gender': student_info.get('gender', 'Unknown'),
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'status': 'DETECTING',  # Changed from 'TIME-IN' - detection hasn't completed yet
                    'guard_id': self.current_guard_id or 'Unknown'
                }
            else:
                # Only use fallback if we really can't find the student
                fallback_info = self.get_person_info_for_main_screen(person_id, person_type)
                if fallback_info:
                    # Override status to DETECTING for students starting detection
                    if fallback_info.get('type', '').lower() == 'student':
                        fallback_info['status'] = 'DETECTING'
                    person_info = fallback_info
                else:
                    person_info = {
                        'id': person_id,
                        'name': 'Unknown Student',
                        'type': 'student',
                        'status': 'DETECTING',
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
        else:
            print("WARNING: No student info found, using fallback")
            fallback_info = self.get_person_info_for_main_screen(person_id, person_type)
            if fallback_info:
                # Override status to DETECTING for students starting detection
                if fallback_info.get('type', '').lower() == 'student':
                    fallback_info['status'] = 'DETECTING'
                person_info = fallback_info
            else:
                person_info = {
                    'id': person_id,
                    'name': 'Unknown',
                    'type': person_type,
                    'status': 'DETECTING',
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
        
        # DO NOT call get_person_info_for_main_screen again - it overwrites status!
        # person_info now has correct status 'DETECTING' for students
        
        # Set detection_active BEFORE updating main screen
        if person_type.lower() == 'student':
            self.detection_active = True
            self.uniform_detection_complete = False
        
        # Update main screen with person information
        if self.main_screen_window and self.main_screen_window.winfo_exists():
            self.update_main_screen_with_person(person_info)
        
        # Clear the input
        self.person_id_var.set("")
        
        # Start detection for students and teachers, but only for ENTRY actions
        try:
            # Determine canonical action for this person_info (if available)
            action = None
            if 'person_info' in locals() and person_info:
                action = person_info.get('action')
                if not action:
                    status = str(person_info.get('status', '')).upper()
                    if status in ('TIME-IN', 'ENTRY', 'COMPLETE UNIFORM', 'DETECTING'):
                        action = 'ENTRY'
                    elif status in ('TIME-OUT', 'EXIT', 'EXITED'):
                        action = 'EXIT'
                    else:
                        action = person_info.get('status', 'Unknown')

            # Default action to ENTRY if not determinable (preserve existing behaviour)
            if not action:
                action = 'ENTRY'

            if person_type in ["student", "teacher"] and str(action).upper() == 'ENTRY':
                try:
                    person_name = self.get_person_name(person_id, person_type)
                    # Mark this RFID/person as having timed-in (awaiting second tap for exit)
                    try:
                        if not hasattr(self, 'rfid_status_tracker'):
                            self.rfid_status_tracker = {}
                        # Only set to ENTRY if we are logging an entry action
                        self.rfid_status_tracker[person_id] = 'ENTRY'
                    except Exception:
                        pass
                    
                    # Save teacher time-in to Firebase (before detection starts)
                    if person_type.lower() == 'teacher':
                        try:
                            current_time = self.get_current_timestamp()
                            self.save_teacher_activity_to_firebase(person_id, 'time_in', current_time)
                            self.add_activity_log(f"Teacher time-in saved: {person_id}")
                        except Exception as e:
                            print(f"WARNING: Failed to save teacher time-in: {e}")
                    
                    # CRITICAL: Check Event Mode before starting detection
                    if getattr(self, 'event_mode_active', False):
                        print(f"🛑 Event Mode active - NOT starting detection in log_person_entry for {person_name}")
                        self.add_activity_log(f"Entry logged (Event Mode): {person_name} ({person_type})")
                        messagebox.showinfo("Success", f"Person entry logged (Event Mode): {person_id} ({person_type})")
                    else:
                        self.start_person_detection(person_id, person_name, person_type)
                        self.add_activity_log(f"Detection started for {person_name} ({person_type})")
                        messagebox.showinfo("Success", f"Person entry logged and detection started: {person_id} ({person_type})")
                except Exception as e:
                    self.add_activity_log(f"Failed to start detection: {str(e)}")
                    messagebox.showwarning("Detection Error", f"Person logged but detection failed: {str(e)}")
            else:
                # Do not start detection on EXIT or unknown actions
                self.add_activity_log(f"Skipping detection start for {person_id} ({person_type}) - action={action}")
                
                # Save teacher time-in to Firebase even if detection is skipped
                if person_type.lower() == 'teacher' and str(action).upper() == 'ENTRY':
                    try:
                        current_time = self.get_current_timestamp()
                        self.save_teacher_activity_to_firebase(person_id, 'time_in', current_time)
                        self.add_activity_log(f"Teacher time-in saved: {person_id}")
                    except Exception as e:
                        print(f"WARNING: Failed to save teacher time-in: {e}")
                
                messagebox.showinfo("Success", f"Person entry logged: {person_id} ({person_type})")
        except Exception as e:
            # Fallback behaviour: attempt to start detection (keeps old behaviour if something goes wrong)
            try:
                if person_type in ["student", "teacher"]:
                    person_name = self.get_person_name(person_id, person_type)
                    self.start_person_detection(person_id, person_name, person_type)
                    self.add_activity_log(f"Detection started for {person_name} ({person_type})")
                    messagebox.showinfo("Success", f"Person entry logged and detection started: {person_id} ({person_type})")
                else:
                    messagebox.showinfo("Success", f"Person entry logged: {person_id} ({person_type})")
            except Exception as e2:
                self.add_activity_log(f"Failed to start detection in fallback: {str(e2)}")
                messagebox.showwarning("Detection Error", f"Person logged but detection failed: {str(e2)}")
    
    def get_person_info_for_main_screen(self, person_id, person_type):
        """Get person information formatted for main screen display"""
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            if person_type.lower() == 'student':
                # Get student information
                student_info = self.get_student_info(person_id)
                return {
                    'id': person_id,
                    'name': student_info.get('name', 'Unknown Student'),
                    'type': 'student',
                    'course': student_info.get('course', 'Unknown'),
                    'gender': student_info.get('gender', 'Unknown'),
                    'timestamp': current_time,
                    'status': 'TIME-IN',
                    'guard_id': self.current_guard_id or 'Unknown'
                }
            elif person_type.lower() == 'teacher':
                return {
                    'id': person_id,
                    'name': f"Teacher {person_id}",
                    'type': 'teacher',
                    'course': 'Faculty',
                    'gender': 'N/A',
                    'timestamp': current_time,
                    'status': 'TIME-IN',
                    'guard_id': self.current_guard_id or 'Unknown'
                }
            elif person_type.lower() == 'visitor':
                return {
                    'id': person_id,
                    'name': f"Visitor {person_id}",
                    'type': 'visitor',
                    'course': 'N/A',
                    'gender': 'N/A',
                    'timestamp': current_time,
                    'status': 'TIME-IN',
                    'guard_id': self.current_guard_id or 'Unknown'
                }
            else:
                return {
                    'id': person_id,
                    'name': f"Person {person_id}",
                    'type': person_type,
                    'course': 'N/A',
                    'gender': 'N/A',
                    'timestamp': current_time,
                    'status': 'TIME-IN',
                    'guard_id': self.current_guard_id or 'Unknown'
                }
        except Exception as e:
            print(f"ERROR: Error getting person info: {e}")
            return {
                'id': person_id,
                    'name': 'Unknown',
                'type': person_type,
                'course': 'N/A',
                'gender': 'N/A',
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'status': 'TIME-IN',
                'guard_id': 'Unknown'
            }
    
    def update_main_screen_with_person(self, person_info):
        """Update main screen with person information"""
        try:
            if not self.main_screen_window or not self.main_screen_window.winfo_exists():
                print("WARNING: Main screen window not available")
                return
            
            # In Event Mode, ALLOW showing student info (but no detection)
            # Event Mode just bypasses detection, but we still show student info
            if getattr(self, 'event_mode_active', False):
                print(f"✅ Event Mode active - Showing student info (detection bypassed)")
                # Continue to show student info in Event Mode (don't return early)
            
            # Clear existing content
            for widget in self.person_display_frame.winfo_children():
                widget.destroy()
            
            # Create top container (75% height) - Picture and Information panels
            top_container = tk.Frame(self.person_display_frame, bg='#ffffff')
            top_container.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
            
            # Status indicator - For students, show DETECTING during detection, ENTRY only after complete uniform
            status_text = person_info.get('status', 'DETECTING')
            status_color = '#ef4444'  # Default red
            
            # Check for HOLD status first (when DENIED is clicked during detection) - takes priority
            is_hold_status = (person_info.get('status', '').upper() == 'HOLD' or 
                            person_info.get('action', '').upper() == 'HOLD')
            
            if person_info.get('type', '').lower() == 'student':
                # Check if this is temporary RFID (forgot ID) - skip detection, show ENTRY immediately
                is_temporary_rfid = person_info.get('temporary_rfid', False) or person_info.get('skip_detection', False)
                
                # Check if this is Event Mode entry - if so, use the status from person_info
                is_event_mode = person_info.get('event_mode', False) or getattr(self, 'event_mode_active', False)
                
                if is_temporary_rfid:
                    # Temporary RFID (forgot ID): Show ENTRY or EXIT based on status/action
                    if person_info.get('status') == 'TIME-OUT' or person_info.get('action') == 'EXIT' or person_info.get('status') == 'EXITED':
                        status_text = "EXIT"
                        status_color = '#ef4444'  # Red for exit
                        print(f"✅ Temporary RFID detected - showing EXIT (detection skipped)")
                    elif person_info.get('status') == 'TIME-IN' or person_info.get('action') == 'ENTRY':
                        status_text = "ENTRY"
                        status_color = '#10b981'  # Green
                        print(f"✅ Temporary RFID detected - showing ENTRY (detection skipped)")
                    else:
                        # Default to ENTRY for temporary RFID if status unclear
                        status_text = "ENTRY"
                        status_color = '#10b981'  # Green
                        print(f"✅ Temporary RFID detected - showing ENTRY (default, detection skipped)")
                elif is_event_mode:
                    # Event Mode: Use status from person_info (should be "TIME-IN (EVENT MODE)" or "TIME-OUT (EVENT MODE)")
                    status_from_info = person_info.get('status', 'ENTRY')
                    if 'EVENT MODE' in status_from_info:
                        # Extract just the action part for display
                        if 'TIME-IN' in status_from_info:
                            status_text = "ENTRY"
                            status_color = '#10b981'  # Green
                        elif 'TIME-OUT' in status_from_info:
                            status_text = "EXIT"
                            status_color = '#ef4444'  # Red
                        else:
                            status_text = "ENTRY"
                            status_color = '#10b981'  # Green
                    else:
                        status_text = "ENTRY"
                        status_color = '#10b981'  # Green
                else:
                    # Normal mode: Check if detection is active and uniform is not complete
                    detection_active = getattr(self, 'detection_active', False)
                    uniform_complete = getattr(self, 'uniform_detection_complete', False)
                    
                    if person_info.get('status') == 'INCOMPLETE UNIFORM':
                        # Incomplete uniform - show INCOMPLETE UNIFORM
                        status_text = "INCOMPLETE UNIFORM"
                        status_color = '#ef4444'  # Red for incomplete
                    elif person_info.get('status') == 'TIME-IN':
                        # TIME-IN status - prioritize showing ENTRY if uniform is complete or detection is not active
                        # This handles the case when cancel is clicked and entry is recorded
                        if uniform_complete or not detection_active:
                            # Entry was recorded (uniform complete or detection stopped) - show ENTRY
                            status_text = "ENTRY"
                            status_color = '#10b981'  # Green
                        elif person_info.get('action') == 'ENTRY' or person_info.get('skip_detection', False):
                            # Explicit ENTRY action - show ENTRY (not DETECTING)
                            status_text = "ENTRY"
                            status_color = '#10b981'  # Green
                        else:
                            # Detection still active - show DETECTING
                            status_text = "DETECTING"
                            status_color = '#f59e0b'  # Orange/amber
                    elif detection_active and not uniform_complete:
                        # Detection in progress - show DETECTING
                        status_text = "DETECTING"
                        status_color = '#f59e0b'  # Orange/amber
                    elif person_info.get('status') == 'COMPLETE UNIFORM' or uniform_complete:
                        # Complete uniform - show ENTRY
                        status_text = "ENTRY"
                        status_color = '#10b981'  # Green
                    else:
                        # Default based on status
                        if person_info.get('status') == 'TIME-OUT' or person_info.get('status') == 'EXIT':
                            status_text = person_info.get('status', 'DETECTING')
                            status_color = '#ef4444'  # Red for exit
                        else:
                            status_text = "DETECTING"
                            status_color = '#f59e0b'  # Orange/amber
                
                # Override with HOLD status if applicable (takes priority over all other statuses for students)
                if is_hold_status:
                    status_text = "HOLD"
                    status_color = '#f59e0b'  # Orange/amber for hold status
                    print(f"✅ HOLD status detected - displaying HOLD status (priority override)")
            else:
                # For non-students (visitors/teachers), use original logic
                if person_info.get('status') == 'TIME-IN' or person_info.get('status') == 'ENTRY':
                    status_text = person_info.get('status', 'DETECTING')
                    status_color = '#10b981'  # Green
                elif person_info.get('status') == 'TIME-OUT' or person_info.get('status') == 'EXIT':
                    status_text = person_info.get('status', 'EXIT')
                    status_color = '#ef4444'  # Red
                else:
                    status_text = person_info.get('status', 'DETECTING')
                    status_color = '#f59e0b'  # Orange/amber
                
                # Override with HOLD status if applicable (for non-students too)
                if is_hold_status:
                    status_text = "HOLD"
                    status_color = '#f59e0b'  # Orange/amber for hold status
                    print(f"✅ HOLD status detected - displaying HOLD status (priority override)")
            
            # Create left panel (Picture) - 66% width
            picture_frame = tk.Frame(top_container, bg='#ffffff', relief='solid', bd=1)
            picture_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 5), pady=10)
            picture_frame.pack_propagate(False)
            
            # Load and get image path for dynamic resizing
            person_type = person_info.get('type', '').lower()
            rfid = person_info.get('rfid')
            image_path = None
            
            if rfid:
                # Determine folder based on person type
                if person_type == 'student':
                    image_folder = 'student_images'
                elif person_type == 'teacher':
                    image_folder = 'teacher_images'
                else:
                    image_folder = None
                
                if image_folder:
                    # Try different image formats
                    image_formats = ['.jpg', '.jpeg', '.png']
                    for ext in image_formats:
                        potential_path = os.path.join(image_folder, f"{rfid}{ext}")
                        if os.path.exists(potential_path):
                            image_path = potential_path
                            break
            
            if image_path and os.path.exists(image_path):
                # Use Canvas for better image scaling and filling
                image_canvas = tk.Canvas(
                    picture_frame,
                    bg='#ffffff',
                    highlightthickness=0
                )
                image_canvas.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
                
                # Store original image path for resizing
                image_canvas.image_path = image_path
                
                # Function to resize and center image when canvas is resized
                def resize_image(event=None):
                    canvas_width = image_canvas.winfo_width()
                    canvas_height = image_canvas.winfo_height()
                    if canvas_width > 1 and canvas_height > 1:  # Canvas must be visible
                        try:
                            image_canvas.delete("all")
                            # Load original image
                            original_img = Image.open(image_canvas.image_path)
                            
                            # Calculate scaling to fill canvas while maintaining aspect ratio
                            img_width, img_height = original_img.size
                            
                            # Calculate scale to fill the box with a slight margin
                            scale_x = canvas_width / img_width
                            scale_y = canvas_height / img_height
                            # Use max to fill the box, then shrink by 10% to add a small margin
                            scale = max(scale_x, scale_y) * 0.90
                            
                            new_width = int(img_width * scale)
                            new_height = int(img_height * scale)
                            
                            # Resize image
                            resized_img_pil = original_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                            resized_img = ImageTk.PhotoImage(resized_img_pil)
                            
                            # Center the image
                            x = canvas_width // 2
                            y = canvas_height // 2
                            
                            image_canvas.create_image(x, y, image=resized_img, anchor='center')
                            image_canvas.image = resized_img  # Keep reference to prevent garbage collection
                        except Exception as e:
                            print(f"⚠️ Error resizing image: {e}")
                
                # Bind resize event
                image_canvas.bind('<Configure>', resize_image)
                # Trigger initial resize after a short delay to ensure canvas is rendered
                image_canvas.after(100, resize_image)
            else:
                # Show blank/empty placeholder when no image
                placeholder_label = tk.Label(
                    picture_frame,
                    text="",
                    bg='#ffffff',
                    font=('Arial', 12),
                    fg='#d1d5db'
                )
                placeholder_label.pack(expand=True)
            
            # Create right panel (Information) - 33% width
            info_frame = tk.Frame(top_container, bg='#ffffff', relief='solid', bd=1)
            info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 10), pady=10)
            info_frame.pack_propagate(False)
            
            # Status indicator
            status_frame = tk.Frame(info_frame, bg=status_color, height=60)
            status_frame.pack(fill=tk.X, padx=10, pady=(10, 10))
            status_frame.pack_propagate(False)
            
            status_label = tk.Label(
                status_frame,
                text=status_text,
                font=('Arial', 20, 'bold'),
                fg='white',
                bg=status_color
            )
            status_label.pack(expand=True)
            
            # Person details
            details_frame = tk.Frame(info_frame, bg='#ffffff')
            details_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Name - strip whitespace
            name_val = person_info.get('name', 'Unknown')
            if name_val:
                name_val = str(name_val).strip()
            name_label = tk.Label(
                details_frame,
                text=name_val,
                font=('Arial', 28, 'bold'),
                fg='#1e3a8a',
                bg='#ffffff'
            )
            name_label.pack(pady=(0, 15))
            
            # ID - Use Student Number from Firebase (not RFID)
            # For students, show student_id (Student Number), for others use id/rfid
            if person_info.get('type', '').lower() == 'student':
                rfid_value = person_info.get('rfid')
                
                # Get Student Number from person_info - check 'id' field first as it may already contain Student Number
                student_number = (person_info.get('id') or 
                                person_info.get('student_id') or 
                                person_info.get('student_number') or
                                person_info.get('Student Number'))
                
                # Validate: Student Number should not be RFID, 'Unknown', 'Student Number Not Found', None, or empty
                is_valid = (student_number and 
                           student_number != 'Unknown' and 
                           student_number != 'Student Number Not Found' and
                           str(student_number).strip() != '' and
                           (not rfid_value or str(student_number).strip() != str(rfid_value).strip()))
                
                if not is_valid:
                    # If Student Number is missing or equals RFID, try to fetch from Firebase
                    rfid_to_lookup = person_info.get('rfid')
                    if rfid_to_lookup:
                        print(f"⚠️ Student Number missing or equals RFID in display, fetching from Firebase for RFID: {rfid_to_lookup}")
                        full_student_info = self.get_student_info_by_rfid(rfid_to_lookup)
                        if full_student_info:
                            student_number = full_student_info.get('student_id') or full_student_info.get('student_number') or full_student_info.get('id')
                            # Validate that we got a valid student number (not None, not empty, not RFID)
                            if student_number and str(student_number).strip() != '' and str(student_number).strip() != str(rfid_to_lookup).strip():
                                # Update person_info with correct Student Number
                                person_info['student_id'] = student_number
                                person_info['student_number'] = student_number
                                person_info['id'] = student_number
                                print(f"✅ Updated person_info with Student Number from Firebase: {student_number}")
                            else:
                                print(f"⚠️ Student Number from Firebase is invalid (None, empty, or equals RFID): {student_number}")
                                # Last resort: try to get it directly from Firebase document
                                try:
                                    if self.firebase_initialized and self.db:
                                        doc_ref = self.db.collection('students').document(rfid_to_lookup)
                                        doc = doc_ref.get()
                                        if doc.exists:
                                            doc_data = doc.to_dict()
                                            direct_student_num = doc_data.get('Student Number')
                                            if direct_student_num and str(direct_student_num).strip() != '' and str(direct_student_num).strip() != str(rfid_to_lookup).strip():
                                                student_number = str(direct_student_num).strip()
                                                person_info['student_id'] = student_number
                                                person_info['student_number'] = student_number
                                                person_info['id'] = student_number
                                                print(f"✅ Found Student Number via direct Firebase document access: {student_number}")
                                            else:
                                                student_number = 'Student Number Not Found'
                                        else:
                                            student_number = 'Student Number Not Found'
                                    else:
                                        student_number = 'Student Number Not Found'
                                except Exception as e:
                                    print(f"⚠️ Error accessing Firebase document directly: {e}")
                                    student_number = 'Student Number Not Found'
                        else:
                            student_number = 'Student Number Not Found'
                    else:
                        student_number = 'Student Number Not Found'
                
                # Strip whitespace from student number if it exists
                if student_number and student_number != 'Student Number Not Found':
                    student_number = str(student_number).strip()
                display_id = student_number
            else:
                # For non-students, use id or rfid
                display_id = person_info.get('id') or person_info.get('rfid', 'Unknown')
            
            id_label = tk.Label(
                details_frame,
                text=f"ID: {display_id}",
                font=('Arial', 20, 'bold'),
                fg='#374151',
                bg='#ffffff'
            )
            id_label.pack(pady=(0, 10))
            
            # Department (for students) - strip whitespace
            if person_info['type'] == 'student':
                course_val = person_info.get('course', 'Unknown Course')
                if course_val:
                    course_val = str(course_val).strip()
                course_label = tk.Label(
                    details_frame,
                    text=f"Department: {course_val}",
                    font=('Arial', 18, 'bold'),
                    fg='#3b82f6',
                    bg='#ffffff'
                )
                course_label.pack(pady=(0, 5))
                
                gender_val = person_info.get('gender', 'Unknown Gender')
                if gender_val:
                    gender_val = str(gender_val).strip()
                gender_label = tk.Label(
                    details_frame,
                    text=f"Gender: {gender_val}",
                    font=('Arial', 18, 'bold'),
                    fg='#3b82f6',
                    bg='#ffffff'
                )
                gender_label.pack(pady=(0, 5))
            
            # Time
            time_label = tk.Label(
                details_frame,
                text=f"Time: {person_info['timestamp']}",
                font=('Arial', 16, 'bold'),
                fg='#1e3a8a',
                bg='#ffffff'
            )
            time_label.pack(pady=(20, 0))
            
            # Note: Activity Log is now persistent and created in create_main_screen_content()
            # It won't be destroyed when returning to standby mode
            # The listbox is already available at self.recent_entries_listbox
            
            # Ensure explicit 'action' exists (map from status if necessary)
            if 'action' not in person_info or not person_info.get('action'):
                status_val = str(person_info.get('status', '')).upper()
                if status_val in ('TIME-IN', 'ENTRY', 'COMPLETE UNIFORM', 'DETECTING'):
                    person_info['action'] = 'ENTRY'
                elif status_val in ('TIME-OUT', 'EXIT', 'EXITED'):
                    person_info['action'] = 'EXIT'
                else:
                    person_info['action'] = person_info.get('status', 'Unknown')

            # Add to recent entries
            # Only add to recent entries if it's an actual ENTRY or EXIT, not DETECTING
            # Entry should only be logged after complete uniform detection
            # Skip for teachers since they already add entries in handle_teacher_timein/timeout
            person_type = person_info.get('type', '').lower()
            status = person_info.get('status', '').upper()
            if status not in ('DETECTING', 'INCOMPLETE UNIFORM'):
                # Only log ENTRY/EXIT, not detection states
                # Skip for teachers to avoid duplicates (they add entries in handle_teacher_timein/timeout)
                if person_type != 'teacher':
                    self.add_to_recent_entries(person_info)
            
            # Schedule clearing main screen after 15 seconds (15000 milliseconds)
            self._schedule_main_screen_clear()
            
            print(f"SUCCESS: Main screen updated with {person_info['name']} ({person_info['status']})")
            
        except Exception as e:
            print(f"ERROR: Error updating main screen: {e}")
    
    def _schedule_main_screen_clear(self):
        """Schedule clearing the main screen after 15 seconds, but skip if violation is pending or DENIED was clicked"""
        try:
            # Check if DENIED button was clicked - if so, don't schedule auto-clear (waiting for guard decision)
            if hasattr(self, 'denied_button_clicked') and self.denied_button_clicked:
                print(f"⏸️ Main screen clear skipped - DENIED button clicked, waiting for guard decision")
                return
            
            # Check if there's a violation pending - if so, don't schedule auto-clear
            # Violation is pending if incomplete_student_info_for_approve is set or if there are active violations
            has_pending_violation = (
                hasattr(self, 'incomplete_student_info_for_approve') and 
                self.incomplete_student_info_for_approve is not None
            ) or (
                hasattr(self, 'active_session_violations') and 
                self.active_session_violations and 
                any(len(violations) > 0 for violations in self.active_session_violations.values())
            )
            
            if has_pending_violation:
                print(f"⏸️ Main screen clear skipped - violation pending, waiting for gate control button")
                return
            
            # Cancel any existing timer
            if self.main_screen_clear_timer is not None:
                try:
                    self.root.after_cancel(self.main_screen_clear_timer)
                except Exception:
                    pass
                self.main_screen_clear_timer = None
            
            # Schedule new clear timer (15 seconds = 15000 milliseconds)
            self.main_screen_clear_timer = self.root.after(15000, self._clear_main_screen_after_delay)
            print(f"✅ Scheduled main screen clear after 15 seconds")
        except Exception as e:
            print(f"⚠️ Error scheduling main screen clear: {e}")
    
    def _clear_main_screen_after_delay(self):
        """Clear the main screen person display and return to standby"""
        try:
            print(f"🔍 _clear_main_screen_after_delay() called - starting clear process")
            # Clear the timer reference
            self.main_screen_clear_timer = None
            
            # Check if DENIED button was clicked - if so, don't clear (waiting for guard decision)
            if hasattr(self, 'denied_button_clicked') and self.denied_button_clicked:
                print(f"⏸️ Main screen clear cancelled - DENIED button clicked, waiting for guard decision")
                return
            
            # Check if detection is still active - if so, don't clear yet
            # (Detection might still be running and we don't want to interrupt it)
            if hasattr(self, 'detection_active') and self.detection_active:
                print(f"⏸️ Main screen clear skipped - detection still active")
                # Reschedule for another 15 seconds
                self._schedule_main_screen_clear()
                return
            
            # CRITICAL: If entry was just saved (detection_active=False, uniform_detection_complete=True),
            # we should clear regardless of violations - this timer was scheduled specifically for that purpose
            entry_was_saved = (
                hasattr(self, 'detection_active') and not self.detection_active and
                hasattr(self, 'uniform_detection_complete') and self.uniform_detection_complete
            )
            
            if not entry_was_saved:
                # Only check for pending violations if entry was NOT saved
                # Violation is pending if incomplete_student_info_for_approve is set or if there are active violations
                has_pending_violation = (
                    hasattr(self, 'incomplete_student_info_for_approve') and 
                    self.incomplete_student_info_for_approve is not None
                ) or (
                    hasattr(self, 'active_session_violations') and 
                    self.active_session_violations and 
                    any(len(violations) > 0 for violations in self.active_session_violations.values())
                )
                
                if has_pending_violation:
                    print(f"⏸️ Main screen clear skipped - violation pending, waiting for gate control button")
                    return
            else:
                print(f"✅ Entry was saved - proceeding with main screen clear (violations already finalized)")
            
            # Clear all widgets from person display frame
            if hasattr(self, 'person_display_frame'):
                for widget in self.person_display_frame.winfo_children():
                    try:
                        widget.destroy()
                    except Exception:
                        pass
            
            # Return to standby mode - show standby message
            self.show_standby_message()
            
            # CRITICAL: Disable DENIED button when returning to standby
            # This ensures buttons are disabled after entry status is displayed and cleared
            if hasattr(self, 'deny_button') and self.deny_button:
                try:
                    self.deny_button.config(state=tk.DISABLED)
                    print(f"✅ DENIED button disabled - returned to standby")
                except Exception as e:
                    print(f"⚠️ WARNING: Could not disable DENIED button: {e}")
            
            # Also ensure APPROVE and CANCEL buttons are disabled
            if hasattr(self, 'approve_button') and self.approve_button:
                try:
                    self.approve_button.config(state=tk.DISABLED)
                except Exception as e:
                    pass
            
            if hasattr(self, 'cancel_button') and self.cancel_button:
                try:
                    self.cancel_button.config(state=tk.DISABLED)
                except Exception as e:
                    pass
            
            print(f"✅ Main screen cleared - returned to standby")
            
        except Exception as e:
            print(f"⚠️ Error clearing main screen: {e}")
    
    def update_main_screen_with_exit(self, person_id, person_type):
        """Update main screen with exit information"""
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Get person name
            person_name = self.get_person_name(person_id, person_type)
            # Try to fetch department/course and gender for students from Firebase
            course_val = 'N/A'
            gender_val = 'N/A'
            try:
                if person_type.lower() == 'student':
                    # Try lookup by RFID first (many setups store RFID as doc id)
                    student_info = self.get_student_info_by_rfid(person_id)
                    if not student_info:
                        # Fallback to general student lookup (handles student-number doc ids)
                        student_info = self.get_student_info(person_id)

                    if student_info:
                        course_val = student_info.get('course', student_info.get('Course', student_info.get('department', 'Unknown')))
                        gender_val = student_info.get('gender', student_info.get('Gender', 'Unknown'))
            except Exception:
                pass

            exit_info = {
                'id': person_id,
                'name': person_name,
                'type': person_type,
                'course': course_val,
                'gender': gender_val,
                'timestamp': current_time,
                'status': 'TIME-OUT',
                'guard_id': self.current_guard_id or 'Unknown'
            }
            
            # Update main screen
            self.update_main_screen_with_person(exit_info)
            
        except Exception as e:
            print(f"ERROR: Error updating main screen with exit: {e}")
    
    def _update_main_screen_with_detection_result(self, status, complete_components, missing_components):
        """Update main screen with uniform detection results"""
        try:
            if not self.main_screen_window or not self.main_screen_window.winfo_exists():
                return
            
            # Get current person info
            if not hasattr(self, 'current_person_id') or not self.current_person_id:
                return
            
            person_id = self.current_person_id
            person_type = getattr(self, 'current_person_type', 'student')
            
            # Get person information
            person_info = self.get_person_info_for_main_screen(person_id, person_type)
            
            # Add detection results to person info
            person_info['detection_status'] = status
            person_info['complete_components'] = complete_components
            person_info['missing_components'] = missing_components
            
            # Update main screen with enhanced information
            self.update_main_screen_with_detection_details(person_info)
            
        except Exception as e:
            print(f"ERROR: Error updating main screen with detection result: {e}")
    
    def update_main_screen_with_detection_details(self, person_info):
        """Update main screen with detailed detection information"""
        try:
            if not self.main_screen_window or not self.main_screen_window.winfo_exists():
                return
            
            # Clear existing content
            for widget in self.person_display_frame.winfo_children():
                widget.destroy()
            
            # Create main info frame
            main_info_frame = tk.Frame(self.person_display_frame, bg='#ffffff')
            main_info_frame.pack(fill=tk.BOTH, expand=True)
            
            # Status indicator based on detection result
            if person_info.get('detection_status') == 'COMPLETE UNIFORM':
                status_color = '#10b981'  # Green for complete
                status_text = 'UNIFORM COMPLIANT'
            else:
                status_color = '#ef4444'  # Red for incomplete
                status_text = 'UNIFORM VIOLATION'
            
            status_frame = tk.Frame(main_info_frame, bg=status_color, height=60)
            status_frame.pack(fill=tk.X, padx=20, pady=(20, 10))
            status_frame.pack_propagate(False)
            
            status_label = tk.Label(
                status_frame,
                text=status_text,
                font=('Arial', 24, 'bold'),
                fg='white',
                bg=status_color
            )
            status_label.pack(expand=True)
            
            # Person details
            details_frame = tk.Frame(main_info_frame, bg='#ffffff')
            details_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
            
            # Name
            name_label = tk.Label(
                details_frame,
                text=person_info['name'],
                font=('Arial', 28, 'bold'),
                fg='#1e3a8a',
                bg='#ffffff'
            )
            name_label.pack(pady=(0, 15))
            
            # ID
            id_label = tk.Label(
                details_frame,
                text=f"ID: {person_info['id']}",
                font=('Arial', 20, 'bold'),
                fg='#374151',
                bg='#ffffff'
            )
            id_label.pack(pady=(0, 10))
            
            # Detection results
            if person_info.get('detection_status'):
                detection_label = tk.Label(
                    details_frame,
                    text=f"Detection: {person_info['detection_status']}",
                    font=('Arial', 18, 'bold'),
                    fg=status_color,
                    bg='#ffffff'
                )
                detection_label.pack(pady=(0, 10))
                
                # Complete components
                if person_info.get('complete_components'):
                    complete_text = f"SUCCESS: Detected: {', '.join(person_info['complete_components'])}"
                    complete_label = tk.Label(
                        details_frame,
                        text=complete_text,
                        font=('Arial', 14, 'bold'),
                        fg='#10b981',
                        bg='#ffffff'
                    )
                    complete_label.pack(pady=(0, 5))
                
                # Missing components
                if person_info.get('missing_components'):
                    missing_text = f"ERROR: Missing: {', '.join(person_info['missing_components'])}"
                    missing_label = tk.Label(
                        details_frame,
                        text=missing_text,
                        font=('Arial', 14, 'bold'),
                        fg='#ef4444',
                        bg='#ffffff'
                    )
                    missing_label.pack(pady=(0, 5))
            
            # Time
            time_label = tk.Label(
                details_frame,
                text=f"Time: {person_info['timestamp']}",
                font=('Arial', 16, 'bold'),
                fg='#1e3a8a',
                bg='#ffffff'
            )
            time_label.pack(pady=(20, 0))
            
            # Note: "Processed by" field removed - RFID is now shown in ID field above
            
            # Add to recent entries
            self.add_to_recent_entries(person_info)
            
            print(f"SUCCESS: Main screen updated with detection results for {person_info['name']}")
            
        except Exception as e:
            print(f"ERROR: Error updating main screen with detection details: {e}")
    
    def get_student_info(self, person_id):
        """Get student information from Firebase or offline data"""
        try:
            # Try Firebase first if available
            if self.firebase_initialized and self.db:
                try:
                    # Query Firebase students collection
                    # First attempt: document id lookup (fastest)
                    doc_ref = self.db.collection('students').document(person_id)
                    doc = doc_ref.get()

                    if doc.exists:
                        # Get document data
                        data = doc.to_dict()
                        # Determine course: prefer Senior High School / Strand for SHS students
                        senior_high = None
                        # check several possible keys for senior high / strand
                        for key in ('Senior High School', 'senior_high_school', 'senior_high', 'Strand', 'strand'):
                            if key in data and data.get(key):
                                senior_high = str(data.get(key)).strip()
                                break

                        # Extract student information, normalize course
                        if senior_high:
                            course_val = f"SHS {senior_high}"
                        else:
                            course_val = data.get('Course', data.get('Department', 'Unknown'))

                        # Get Student Number from Firebase
                        student_number = None
                        if 'Student Number' in data:
                            student_number = str(data.get('Student Number', '')).strip()
                            if not student_number or student_number == '':
                                student_number = None

                        student_info = {
                            'name': data.get('Name', 'Unknown'),
                            'course': course_val,
                            'gender': data.get('Gender', 'Unknown'),
                            'rfid': person_id,  # CRITICAL: person_id is the RFID (document ID)
                            'student_id': student_number,  # Student Number from Firebase
                            'student_number': student_number,  # Alternative field name
                            'id': student_number if student_number else person_id,  # Use Student Number if available, otherwise RFID
                        }
                        
                        print(f"SUCCESS: Student found in Firebase: {student_info['name']} ({student_info['course']})")
                        print(f"🔍 DEBUG get_student_info: Returning keys: {list(student_info.keys())}, RFID: {student_info.get('rfid')}, Student Number: {student_info.get('student_id')}")
                        return student_info
                    # If document lookup failed, try common student-number fields
                    candidate_fields = ['Student Number', 'student_number', 'student_id', 'StudentID', 'ID', 'id']
                    for field in candidate_fields:
                        try:
                            query = self.db.collection('students').where(field, '==', person_id).limit(1).stream()
                            # stream() returns generator of docs
                            results = list(query)
                            if results:
                                data = results[0].to_dict()
                                senior_high = None
                                for key in ('Senior High School', 'senior_high_school', 'senior_high', 'Strand', 'strand'):
                                    if key in data and data.get(key):
                                        senior_high = str(data.get(key)).strip()
                                        break

                                if senior_high:
                                    course_val = f"SHS {senior_high}"
                                else:
                                    course_val = data.get('Course', data.get('Department', 'Unknown'))

                                # Get Student Number from Firebase
                                student_number = None
                                if 'Student Number' in data:
                                    student_number = str(data.get('Student Number', '')).strip()
                                    if not student_number or student_number == '':
                                        student_number = None
                                
                                # Get document ID as RFID
                                doc_id = results[0].id

                                student_info = {
                                    'name': data.get('Name', 'Unknown'),
                                    'course': course_val,
                                    'gender': data.get('Gender', 'Unknown'),
                                    'rfid': doc_id,  # CRITICAL: Document ID is the RFID
                                    'student_id': student_number,  # Student Number from Firebase
                                    'student_number': student_number,  # Alternative field name
                                    'id': student_number if student_number else doc_id,  # Use Student Number if available, otherwise RFID
                                }
                                print(f"SUCCESS: Student found in Firebase by field {field}: {student_info['name']} ({student_info['course']})")
                                print(f"🔍 DEBUG get_student_info: Returning keys: {list(student_info.keys())}, RFID: {student_info.get('rfid')}, Student Number: {student_info.get('student_id')}")
                                return student_info
                        except Exception:
                            # ignore field-specific query errors and continue
                            pass

                    # As a last resort, scan a small subset and compare normalized field values to handle quoted/trailing-space values
                    try:
                        docs = self.db.collection('students').stream()
                        for d in docs:
                            data = d.to_dict()
                            # try multiple candidate fields and compare normalized strings
                            for fld in candidate_fields:
                                if fld in data and data.get(fld):
                                    val = str(data.get(fld)).strip().strip('"').strip("'")
                                    if val == str(person_id).strip():
                                        senior_high = None
                                        for key in ('Senior High School', 'senior_high_school', 'senior_high', 'Strand', 'strand'):
                                            if key in data and data.get(key):
                                                senior_high = str(data.get(key)).strip()
                                                break

                                        if senior_high:
                                            course_val = f"SHS {senior_high}"
                                        else:
                                            course_val = data.get('Course', data.get('Department', 'Unknown'))

                                        # Get Student Number from Firebase
                                        student_number = None
                                        if 'Student Number' in data:
                                            student_number = str(data.get('Student Number', '')).strip()
                                            if not student_number or student_number == '':
                                                student_number = None
                                        
                                        # Get document ID as RFID
                                        doc_id = d.id

                                        student_info = {
                                            'name': data.get('Name', 'Unknown'),
                                            'course': course_val,
                                            'gender': data.get('Gender', 'Unknown'),
                                            'rfid': doc_id,  # CRITICAL: Document ID is the RFID
                                            'student_id': student_number,  # Student Number from Firebase
                                            'student_number': student_number,  # Alternative field name
                                            'id': student_number if student_number else doc_id,  # Use Student Number if available, otherwise RFID
                                        }
                                        print(f"SUCCESS: Student found in Firebase by normalized scan: {student_info['name']} ({student_info['course']})")
                                        print(f"🔍 DEBUG get_student_info: Returning keys: {list(student_info.keys())}, RFID: {student_info.get('rfid')}, Student Number: {student_info.get('student_id')}")
                                        return student_info
                    except Exception:
                        pass
                except Exception as e:
                    print(f"WARNING: Firebase query failed: {e}")
            
            # Fallback to offline data
            return self.get_offline_student_info(person_id)
                
        except Exception as e:
            print(f"ERROR: Error fetching student info: {e}")
            return self.get_offline_student_info(person_id)
    
    def get_offline_student_info(self, person_id):
        """Get student information from offline data"""
        try:
            import json
            if os.path.exists("offline_students.json"):
                with open("offline_students.json", "r") as f:
                    offline_students = json.load(f)
                
                if person_id in offline_students:
                    student_info = offline_students[person_id]
                    print(f"SUCCESS: Student found in offline data: {student_info['name']} ({student_info['course']})")
                    return student_info
            
            # Default fallback
            print(f"WARNING: Student ID '{person_id}' not found - using default")
            return {
                'name': f'Student {person_id}',
                'course': 'ICT',
                'gender': 'MALE'
            }
                
        except Exception as e:
            print(f"ERROR: Error loading offline student data: {e}")
            return {
                'name': f'Student {person_id}',
                'course': 'ICT',
                'gender': 'MALE'
            }
    
    def retry_person_entry(self):
        """Retry person entry after Firebase initialization"""
        try:
            print("INFO: Retrying person entry after Firebase initialization...")
            self.add_activity_log("INFO: Retrying person entry after Firebase initialization...")
            
            # Try to initialize Firebase again with more detailed error handling
            if not self.firebase_initialized:
                print("INFO: Attempting Firebase initialization in retry...")
                self.add_activity_log("INFO: Attempting Firebase initialization in retry...")
                
                # Try multiple initialization methods
                success = False
                try:
                    success = self.init_firebase()
                except Exception as e:
                    print(f"ERROR: Firebase initialization failed in retry: {e}")
                    self.add_activity_log(f"ERROR: Firebase initialization failed in retry: {e}")
                
                if not success:
                    # Try alternative initialization
                    try:
                        print("INFO: Trying alternative Firebase initialization...")
                        self.init_firebase_async()
                        time.sleep(3)  # Wait longer for async initialization
                    except Exception as e:
                        print(f"ERROR: Alternative Firebase initialization failed: {e}")
                        self.add_activity_log(f"ERROR: Alternative Firebase initialization failed: {e}")
            
            if self.firebase_initialized:
                print("SUCCESS: Firebase is now ready, processing person entry...")
                self.add_activity_log("SUCCESS: Firebase is now ready, processing person entry...")
                # Process the person entry directly instead of calling log_person_entry again
                self.process_person_entry_after_retry()
            else:
                print("WARNING: Firebase still not ready, will retry again in 5 seconds...")
                self.add_activity_log("WARNING: Firebase still not ready, will retry again...")
                # Limit retries to prevent infinite loop
                if not hasattr(self, 'firebase_retry_count'):
                    self.firebase_retry_count = 0
                self.firebase_retry_count += 1
                
                if self.firebase_retry_count <= 2:  # Max 2 retries
                    print(f"INFO: Retry attempt {self.firebase_retry_count}/2")
                    self.root.after(3000, lambda: self.retry_person_entry())
                else:
                    print("ERROR: Firebase initialization failed after multiple retries - switching to offline mode")
                    self.add_activity_log("ERROR: Firebase initialization failed after multiple retries - switching to offline mode")
                    self.firebase_initialized = False
                    # Process with offline data
                    self.process_person_entry_offline()
        except Exception as e:
            print(f"ERROR: Error in retry person entry: {e}")
            self.add_activity_log(f"ERROR: Error in retry person entry: {e}")
    
    def process_person_entry_after_retry(self):
        """Process person entry after successful Firebase retry"""
        try:
            person_id = self.person_id_var.get().strip().upper()
            person_type = self.person_type_var.get()
            
            if not person_id:
                return
            
            print(f"INFO: Processing person entry after retry: {person_id} ({person_type})")
            self.add_activity_log(f"INFO: Processing person entry after retry: {person_id} ({person_type})")
            
            # Check if this is a permanent student RFID tap
            if person_type == "student" and self.is_permanent_student_rfid(person_id):
                self.handle_permanent_student_rfid_tap(person_id)
                self.person_id_var.set("")  # Clear input
                return
            
            # For other types, use the regular processing
            # (This is a simplified version - you might need to add other cases)
            print(f"INFO: Regular processing for {person_id} ({person_type})")
            
        except Exception as e:
            print(f"ERROR: Error processing person entry after retry: {e}")
            self.add_activity_log(f"ERROR: Error processing person entry after retry: {e}")
    
    def process_person_entry_offline(self):
        """Process person entry using offline data when Firebase is unavailable"""
        try:
            person_id = self.person_id_var.get().strip().upper()
            person_type = self.person_type_var.get()
            
            if not person_id:
                return
            
            print(f"INFO: Processing person entry offline: {person_id} ({person_type})")
            self.add_activity_log(f"INFO: Processing person entry offline: {person_id} ({person_type})")
            
            # Create basic person info for offline mode
            from datetime import datetime
            person_info = {
                'id': person_id,
                'name': f'Student {person_id}' if person_type == 'student' else f'{person_type.title()} {person_id}',
                'type': person_type,
                'course': 'Unknown',
                'gender': 'Unknown',
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'status': 'TIME-IN',
                'guard_id': self.current_guard_id or 'Unknown'
            }
            
            # Update main screen with offline person information
            if self.main_screen_window and self.main_screen_window.winfo_exists():
                self.update_main_screen_with_person(person_info)
            
            # Clear the input
            self.person_id_var.set("")
            
            print(f"SUCCESS: Person entry processed offline: {person_info['name']}")
            self.add_activity_log(f"SUCCESS: Person entry processed offline: {person_info['name']}")
            
        except Exception as e:
            print(f"ERROR: Error processing person entry offline: {e}")
            self.add_activity_log(f"ERROR: Error processing person entry offline: {e}")
    
    def get_person_name(self, person_id, person_type):
        """Get person name based on ID and type"""
        if person_type == "student":
            student_info = self.get_student_info(person_id)
            return student_info['name']
        elif person_type == "teacher":
            return f"Teacher {person_id}"
        else:
            return f"Person {person_id}"
    
    def _get_model_path_for_person(self, person_id, person_type):
        """Get the appropriate model path for a person based on their type and gender"""
        import os
        try:
            model_path = None
            if person_type.lower() == 'student':
                # Get student info to determine gender
                student_info = self.get_student_info(person_id)
                gender = student_info.get('gender', '').lower()
                course = student_info.get('course', '').upper()
                
                print(f"🔍 Model selection for: {person_id} | Course: {course} | Gender: {gender}")
                
                # Special RFID overrides (check these first)
                if str(person_id).strip() == '0095095703':
                    model_path = 'ict and eng.pt'
                elif str(person_id).strip() == '0095129433':
                    model_path = 'arts and science.pt'
                elif str(person_id).strip() == '0095339862':
                    model_path = 'tourism male.pt'
                elif str(person_id).strip() == '0095272249':
                    model_path = 'bshm.pt'  # BSHM student
                elif str(person_id).strip() == '0095365253':
                    model_path = 'shs.pt'
                # Special-case: certain courses (ICT/BSCPE/ENG) use the ICT+ENG combined model
                elif any(k in course for k in ("ICT", "BSCPE", "ENG", "ICT AND ENG", "ICT/ENG")):
                    model_path = 'ict and eng.pt'
                # Arts & Science course uses its own model
                elif any(k in course for k in ("ARTS", "SCIENCE", "ARTS AND SCIENCE", "ARTS/SCIENCE")):
                    model_path = 'arts and science.pt'
                # Tourism course uses the tourism model
                elif any(k in course for k in ("TOURISM", "TOURISM MALE", "TOURISM_FEMALE")):
                    model_path = 'tourism male.pt'
                # BSHM / Hospitality Management model mapping
                elif any(k in course for k in ("BSHM", "HOSPITALITY", "HOSPITALITY MANAGEMENT")):
                    model_path = 'bshm.pt'
                # HM model mapping
                elif any(k in course for k in ("HM", "HONORS", "HUMANITIES")):
                    model_path = 'hm.pt'
                # SHS model mapping (Senior High School) - includes SHS tracks like STEM, ABM, HUMSS, GAS, TVL
                elif any(k in course for k in ("SHS", "SENIOR HIGH", "SENIOR_HIGH", "STEM", "ABM", "HUMSS", "GAS", "TVL")):
                    model_path = 'shs.pt'
                # Default to BSBA by gender
                elif 'female' in gender:
                    model_path = 'bsba_female.pt'
                elif 'male' in gender:
                    model_path = 'bsba male2.pt'
                else:
                    # Default to male model if gender is unclear
                    model_path = 'bsba male2.pt'
            else:
                # For non-students, default to male model
                model_path = 'bsba male2.pt'
            
            # Verify model file exists
            if model_path:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                full_path = os.path.join(script_dir, model_path)
                
                # Normalize path to handle spaces and ensure correct format
                full_path = os.path.normpath(full_path)
                
                if os.path.exists(full_path):
                    print(f"✅ Model file found: {model_path} (Full path: {full_path})")
                    # Return relative path as YOLO handles it correctly from current directory
                    return model_path
                else:
                    print(f"❌ Model file NOT found: {full_path}")
                    print(f"📁 Checking directory: {script_dir}")
                    # List available .pt files for debugging
                    try:
                        pt_files = [f for f in os.listdir(script_dir) if f.endswith('.pt')]
                        if pt_files:
                            print(f"📋 Available .pt files in directory:")
                            for f in pt_files:
                                print(f"   - {f}")
                        else:
                            print(f"⚠️ No .pt files found in directory")
                    except Exception as e:
                        print(f"⚠️ Could not list directory contents: {e}")
                    print(f"⚠️ Falling back to default model: bsba male2.pt")
                    return 'bsba male2.pt'
            else:
                print(f"⚠️ No model path determined, using default: bsba male2.pt")
                return 'bsba male2.pt'
        except Exception as e:
            print(f"WARNING: Could not determine model for person {person_id}: {e}")
            import traceback
            traceback.print_exc()
            # Default fallback
            return 'bsba male2.pt'
    
    def reset_uniform_tracking(self):
        """Reset uniform tracking for new person"""
        try:
            # Reset any uniform tracking variables if they exist
            if hasattr(self, 'detected_components'):
                self.detected_components = {}
            if hasattr(self, 'uniform_complete'):
                self.uniform_complete = False
            if hasattr(self, 'detection_start_time'):
                self.detection_start_time = None
            print("DEBUG: Uniform tracking reset")
        except Exception as e:
            print(f"WARNING: Could not reset uniform tracking: {e}")
    
    def start_person_detection(self, person_id, person_name, person_type):
        """Start detection for a person"""
        try:
            # CRITICAL: Check Event Mode FIRST - if active, do NOT start detection or show student info
            if getattr(self, 'event_mode_active', False):
                print(f"🛑 Event Mode active - NOT starting detection for {person_name}")
                # Keep camera label in standby mode (no student info)
                try:
                    if hasattr(self, 'camera_label') and self.camera_label:
                        standby_text = (
                            "📷 CAMERA PREVIEW DISABLED\n\n"
                            "Detection runs in the external Camera Detection window.\n"
                            "Tap a student RFID to start detection.\n\n"
                            "Press 'q' in the detection window to close it."
                        )
                        self.camera_label.config(text=standby_text, bg='#dbeafe', fg='#374151')
                        self.camera_label.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
                        print("✅ Camera label kept in standby (Event Mode)")
                except Exception as e:
                    print(f"⚠️ Warning: Could not set camera label to standby: {e}")
                return False  # Do not start detection
            
            # Store current person ID for detection context
            self.current_person_id = person_id
            self.last_person_id = person_id
            
            # Determine model based on person type or ID
            model_path = self._get_model_path_for_person(person_id, person_type)
            if model_path != self.current_model_path:
                self.current_model_path = model_path
                print(f"INFO: Switching to model: {model_path}")
                
                # Log model selection with student details
                if person_type.lower() == 'student':
                    student_info = self.get_student_info(person_id)
                    print(f"📋 Student: {student_info['name']} | Course: {student_info['course']} | Gender: {student_info['gender']}")
                    print(f"🤖 Selected Model: {model_path}")
            
            # Update camera label
            self.update_camera_label_for_detection(person_id, person_name, person_type)
            
            # Reset uniform tracking for new person
            self.reset_uniform_tracking()
            
            # Reset detection history for new detection session
            if hasattr(self, 'detection_system') and self.detection_system:
                self.detection_system.reset_detection_history()
            
            # Update main screen with person information
            self.update_main_screen_with_person_info(person_id, person_name, person_type)
            
            # Check if student should use external camera detection (Tourism, BSHM, SHS)
            use_external_detection = False
            if person_type.lower() == 'student':
                try:
                    student_info = self.get_student_info(person_id)
                    if student_info:
                        course = student_info.get('course', '').upper()
                        # Use external camera detection for Tourism, BSHM, and SHS
                        if any(k in course for k in ("TOURISM", "BSHM", "HOSPITALITY", "SHS", "SENIOR HIGH")):
                            use_external_detection = True
                            print(f"🎯 Using external camera detection for course: {course}")
                except Exception as e:
                    print(f"⚠️ Error checking course for external detection: {e}")
            
            # Start detection using appropriate method
            print(f"DEBUG: CV2_AVAILABLE: {CV2_AVAILABLE}, YOLO_AVAILABLE: {YOLO_AVAILABLE}")
            self.add_activity_log(f"DEBUG: CV2_AVAILABLE: {CV2_AVAILABLE}, YOLO_AVAILABLE: {YOLO_AVAILABLE}")
            
            if use_external_detection:
                # Use external camera detection (camera detection.py) for Tourism, BSHM, SHS
                print("DEBUG: Starting external camera detection")
                self.add_activity_log("DEBUG: Starting external camera detection")
                try:
                    student_info = self.get_student_info(person_id)
                    if student_info:
                        self.start_external_camera_detection(student_info)
                    else:
                        print("⚠️ Could not get student info for external detection")
                except Exception as e:
                    print(f"❌ Failed to start external detection: {e}")
            elif CV2_AVAILABLE and YOLO_AVAILABLE:
                print("DEBUG: Starting real detection with integrated system")
                self.add_activity_log("DEBUG: Starting real detection with integrated system")
                self.start_person_detection_integrated(person_id, person_name, person_type)
            else:
                print("WARNING: CV2 or YOLO not available, detection disabled")
                self.add_activity_log("WARNING: CV2 or YOLO not available, detection disabled")
            
        except Exception as e:
            print(f"ERROR: Failed to start person detection: {e}")
            # Only log error if Event Mode is OFF (in Event Mode, detection shouldn't start anyway)
            if not getattr(self, 'event_mode_active', False):
                self.add_activity_log(f"Failed to start detection: {e}")
            else:
                print(f"🛑 Event Mode active - NOT logging detection failure")
    
    def update_camera_label_for_detection(self, person_id, person_name, person_type):
        """Update camera label to show detection status"""
        try:
            # CRITICAL: In Event Mode, do NOT show student info - keep in standby
            if getattr(self, 'event_mode_active', False):
                print(f"🛑 Event Mode active - NOT updating camera label with student info")
                # Keep camera label in standby mode
                try:
                    if hasattr(self, 'camera_label') and self.camera_label:
                        standby_text = (
                            "📷 CAMERA PREVIEW DISABLED\n\n"
                            "Detection runs in the external Camera Detection window.\n"
                            "Tap a student RFID to start detection.\n\n"
                            "Press 'q' in the detection window to close it."
                        )
                        self.camera_label.config(text=standby_text, bg='#dbeafe', fg='#374151')
                        self.camera_label.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
                except Exception:
                    pass
                return  # Do not show student info in Event Mode
            
            # Check if camera label exists and UI is ready
            if not hasattr(self, 'camera_label') or not self.camera_label or not self.root.winfo_exists():
                print("WARNING: Camera label not available for detection update")
                return
            
            # Get student information to show course and gender
            if person_type.lower() == 'student':
                student_info = self.get_student_info(person_id)
                course = student_info['course']
                gender = student_info['gender']
                model_info = f"Model: {self.current_model_path}"
            else:
                course = "N/A"
                gender = "N/A"
                model_info = f"Model: {self.current_model_path}"
            
            detection_text = f"""🎥 LIVE CAMERA FEED - DETECTION ACTIVE

👤 Person: {person_name}
🆔 ID: {person_id}
📚 Type: {person_type.title()}
🎓 Course: {course}
👤 Gender: {gender}

📷 Camera Starting...
INFO: AI Detection Initializing...
🤖 {model_info}

💡 Position yourself in front of the camera
for uniform detection

WARNING: Live video preview will appear here
once camera is fully initialized"""
            
            self.camera_label.config(
                text=detection_text,
                fg='#059669',
                font=('Arial', 11, 'bold'),
                bg='#d1fae5',
                relief='sunken',
                bd=3,
                justify=tk.CENTER
            )
            # Maintain portrait orientation
            self.camera_label.pack(expand=True, fill=tk.BOTH, padx=15, pady=15)
            
            print(f"SUCCESS: Camera label updated for {person_name} ({person_id})")
            
        except Exception as e:
            print(f"ERROR: Failed to update camera label: {e}")
    
    def get_detection_performance_stats(self):
        """Get current detection performance statistics"""
        if hasattr(self, 'detection_system') and self.detection_system:
            return {
                'fps': getattr(self.detection_system, 'fps', 0),
                'frame_count': getattr(self.detection_system, 'frame_count', 0),
                'frame_skip': getattr(self.detection_system, 'frame_skip', 2)
            }
        return {}
    def stop_detection(self):
        """Stop the detection system"""
        self.detection_active = False
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=2)
        
        # Stop camera display loop
        self.camera_active = False
        
        # Close camera
        if hasattr(self, 'camera_cap') and self.camera_cap:
            self.camera_cap.release()
            print("INFO: Camera closed")
        
        # Close camera display window
        try:
            print("INFO: Camera display window closed")
        except Exception as e:
            print(f"WARNING: Error closing camera window: {e}")
        
        # Cleanup detection system
        if self.detection_system:
            self.detection_system.cleanup()
            self.detection_system = None
        
        # Reset camera to standby mode
        self.reset_camera_to_standby()
        
        # Reset uniform tracking
        self.reset_uniform_tracking()
        
        # Keep requirements section visible - only hide when gate control button is clicked
        # Don't hide when detection stops - requirements stay visible until guard clicks a button
        # try:
        #     self.hide_requirements_section()
        # except Exception:
        #     pass
        
        # Return main screen to standby
        if self.main_screen_window and self.main_screen_window.winfo_exists():
            self.show_standby_message()
        
        print("🛑 Detection system stopped - Camera closed and returned to standby")
    
    def add_activity_log(self, message):
        """Add message to activity log"""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_entry = f"[{timestamp}] {message}\n"
            
            # Add to logs list
            self.activity_logs.append(log_entry)
            
            # Keep only the last max_logs entries
            if len(self.activity_logs) > self.max_logs:
                self.activity_logs = self.activity_logs[-self.max_logs:]
            
            # Update logs text widget if it exists
            if hasattr(self, 'logs_text') and self.logs_text and self.logs_text.winfo_exists():
                try:
                    self.logs_text.config(state=tk.NORMAL)
                    self.logs_text.insert(tk.END, log_entry)
                    self.logs_text.see(tk.END)
                    self.logs_text.config(state=tk.DISABLED)
                except Exception as e:
                    print(f"WARNING: Could not update logs text widget: {e}")
            
            print(f"📝 Activity Log: {message}")
            
        except Exception as e:
            print(f"ERROR: Error adding to activity log: {e}")
    
    def show_green_success_message(self, title, message):
        """Show a success message with green color"""
        try:
            # Create a custom messagebox window
            success_window = tk.Toplevel(self.root)
            success_window.title(title)
            success_window.geometry("400x150")
            success_window.configure(bg='#f0f9ff')
            success_window.resizable(False, False)
            
            # Ensure it's a child of the Guard UI window only
            success_window.transient(self.root)
            success_window.grab_set()
            
            # Center on the Guard UI window, not the entire screen
            self.root.update_idletasks()
            guard_x = self.root.winfo_x()
            guard_y = self.root.winfo_y()
            guard_width = self.root.winfo_width()
            guard_height = self.root.winfo_height()
            
            # Calculate center position within the Guard UI window
            x = guard_x + (guard_width // 2) - (400 // 2)
            y = guard_y + (guard_height // 2) - (150 // 2)
            success_window.geometry(f"400x150+{x}+{y}")
            
            # Ensure the window stays on top of Guard UI but not main screen
            success_window.lift(self.root)
            success_window.attributes('-topmost', True)
            
            # Main frame
            main_frame = tk.Frame(success_window, bg='#f0f9ff')
            main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
            
            # Success icon (green checkmark)
            icon_label = tk.Label(
                main_frame,
                text="SUCCESS:",
                font=('Arial', 24),
                bg='#f0f9ff',
                fg='#10b981'
            )
            icon_label.pack(pady=(0, 10))
            
            # Title
            title_label = tk.Label(
                main_frame,
                text=title,
                font=('Arial', 16, 'bold'),
                bg='#f0f9ff',
                fg='#10b981'
            )
            title_label.pack(pady=(0, 5))
            
            # Message
            message_label = tk.Label(
                main_frame,
                text=message,
                font=('Arial', 12),
                bg='#f0f9ff',
                fg='#1f2937',
                wraplength=350,
                justify=tk.CENTER
            )
            message_label.pack(pady=(0, 15))
            
            # OK button
            ok_button = tk.Button(
                main_frame,
                text="OK",
                font=('Arial', 12, 'bold'),
                bg='#10b981',
                fg='white',
                relief='raised',
                bd=2,
                padx=20,
                pady=5,
                cursor='hand2',
                activebackground='#059669',
                activeforeground='white',
                command=success_window.destroy
            )
            ok_button.pack()
            
            # Auto-close after 3 seconds
            success_window.after(3000, success_window.destroy)
            
            # Focus on the window
            success_window.focus_set()
            
            # Ensure it doesn't interfere with main screen
            def cleanup_success_window():
                try:
                    if success_window.winfo_exists():
                        success_window.destroy()
                except:
                    pass
            
            # Store reference for cleanup
            if not hasattr(self, 'success_windows'):
                self.success_windows = []
            self.success_windows.append(success_window)
            
        except Exception as e:
            print(f"ERROR: Error showing green success message: {e}")
            # Fallback to regular messagebox
            messagebox.showinfo(title, message)
    
    def init_detection_system(self):
        """Initialize detection system"""
        try:
            self.current_model_path = "bsba male.pt"
            self.detection_system = None
            self.detection_active = False
            self.detection_thread = None
            self.violation_count = 0
            self.compliant_count = 0
            self.total_detections = 0
            print("SUCCESS: Detection system initialized")
        except Exception as e:
            print(f"ERROR: Failed to initialize detection system: {e}")
    
    def quit_application(self):
        """Quit the application with proper cleanup"""
        try:
            if messagebox.askyesno("Quit", "Are you sure you want to quit?"):
                self.cleanup_resources()
                self.root.quit()
        except Exception as e:
            print(f"ERROR: Error during quit: {e}")
            self.root.quit()
    
    def cleanup_resources(self):
        """Clean up all resources before exit"""
        if self.cleanup_done:
            return
        
        try:
            print("INFO: Cleaning up resources...")
            
            # Stop detection if active
            if self.detection_active:
                self.stop_detection()
            
            # Cleanup detection system
            if self.detection_system:
                self.detection_system.cleanup()
                self.detection_system = None
            
            # Release camera
            if self.cap:
                self.cap.release()
                self.cap = None
            
            # Close main screen
            if self.main_screen_window and self.main_screen_window.winfo_exists():
                self.main_screen_window.destroy()
                self.main_screen_window = None
            
            # Close any success windows
            if hasattr(self, 'success_windows'):
                for window in self.success_windows:
                    try:
                        if window.winfo_exists():
                            window.destroy()
                    except:
                        pass
                self.success_windows = []

            # Reset temporary visitor RFID assignments back to available on exit
            try:
                if hasattr(self, 'visitor_rfid_assignments') and isinstance(self.visitor_rfid_assignments, dict):
                    for rfid, assign in list(self.visitor_rfid_assignments.items()):
                        try:
                            # Only reset if still within our temporary assignment (we keep assignments until expiry normally)
                            expiry = assign.get('expiry_time')
                            reset_now = True
                            # If expiry exists and is still in future, allow resetting for test/dev exit
                            # (This makes the app put RFIDs back to the pool when the UI exits during testing)
                            if expiry:
                                from datetime import datetime
                                try:
                                    expiry_dt = datetime.strptime(expiry, "%Y-%m-%d %H:%M:%S")
                                    # If expiry already passed, it's already invalid; still reset
                                except Exception:
                                    pass

                            # Mark RFID available again
                            if rfid not in self.available_rfids:
                                self.available_rfids.append(rfid)
                            try:
                                self.update_rfid_availability_in_firebase(rfid, available=True)
                            except Exception:
                                pass

                            # Remove assignment from in-memory registry so next run treats it as empty
                            try:
                                if rfid in self.visitor_rfid_assignments:
                                    del self.visitor_rfid_assignments[rfid]
                            except Exception:
                                pass
                        except Exception:
                            pass
            except Exception:
                pass

            # CRITICAL: Return all assigned RFIDs from Firebase (not just in-memory)
            # This ensures RFIDs are returned even if UI was restarted
            try:
                # First, return RFIDs from in-memory assignments
                if hasattr(self, 'student_rfid_assignments') and isinstance(self.student_rfid_assignments, dict):
                    for rfid, assign in list(self.student_rfid_assignments.items()):
                        try:
                            if rfid not in self.available_rfids:
                                self.available_rfids.append(rfid)
                            try:
                                self.update_rfid_availability_in_firebase(rfid, available=True)
                            except Exception:
                                pass
                            try:
                                if rfid in self.student_rfid_assignments:
                                    del self.student_rfid_assignments[rfid]
                            except Exception:
                                pass
                        except Exception:
                            pass
                
                # Also query Firebase for all unavailable RFIDs and return them
                if self.firebase_initialized and self.db:
                    try:
                        # Check Empty RFID collection for unavailable RFIDs
                        empty_rfid_ref = self.db.collection('Empty RFID')
                        docs = empty_rfid_ref.get()
                        returned_count = 0
                        for doc in docs:
                            rfid_id = doc.id
                            rfid_data = doc.to_dict() or {}
                            
                            # Check if RFID is unavailable
                            is_unavailable = False
                            if 'assigned_to' in rfid_data and rfid_data['assigned_to']:
                                is_unavailable = True
                            elif 'available' in rfid_data and not rfid_data['available']:
                                is_unavailable = True
                            
                            # If unavailable, return it
                            if is_unavailable:
                                try:
                                    self.update_rfid_availability_in_firebase(rfid_id, available=True)
                                    returned_count += 1
                                    print(f"✅ Returned RFID {rfid_id} on cleanup")
                                except Exception as e:
                                    print(f"⚠️ Warning: Could not return RFID {rfid_id}: {e}")
                        
                        if returned_count > 0:
                            print(f"✅ Returned {returned_count} RFIDs from Firebase on cleanup")
                    except Exception as e:
                        print(f"⚠️ Warning: Could not query Firebase for unavailable RFIDs: {e}")
            except Exception:
                pass

            # Refresh RFID lists in UI
            try:
                if hasattr(self, 'load_rfid_from_firebase'):
                    self.load_rfid_from_firebase()
                if hasattr(self, 'update_student_rfid_list'):
                    self.update_student_rfid_list()
            except Exception:
                pass
            
            self.cleanup_done = True
            print("SUCCESS: Resources cleaned up successfully")
            
        except Exception as e:
            print(f"WARNING: Error during cleanup: {e}")
    
    def run(self):
        """Start the guard control center"""
        try:
            # Set up proper exit handling
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            
            # Center the window
            self.center_window()
            
            # Update the window to ensure it's properly rendered
            self.root.update()
            
            # Start the main loop
            print("INFO: Starting main application loop...")
            self.root.mainloop()
            
        except Exception as e:
            print(f"ERROR: Error running application: {e}")
            traceback.print_exc()
    
    def on_closing(self):
        """Handle window closing event"""
        try:
            self.cleanup_resources()
            self.root.destroy()
        except Exception as e:
            print(f"ERROR: Error during window close: {e}")
            self.root.destroy()
    
    def close(self):
        """Close the guard control center (legacy method)"""
        self.cleanup_resources()
        self.root.destroy()

if __name__ == "__main__":
    import signal
    import sys
    
    def signal_handler(signum, frame):
        """Handle system signals gracefully"""
        print(f"\n🛑 Received signal {signum} - shutting down gracefully...")
        try:
            if 'app' in locals():
                app.cleanup_resources()
        except:
            pass
        sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        print("🚀 Starting AI-niform Guard System...")
        print("=" * 50)
        
        # Initialize the application
        app = GuardMainControl()
        
        # Add some startup delay to ensure everything is ready
        time.sleep(0.5)
        
        print("SUCCESS: Application initialized successfully")
        print("🖥️ Starting GUI...")
        
        # Run the application
        app.run()
        
    except KeyboardInterrupt:
        print("\n🛑 Application interrupted by user")
    except Exception as e:
        print(f"ERROR: Application error: {e}")
        traceback.print_exc()
        print("\n🔧 Please check the error above and try again")
    finally:
        try:
            if 'app' in locals():
                app.cleanup_resources()
        except:
            pass
        print("👋 Application closed")
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                