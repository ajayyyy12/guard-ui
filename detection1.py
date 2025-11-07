import cv2
from ultralytics import YOLO
import time
import warnings
import os
from collections import defaultdict, deque
import numpy as np

# Optional torch acceleration
try:
    import torch
    TORCH_CUDA = torch.cuda.is_available()
except Exception:
    torch = None
    TORCH_CUDA = False
# Global model cache to avoid reload delays
MODEL_CACHE = {}


# Suppress OpenCV warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# BSCPE Uniform Requirements
BSCPE_REQUIRED_PARTS = {
    "BSCPE_MALE": [
        "black shoes",
        "gray polo",
        "rtw pants"
    ],
    "BSCPE_FEMALE": [
        "close shoes",
        "gray polo",
        "rtw pants"
    ],
    "BSCPE_FEMALE_SKIRT": [
        "close shoes",
        "gray polo",
        "gray skirt"
    ]
}

class BSCPEUniformTracker:
    """BSCPE uniform tracker with sequential verification: polo+pants twice, then shoes once"""
    
    def __init__(self, course_type="BSCPE_MALE"):
        self.course_type = course_type
        self.required_parts = BSCPE_REQUIRED_PARTS.get(course_type, [])
        self.temporary_detections = {}  # Track temporary detections with counts
        self.verification_required = True  # BSCPE requires verification
        self.verification_complete = False
        self.last_verification_time = 0
        self.start_time = time.time()
        self.timeout_duration = 10.0  # 10 seconds timeout for incomplete uniform
        self.permanent_green_items = set()  # Items that stay green once detected enough times
        self.standby_detected_items = set()  # Items in standby (polo+pants detected twice)
        self.sequential_phase = "waiting_polo_pants"  # Current verification phase
        
    def reset(self, course_type="BSCPE_MALE"):
        """Reset tracking for new detection session"""
        self.course_type = course_type
        self.required_parts = BSCPE_REQUIRED_PARTS.get(course_type, [])
        self.temporary_detections = {}
        self.verification_required = True
        self.verification_complete = False
        self.last_verification_time = 0
        self.start_time = time.time()
        self.permanent_green_items = set()
        self.standby_detected_items = set()
        self.sequential_phase = "waiting_polo_pants"
        
    def update_detections(self, detected_classes, current_time=None):
        """Update tracking with sequential verification: polo+pants twice, then shoes once"""
        if current_time is None:
            current_time = time.time()
            
        # Mark detections as temporary and track counts
        for class_name in detected_classes:
            if class_name in self.required_parts:
                if class_name not in self.temporary_detections:
                    self.temporary_detections[class_name] = {'count': 0, 'first_seen': current_time, 'last_seen': current_time}
                
                self.temporary_detections[class_name]['count'] += 1
                self.temporary_detections[class_name]['last_seen'] = current_time
        
        # Sequential verification logic
        if self.sequential_phase == "waiting_polo_pants":
            # Check if polo and pants are both detected twice
            polo_detected = self.temporary_detections.get("gray polo", {}).get('count', 0) >= 2
            pants_detected = self.temporary_detections.get("rtw pants", {}).get('count', 0) >= 2
            
            if polo_detected and pants_detected:
                self.standby_detected_items.add("gray polo")
                self.standby_detected_items.add("rtw pants")
                self.sequential_phase = "waiting_shoes"
                
        elif self.sequential_phase == "waiting_shoes":
            # Check if shoes are detected once
            shoes_detected = self.temporary_detections.get("black shoes", {}).get('count', 0) >= 1
            
            if shoes_detected:
                self.standby_detected_items.add("black shoes")
                self.verification_complete = True
                self.sequential_phase = "complete"
        
        return self.is_complete(), self.get_status_text()
        
    def is_complete(self):
        """Check if sequential verification is complete"""
        return self.sequential_phase == "complete"
        
    def get_status_text(self):
        """Get current status text with verification info - 10-second timeout for incomplete"""
        if self.verification_complete and self.is_complete():
            return "COMPLETE UNIFORM - ALL ITEMS VERIFIED"
        elif self.is_timeout():
            return "INCOMPLETE UNIFORM - 10 seconds elapsed"
        else:
            # Show progress
            progress_parts = []
            for part in self.required_parts:
                if part in self.standby_detected_items:
                    progress_parts.append(f"{part}: ✅")
                elif part in self.temporary_detections:
                    count = self.temporary_detections[part]['count']
                    required = 1 if part in ["black shoes", "close shoes"] else 2
                    progress_parts.append(f"{part}: {count}/{required}")
                else:
                    required = 1 if part in ["black shoes", "close shoes"] else 2
                    progress_parts.append(f"{part}: 0/{required}")
            
            return f"BSCPE VERIFICATION - {' | '.join(progress_parts)}"
            
    def is_timeout(self):
        """Check if 10 seconds have passed without completion"""
        elapsed = time.time() - self.start_time
        is_timeout = elapsed > self.timeout_duration
        return is_timeout
        
    def get_checklist_status(self):
        """Get detailed checklist status for sequential verification display"""
        checklist = {}
        for part in self.required_parts:
            required_count = 1 if part in ["black shoes", "close shoes"] else 2
            
            if part in self.standby_detected_items:
                # Item is in standby (permanently green)
                checklist[part] = {'status': 'complete', 'count': required_count, 'required': required_count}
            elif part in self.temporary_detections:
                count = self.temporary_detections[part]['count']
                if count >= required_count:
                    # Item meets requirement but not in standby yet
                    checklist[part] = {'status': 'ready', 'count': count, 'required': required_count}
                else:
                    # Item partially detected
                    checklist[part] = {'status': 'partial', 'count': count, 'required': required_count}
            else:
                # Item not detected
                checklist[part] = {'status': 'missing', 'count': 0, 'required': required_count}
        return checklist
        
    def get_verification_status(self):
        """Get detailed verification status with 10-second timeout for incomplete uniform"""
        if self.verification_complete:
            return "COMPLETE", "All uniform requirements verified and complete"
        elif self.is_timeout():
            return "INCOMPLETE", "Uniform verification incomplete - 10 seconds elapsed"
        else:
            return "PENDING", "Waiting for uniform verification"

class UniformDetectionService:
    """Detection service that can be used by the main UI"""
    
    def __init__(self, model_path="bsba male2.pt", conf_threshold=0.65):
        # Load or reuse cached model
        self.model = MODEL_CACHE.get(model_path) or YOLO(model_path)
        try:
            if model_path not in MODEL_CACHE:
                MODEL_CACHE[model_path] = self.model
        except Exception:
            pass

        # Move to GPU if available
        try:
            if TORCH_CUDA and hasattr(self.model, 'model'):
                self.model.model.to('cuda')
                print("✅ Model moved to CUDA")
        except Exception as e:
            print(f"⚠️ CUDA move failed or unavailable: {e}")
        self.conf_threshold = conf_threshold
        self.cap = None
        self.detection_active = False
        
        # Performance optimization settings
        self.frame_skip = 1  # Process every frame for lowest latency
        self.frame_count = 0
        self.last_process_time = 0
        self.process_interval = 0.03  # Process about every 30ms for responsiveness
        self.use_existing_camera = False
        
        # Camera health monitoring
        self.last_frame_time = 0
        self.freeze_threshold = 3.0  # Consider frozen if no frame for 3 seconds
        self.consecutive_failures = 0
        self.max_consecutive_failures = 5
        
        # BSCPE uniform verification - Initialize with default BSCPE tracker for testing
        self.bscpe_tracker = BSCPEUniformTracker("BSCPE_MALE")  # Always create BSCPE tracker
        self.current_course = "BSCPE"
        self.current_gender = "Male"
        self.current_person_name = ""  # Will be set when RFID is tapped
        
        # Track currently loaded model path for fast switching
        try:
            self._model_path = model_path
        except Exception:
            pass

        print(f"✅ Detection service initialized with model: {model_path}")
        print(f"✅ BSCPE tracker created by default: {self.bscpe_tracker is not None}")
        print(f"✅ BSCPE tracker phase: {self.bscpe_tracker.sequential_phase}")

    def preload_models(self, model_paths):
        """Preload and warm-up a set of models asynchronously to reduce first-use latency."""
        try:
            import threading
            def _preload():
                for p in model_paths:
                    if not p or not isinstance(p, str):
                        continue
                    if p in MODEL_CACHE:
                        continue
                    try:
                        print(f"🧠 Preloading model: {p}")
                        m = YOLO(p)
                        if TORCH_CUDA and hasattr(m, 'model'):
                            try:
                                m.model.to('cuda')
                            except Exception:
                                pass
                        # Warm-up with a tiny dummy image
                        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
                        _ = m(dummy, conf=self.conf_threshold)
                        MODEL_CACHE[p] = m
                        print(f"✅ Preloaded: {p}")
                    except Exception as e:
                        print(f"⚠️ Preload failed for {p}: {e}")
            threading.Thread(target=_preload, daemon=True).start()
        except Exception as e:
            print(f"⚠️ Failed to start preload thread: {e}")

    def switch_model(self, new_model_path):
        """Switch YOLO model in-place without rebuilding service/camera."""
        try:
            current = getattr(self, '_model_path', None)
            if current == new_model_path and self.model is not None:
                print(f"🔁 Model already active: {new_model_path}")
                return True
            print(f"🔄 Loading model in-place: {new_model_path}")
            # Use cache or load then cache
            new_model = MODEL_CACHE.get(new_model_path)
            if new_model is None:
                new_model = YOLO(new_model_path)
                try:
                    if TORCH_CUDA and hasattr(new_model, 'model'):
                        new_model.model.to('cuda')
                except Exception:
                    pass
                try:
                    MODEL_CACHE[new_model_path] = new_model
                except Exception:
                    pass
            self.model = new_model
            try:
                self._model_path = new_model_path
            except Exception:
                pass
            return True
        except Exception as e:
            print(f"❌ Failed to switch model to {new_model_path}: {e}")
            return False
    
    def set_existing_camera(self, camera_cap):
        """Set an existing camera instance instead of creating a new one"""
        try:
            print("🔄 Setting existing camera for detection service...")
            
            # Release any existing camera
            if self.cap and not self.use_existing_camera:
                self.cap.release()
                self.cap = None
            
            # Use the provided camera
            self.cap = camera_cap
            self.use_existing_camera = True
            # Initialize last_frame_time to now so health check doesn't think camera is frozen
            try:
                self.last_frame_time = time.time()
            except Exception:
                pass
            print("✅ Using existing camera for detection")
            return True
            
        except Exception as e:
            print(f"❌ Error setting existing camera: {e}")
            return False
    
    def check_camera_health(self):
        """Check if camera is healthy and not frozen"""
        current_time = time.time()
        # If we haven't yet received any frames (last_frame_time == 0), consider camera healthy
        # and allow the capture loop to set the timestamp on first successful read.
        try:
            if not self.last_frame_time:
                return True
        except Exception:
            pass

        # Check if camera is frozen (no frames for too long)
        if current_time - self.last_frame_time > self.freeze_threshold:
            print(f"⚠️ Camera appears frozen - no frames for {current_time - self.last_frame_time:.1f}s")
            return False
        
        # Check consecutive failures
        if self.consecutive_failures >= self.max_consecutive_failures:
            print(f"⚠️ Too many consecutive failures: {self.consecutive_failures}")
            return False
        
        return True
    
    def restart_camera(self):
        """Restart camera when it's frozen or failing"""
        try:
            print("🔄 Restarting camera due to health issues...")
            
            # Stop current camera
            if self.cap:
                self.cap.release()
                self.cap = None
            
            # Wait for cleanup
            time.sleep(0.5)
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Start fresh camera
            if self.start_camera():
                print("✅ Camera restarted successfully")
                self.consecutive_failures = 0
                self.last_frame_time = time.time()
                return True
            else:
                print("❌ Failed to restart camera")
                return False
                
        except Exception as e:
            print(f"❌ Error restarting camera: {e}")
            return False
    
    def release_camera_control(self):
        """Release camera control back to the original owner"""
        try:
            print("🔄 Releasing camera control...")
            # Return the VideoCapture instance to the caller so ownership can be returned
            returned_cap = self.cap
            self.use_existing_camera = False
            # Keep the underlying camera open but drop our reference
            self.cap = None
            print("✅ Camera control released")
            return returned_cap
        except Exception as e:
            print(f"❌ Error releasing camera control: {e}")
            return None
    
    def set_bscpe_verification(self, course, gender, person_name):
        """Set BSCPE verification parameters"""
        print(f"🔧 Setting BSCPE verification: {person_name} ({course}, {gender})")
        print(f"   Course: {course}")
        print(f"   Gender: {gender}")
        print(f"   Person: {person_name}")
        
        self.current_course = course
        self.current_gender = gender
        self.current_person_name = person_name
        
        # Determine uniform type
        if gender.upper() in ["MALE", "M"]:
            uniform_type = "BSCPE_MALE"
        elif gender.upper() in ["FEMALE", "F"]:
            uniform_type = "BSCPE_FEMALE"
        else:
            uniform_type = "BSCPE_MALE"  # Default
            
        # Initialize BSCPE tracker
        self.bscpe_tracker = BSCPEUniformTracker(uniform_type)
        print(f"🎓 BSCPE verification initialized for {person_name} ({course}, {gender})")
        print(f"   Required parts: {', '.join(self.bscpe_tracker.required_parts)}")
        print(f"   BSCPE tracker created: {self.bscpe_tracker is not None}")
        print(f"   Sequential phase: {self.bscpe_tracker.sequential_phase}")
    
    def clear_bscpe_verification(self):
        """Clear BSCPE verification"""
        self.bscpe_tracker = None
        self.current_course = None
        self.current_gender = None
        self.current_person_name = None
        print("🔄 BSCPE verification cleared")
    
    def start_camera(self, camera_index=0):
        """Start camera for detection with better backend handling"""
        try:
            print(f"🔍 Starting camera for detection service (index {camera_index})...")

            # If an external caller already provided a working VideoCapture via set_existing_camera,
            # prefer to reuse it instead of force-releasing and reopening the device. This
            # avoids toggling the physical device (LED off) and prevents device contention.
            if getattr(self, 'use_existing_camera', False) and getattr(self, 'cap', None) is not None:
                try:
                    print("🔁 Using existing VideoCapture (provided by main UI) — testing read")
                    # Test read a frame to ensure the handed cap is usable
                    ret, test_frame = self.cap.read()
                    if ret and test_frame is not None:
                        try:
                            self.last_frame_time = time.time()
                        except Exception:
                            pass
                        print("✅ Existing VideoCapture is usable for detection")
                        return True
                    else:
                        print("⚠️ Existing VideoCapture provided but failed to read a test frame")
                        # Fall through to a normal restart/open path
                except Exception as e:
                    print(f"⚠️ Error testing existing VideoCapture: {e}")

            # First, ensure any existing camera is released
            self.force_camera_release()

            # Reduced wait time for faster camera startup
            time.sleep(0.1)

            # Try different camera backends in order of preference for Windows
            backends = [
                (cv2.CAP_ANY, "Any available"),
                (cv2.CAP_DSHOW, "DirectShow") if hasattr(cv2, 'CAP_DSHOW') else (cv2.CAP_ANY, "Any available"),
                (cv2.CAP_MSMF, "Media Foundation") if hasattr(cv2, 'CAP_MSMF') else (cv2.CAP_ANY, "Any available"),
            ]

            camera_opened = False
            for backend, name in backends:
                try:
                    print(f"🔍 Trying camera with {name} backend...")
                    self.cap = cv2.VideoCapture(camera_index, backend)

                    if self.cap and self.cap.isOpened():
                        # Test if we can actually read a frame
                        ret, test_frame = self.cap.read()
                        if ret and test_frame is not None:
                            print(f"✅ Camera opened successfully with {name} backend")
                            camera_opened = True
                            break
                        else:
                            print(f"⚠️ Camera opened but can't read frames with {name}")
                            try:
                                self.cap.release()
                            except Exception:
                                pass
                            self.cap = None
                    else:
                        print(f"❌ Camera failed to open with {name}")
                        try:
                            if self.cap:
                                self.cap.release()
                        except Exception:
                            pass
                        self.cap = None
                except Exception as e:
                    print(f"❌ Error with {name}: {e}")
                    try:
                        if self.cap:
                            self.cap.release()
                    except Exception:
                        pass
                    self.cap = None

            # If all backends failed, try one final attempt without specifying backend
            if not camera_opened:
                try:
                    print(f"🔍 Last attempt: Opening camera with default backend...")
                    self.cap = cv2.VideoCapture(camera_index)
                    if self.cap and self.cap.isOpened():
                        ret, test_frame = self.cap.read()
                        if ret and test_frame is not None:
                            print(f"✅ Camera opened with default backend")
                            camera_opened = True
                        else:
                            print(f"⚠️ Camera opened but can't read frames with default backend")
                            try:
                                self.cap.release()
                            except Exception:
                                pass
                            self.cap = None
                    else:
                        try:
                            if self.cap:
                                self.cap.release()
                        except Exception:
                            pass
                        self.cap = None
                except Exception as e:
                    print(f"❌ Final attempt failed: {e}")
                    try:
                        if self.cap:
                            self.cap.release()
                    except Exception:
                        pass
                    self.cap = None

            if not camera_opened:
                print("❌ Cannot open any camera for detection")
                return False

            # Set camera properties for better performance and low-latency buffer
            try:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                if hasattr(cv2, 'CAP_PROP_BUFFERSIZE'):
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                print("✅ Camera properties set")
            except Exception as e:
                print(f"⚠️ Could not set camera properties: {e}")

            # Initialize last_frame_time to now so health checks don't trigger immediately
            try:
                self.last_frame_time = time.time()
            except Exception:
                pass

            print("✅ Camera opened for detection service.")
            return True

        except Exception as e:
            print(f"❌ Error starting camera: {e}")
            return False
    
    def reset_for_new_detection(self):
        """Reset detection service for new detection session"""
        try:
            print("🔄 Resetting detection service for new detection...")
            
            # Stop any active detection
            self.stop_detection()
            
            # Force camera release with multiple attempts
            self.force_camera_release()
            
            # Reset BSCPE tracker
            if self.bscpe_tracker:
                self.bscpe_tracker.reset()
                print("✅ BSCPE tracker reset")
            
            # Clear current tracking data
            self.current_course = ""
            self.current_gender = ""
            self.current_person_name = ""
            self.current_rfid = ""
            
            print("✅ Detection service reset complete")
            
        except Exception as e:
            print(f"❌ Error resetting detection service: {e}")
    
    def force_camera_release(self):
        """Force camera release with multiple attempts"""
        try:
            print("🔄 Force releasing camera...")
            
            # Fast single release attempt (removed multiple attempts for speed)
            if self.cap:
                try:
                    self.cap.release()
                    print(f"✅ Camera released")
                except Exception as e:
                    print(f"⚠️ Camera release failed: {e}")
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear the cap reference
            self.cap = None
            
            print("✅ Camera force release complete")
            
        except Exception as e:
            print(f"❌ Error in force camera release: {e}")
    
    def stop_camera(self):
        """Stop camera"""
        if self.cap:
            self.cap.release()
            self.cap = None
        print("✅ Camera stopped.")
    
    def detect_frame(self, frame):
        """Detect uniform components in a single frame - clean output with only bounding boxes and class names"""
        if self.model is None:
            print("⚠️ Model is None in detect_frame")
            return None, []

        # Resize frame for faster inference
        try:
            h, w = frame.shape[:2]
            target_w = 640
            if w > target_w:
                scale = target_w / float(w)
                new_h = max(320, int(h * scale))
                resized = cv2.resize(frame, (target_w, new_h))
            else:
                resized = frame
        except Exception:
            resized = frame

        # Perform detection on resized frame
        results = self.model(resized, conf=self.conf_threshold)
        
        # Get detected classes
        detected_classes = []
        if results and len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                if conf >= self.conf_threshold:
                    class_name = results[0].names.get(cls, f"class_{cls}")
                    detected_classes.append({
                        'class_name': class_name,
                        'confidence': float(conf),
                        'class_id': cls
                    })
        
        # Create clean annotated frame with only bounding boxes and class names
        annotated_frame = resized.copy()
        if results and len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                conf = box.conf[0].cpu().numpy()
                if conf >= self.conf_threshold:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Get class name
                    cls = int(box.cls[0].cpu().numpy())
                    class_name = results[0].names.get(cls, f"class_{cls}")
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw class name above the box
                    cv2.putText(annotated_frame, class_name, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Add detected classes display in top left corner (always visible)
        annotated_frame = self._add_detected_classes_display(annotated_frame, detected_classes)
        
        # BSCPE overlay disabled - showing requirements in UI panel instead
        # No overlay added to keep camera feed clean
        
        return annotated_frame, detected_classes
    
    def _add_detected_classes_display(self, frame, detected_classes):
        """Add detected classes display in top left corner with persistent status"""
        try:
            # Get frame dimensions
            frame_height, frame_width = frame.shape[:2]
            
            # Create semi-transparent background for detected classes
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (350, 120), (0, 0, 0), -1)  # Black background - made bigger
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Add border
            cv2.rectangle(frame, (10, 10), (350, 120), (0, 255, 0), 2)
            
            # Add title
            cv2.putText(frame, "DETECTED CLASSES:", (15, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show persistent status if BSCPE tracker is active
            if self.bscpe_tracker:
                checklist = self.bscpe_tracker.get_checklist_status()
                y_pos = 50
                
                for part, status_info in checklist.items():
                    if status_info['status'] == 'complete':
                        # Green checkmark for completed items (persistent)
                        text = f"✅ {part} ✓"
                        color = (0, 255, 0)  # Green
                    elif status_info['status'] == 'ready':
                        # Blue for ready items (detected enough but waiting for phase)
                        text = f"🔵 {part}: {status_info['count']}/{status_info['required']}"
                        color = (255, 0, 255)  # Magenta
                    elif status_info['status'] == 'partial':
                        # Yellow for partially detected items
                        text = f"🟡 {part}: {status_info['count']}/{status_info['required']}"
                        color = (0, 255, 255)  # Yellow
                    else:
                        # White for missing items
                        text = f"⚪ {part}: 0/{status_info['required']}"
                        color = (255, 255, 255)  # White
                    
                    cv2.putText(frame, text, (15, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    y_pos += 15
                    
                # Add current frame detections below
                if detected_classes:
                    cv2.putText(frame, "Current frame:", (15, y_pos + 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                    y_pos += 20
                    
                    class_names = [d['class_name'] for d in detected_classes]
                    confidences = [f"{d['confidence']:.2f}" for d in detected_classes]
                    
                    for i, (class_name, conf) in enumerate(zip(class_names, confidences)):
                        if i < 2:  # Show max 2 current detections
                            text = f"• {class_name} ({conf})"
                            cv2.putText(frame, text, (15, y_pos), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                            y_pos += 12
            else:
                # Fallback to showing current frame detections only
                if detected_classes:
                    class_names = [d['class_name'] for d in detected_classes]
                    confidences = [f"{d['confidence']:.2f}" for d in detected_classes]
                    
                    # Display classes with confidence scores
                    for i, (class_name, conf) in enumerate(zip(class_names, confidences)):
                        if i < 3:  # Show max 3 classes to fit in the box
                            text = f"• {class_name} ({conf})"
                            cv2.putText(frame, text, (15, 50 + i * 15), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                else:
                    cv2.putText(frame, "• No classes detected", (15, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
            
            return frame
            
        except Exception as e:
            print(f"ERROR: Failed to add detected classes display: {e}")
            return frame
    
    def _add_bscpe_overlay(self, frame, detected_classes):
        """Add BSCPE verification status overlay to camera frame"""
        try:
            if not self.bscpe_tracker:
                print("⚠️ BSCPE tracker not available for overlay")
                return frame
            
            
            # Update BSCPE tracker with detected classes
            class_names = [d['class_name'] for d in detected_classes]
            self.bscpe_tracker.update_detections(class_names)
            
            # Get checklist status
            checklist = self.bscpe_tracker.get_checklist_status()
            verification_status, verification_message = self.bscpe_tracker.get_verification_status()
            
            # Set overlay colors based on status
            if verification_status == "COMPLETE":
                color = (0, 255, 0)  # Green
                bg_color = (0, 100, 0)  # Dark green background
            elif verification_status == "INCOMPLETE":
                color = (0, 0, 255)  # Red
                bg_color = (100, 0, 0)  # Dark red background
            else:  # PENDING
                color = (0, 165, 255)  # Orange
                bg_color = (100, 50, 0)  # Dark orange background
            
            # Add background rectangle - larger to accommodate student name and retry message
            cv2.rectangle(frame, (10, 10), (450, 200), bg_color, -1)
            cv2.rectangle(frame, (10, 10), (450, 200), color, 2)
            
            # Add title only
            cv2.putText(frame, "BSCPE UNIFORM CHECKLIST", (20, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Add student information
            cv2.putText(frame, f"Student: {self.current_person_name}", (20, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add sequential phase indicator
            if self.bscpe_tracker.sequential_phase == "waiting_polo_pants":
                cv2.putText(frame, "PHASE 1: Waiting for polo+pants (2x each)", (20, 75), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                y_pos = 100
            elif self.bscpe_tracker.sequential_phase == "waiting_shoes":
                cv2.putText(frame, "PHASE 2: Waiting for shoes (1x)", (20, 75), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                y_pos = 100
            else:
                y_pos = 75
            
            # Add checklist items
            for part, status_info in checklist.items():
                if status_info['status'] == 'complete':
                    # Green checkmark for standby detected items
                    item_text = f"✅ {part} ✓ (STANDBY)"
                    text_color = (0, 255, 0)  # Green
                elif status_info['status'] == 'ready':
                    # Blue for ready items (detected enough but waiting for phase)
                    item_text = f"🔵 {part}: {status_info['count']}/{status_info['required']} (READY)"
                    text_color = (255, 0, 255)  # Magenta
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
            
            # Add prominent COMPLETE status if all items are complete
            if self.bscpe_tracker.sequential_phase == "complete":
                # Add large COMPLETE text
                cv2.putText(frame, "🎉 COMPLETE UNIFORM! 🎉", (20, y_pos + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, "All BSCPE requirements verified!", (20, y_pos + 45), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            elif verification_status == "INCOMPLETE":
                # Add incomplete uniform message with retry instruction
                cv2.putText(frame, "❌ INCOMPLETE UNIFORM", (20, y_pos + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, "Uniform verification incomplete!", (20, y_pos + 45), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(frame, "💡 TAP YOUR ID AGAIN TO RETRY", (20, y_pos + 65), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            else:
                # Add timer if pending
                elapsed_time = time.time() - self.bscpe_tracker.start_time
                remaining_time = max(0, self.bscpe_tracker.timeout_duration - elapsed_time)
                timer_text = f"Time remaining: {remaining_time:.1f}s"
                cv2.putText(frame, timer_text, (20, y_pos + 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            return frame
            
        except Exception as e:
            print(f"ERROR: Failed to add BSCPE overlay: {e}")
            return frame
    
    def get_detection_loop(self):
        """Generator that yields detection results for each frame - optimized for performance"""
        if not self.cap or not self.cap.isOpened():
            print("❌ Camera not available for detection loop.")
            return
        
        self.detection_active = True
        print("🔍 Starting optimized detection loop...")
        print(f"🔍 Camera opened: {self.cap.isOpened()}")
        print(f"🔍 Model loaded: {self.model is not None}")
        print(f"🔍 BSCPE tracker: {self.bscpe_tracker}")
        print(f"🔍 Frame skip: {self.frame_skip} (processing every {self.frame_skip} frames)")
        print(f"🔍 Process interval: {self.process_interval}s")
        
        consecutive_failures = 0
        max_failures = 10  # Stop after 10 consecutive failures
        
        while self.detection_active:
            current_time = time.time()
            
            # Check camera health before processing
            if not self.check_camera_health():
                print("🔄 Camera health check failed - attempting restart...")
                if not self.restart_camera():
                    print("❌ Camera restart failed - stopping detection")
                    break
                continue
            
            # Check if we should process this frame (time-based throttling)
            if current_time - self.last_process_time < self.process_interval:
                time.sleep(0.01)  # Small sleep to prevent busy waiting
                continue
            
            ret, frame = self.cap.read()
            if not ret:
                self.consecutive_failures += 1
                if self.consecutive_failures <= 3:  # Only print first 3 failures
                    print(f"⚠️ Failed to grab frame (attempt {self.consecutive_failures})")
                elif self.consecutive_failures == 4:
                    print("⚠️ Multiple frame grab failures - camera may be in use by another app")
                elif self.consecutive_failures >= self.max_consecutive_failures:
                    print("❌ Too many consecutive failures - attempting camera restart")
                    if not self.restart_camera():
                        print("❌ Camera restart failed - stopping detection")
                        break
                    continue
                time.sleep(0.1)  # Wait before retrying
                continue
            else:
                self.consecutive_failures = 0  # Reset on success
                self.last_frame_time = current_time  # Update frame time
            
            # Frame skipping for performance
            self.frame_count += 1
            if self.frame_count % self.frame_skip != 0:
                # Skip this frame, just yield the original frame
                yield {
                    'frame': frame,
                    'annotated_frame': frame,  # No processing, just pass through
                    'detected_classes': [],
                    'timestamp': current_time,
                    'skipped': True
                }
                continue
            
            # Perform detection only on selected frames
            annotated_frame, detected_classes = self.detect_frame(frame)
            self.last_process_time = current_time
            
            yield {
                'frame': frame,
                'annotated_frame': annotated_frame,
                'detected_classes': detected_classes,
                'timestamp': current_time,
                'skipped': False
            }
    
    def stop_detection(self):
        """Stop the detection loop"""
        self.detection_active = False
        print("🛑 Detection stopped.")

# Global detection service instance
detection_service = None

def get_detection_service():
    """Get or create the global detection service"""
    global detection_service
    if detection_service is None:
        detection_service = UniformDetectionService()
    return detection_service

def test_model(model_path, model_name="Model"):
    """Test a specific model"""
    print(f"🧪 Testing {model_name} with model: {model_path}")
    print("=" * 50)
    
    try:
        service = UniformDetectionService(model_path=model_path, conf_threshold=0.65)
        
        if not service.start_camera():
            print("❌ Failed to start camera")
            return False
        
        print("✅ Camera opened. Press 'q' to quit.")
        print(f"🔍 Using {model_name}: {model_path}")
        
        frame_count = 0
        for detection_result in service.get_detection_loop():
            frame_count += 1
            
            # Show output
            cv2.imshow(f"AI-Niform Detection - {model_name}", detection_result['annotated_frame'])
            
            # Print detected classes every 30 frames
            if detection_result['detected_classes'] and frame_count % 30 == 0:
                class_names = [d['class_name'] for d in detection_result['detected_classes']]
                confidences = [f"{d['confidence']:.2f}" for d in detection_result['detected_classes']]
                print(f"Frame {frame_count}: Detected {class_names} (conf: {confidences})")
            
            # Quit with 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        return True
        
    except Exception as e:
        print(f"❌ Error testing {model_name}: {e}")
        return False
    finally:
        service.stop_detection()
        service.stop_camera()
        cv2.destroyAllWindows()

def test_bscpe_verification():
    """Test BSCPE verification system"""
    print("🧪 Testing BSCPE verification system")
    print("=" * 50)
    
    # Create detection service
    service = UniformDetectionService(model_path="bsba male2.pt", conf_threshold=0.65)
    
    # Set BSCPE verification
    service.set_bscpe_verification("BSCPE", "Male", "Test Student")
    
    # Test with some detected classes
    test_classes = ["gray polo", "rtw pants", "black shoes"]
    print(f"\n🔍 Testing with classes: {test_classes}")
    
    # Simulate detections
    for i in range(3):
        print(f"\n--- Detection Round {i+1} ---")
        is_complete, status = service.bscpe_tracker.update_detections(test_classes)
        print(f"Status: {status}")
        print(f"Complete: {is_complete}")
        
        # Show checklist
        checklist = service.bscpe_tracker.get_checklist_status()
        for part, info in checklist.items():
            print(f"  {part}: {info['status']} ({info['count']}/{info['required']})")
    
    print("\n✅ BSCPE verification test completed")
    return True

def test_bscpe_detection_with_camera():
    """Test BSCPE detection with camera to debug the issue"""
    print("🧪 Testing BSCPE Detection with Camera")
    print("=" * 50)
    
    # Create detection service
    service = UniformDetectionService("ict and eng.pt")  # Use ICT model for BSCPE
    
    # Set BSCPE verification
    service.set_bscpe_verification("BSCPE", "Male", "Test Student")
    
    # Start camera
    if not service.start_camera(0):
        print("❌ Failed to start camera")
        return
    
    print("✅ Camera started - Press 'q' to quit")
    print("🎯 BSCPE uniform checklist should appear on camera feed")
    print("🔍 Look for debug output in console")
    
    try:
        for detection_result in service.get_detection_loop():
            frame = detection_result['frame']
            annotated_frame = detection_result['annotated_frame']
            detected_classes = detection_result['detected_classes']
            
            # Show the frame
            cv2.imshow("BSCPE Detection Test", annotated_frame)
            
            # Print detected classes
            if detected_classes:
                class_names = [d['class_name'] for d in detected_classes]
                print(f"🔍 Detected: {class_names}")
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        service.stop_camera()
        cv2.destroyAllWindows()
        print("✅ BSCPE detection test completed")

def main():
    """Main function with model selection"""
    import sys
    
    print("🚀 AI-Niform Detection Test")
    print("=" * 50)
    
    # Available models
    models = {
        "1": ("bsba male2.pt", "BSBA Male"),
        "2": ("bsba_female.pt", "BSBA Female"),
        "3": ("bsba_male.pt", "BSBA Male (Original)"),
    # ICT/ENG combined model (supports male and female classes inside the same model)
    "4": ("ict and eng.pt", "ICT / ENG (Male+Female)"),
    # Arts & Science combined model
    "5": ("arts and science.pt", "Arts & Science (Male+Female)")
    ,
    # Tourism male model
    "6": ("tourism male.pt", "Tourism Male")
    }
    # Add HM (honors/module) model option if present
    models["7"] = ("hm.pt", "HM Model")
    # Add SHS (Senior High School) model
    models["8"] = ("shs.pt", "Senior High (SHS)")
    # Add BSCPE test options
    models["9"] = ("BSCPE_TEST", "BSCPE Verification Test")
    models["10"] = ("BSCPE_CAMERA_TEST", "BSCPE Detection Test with Camera")
    
    print("Available models:")
    for key, (path, name) in models.items():
        print(f"  {key}. {name} ({path})")
    
    # Get user choice
    choice = input("\nSelect model (1-10) or press Enter for default (BSBA Male): ").strip()
    
    if choice in models:
        model_path, model_name = models[choice]
    else:
        model_path, model_name = models["1"]  # Default to male model
    
    # Handle BSCPE tests
    if model_path == "BSCPE_TEST":
        success = test_bscpe_verification()
    elif model_path == "BSCPE_CAMERA_TEST":
        test_bscpe_detection_with_camera()
    else:
        # Check if model file exists
        import os
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            print("Available files in current directory:")
            for file in os.listdir("."):
                if file.endswith(".pt"):
                    print(f"  - {file}")
            return
        
        # Test the selected model
        success = test_model(model_path, model_name)
    
    if success:
        print(f"\n✅ {model_name} test completed successfully!")
    else:
        print(f"\n❌ {model_name} test failed!")

if __name__ == "__main__":
    main()