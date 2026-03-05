import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import FaceLandmarkerOptions, RunningMode
import numpy as np
import pyautogui
import os
import subprocess
from collections import deque
import time
import urllib.request

# ──────────────────────────────────────────────────────────────────────────────
# Auto-download the model if missing
# ──────────────────────────────────────────────────────────────────────────────
MODEL_PATH = "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading face_landmarker.task model (~30MB), please wait...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")
    else:
        print("Model file found.")


# ──────────────────────────────────────────────────────────────────────────────
class EyeTracker:
    def __init__(self):
        ensure_model()

        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.7,
            min_face_presence_confidence=0.7,
            min_tracking_confidence=0.7,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
        print("FaceLandmarker created successfully.")

        # ── landmark indices ──────────────────────────────────────────────────
        self.LEFT_IRIS  = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        self.LEFT_EYE   = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE  = [33,  160, 158, 133, 153, 144]
        self.LEFT_BROW  = [70,  63,  105]
        self.RIGHT_BROW = [336, 296, 334]
        self.MOUTH_TOP    = 13
        self.MOUTH_BOTTOM = 14

        self.screen_width, self.screen_height = pyautogui.size()
        pyautogui.FAILSAFE = False

        self.position_buffer_x = deque([0.5] * 5, maxlen=5)
        self.position_buffer_y = deque([0.5] * 5, maxlen=5)

        self.center_offset_x         = 0
        self.center_offset_y         = 0
        self.movement_scale          = 2.5
        self.horizontal_scale_factor = 1.2

        self.last_action_time = time.time()

        # Eyebrow tracking
        self.brow_raises    = 0
        self.last_brow_time = 0
        self.brow_cooldown  = 2.0

        # Long-blink
        self.blink_start_time    = 0
        self.is_blinking         = False
        self.long_blink_duration = 1.5

        # Mouth open
        self.mouth_open_start     = 0
        self.is_mouth_open        = False
        self.mouth_open_duration  = 1.2
        self.mouth_open_threshold = 0.1

        self.last_position        = (self.screen_width / 2, self.screen_height / 2)
        self.initialized          = False
        self.init_frames          = 0
        self.init_required_frames = 10
        self.init_positions       = []

        self.zoom_cooldown    = 1.0
        self.keyboard_open    = False
        self.keyboard_cooldown = 10.0

        # Wall-clock start for timestamps (MediaPipe VIDEO mode needs ms, always increasing)
        self._start_time = time.time()

    # ── timestamp ─────────────────────────────────────────────────────────────
    def _get_ts_ms(self):
        return int((time.time() - self._start_time) * 1000)

    # ── calibration ───────────────────────────────────────────────────────────
    def initialize_center_position(self, x, y):
        self.init_positions.append((x, y))
        self.init_frames += 1
        if self.init_frames >= self.init_required_frames:
            xs, ys = zip(*self.init_positions)
            self.center_offset_x = float(np.median(xs))
            self.center_offset_y = float(np.median(ys))
            self.initialized = True
            print(f"Initialized: center = ({self.center_offset_x:.3f}, {self.center_offset_y:.3f})")

    # ── eye geometry ──────────────────────────────────────────────────────────
    def get_relative_eye_position(self, iris_points, eye_points):
        iris_center = np.mean(iris_points, axis=0)
        eye_arr     = np.array(eye_points)

        eye_left   = float(np.min(eye_arr[:, 0]))
        eye_right  = float(np.max(eye_arr[:, 0]))
        eye_top    = float(np.min(eye_arr[:, 1]))
        eye_bottom = float(np.max(eye_arr[:, 1]))

        eye_width  = max(eye_right  - eye_left, 1)
        eye_height = max(eye_bottom - eye_top,  1)

        rel_x = (iris_center[0] - eye_left) / eye_width
        rel_y = (iris_center[1] - eye_top)  / eye_height
        return rel_x, rel_y, iris_center

    def calculate_cursor_position(self, left_rel, right_rel):
        avg_x = (left_rel[0] + right_rel[0]) / 2
        avg_y = (left_rel[1] + right_rel[1]) / 2

        self.position_buffer_x.append(avg_x)
        self.position_buffer_y.append(avg_y)

        smooth_x = sum(self.position_buffer_x) / len(self.position_buffer_x)
        smooth_y = sum(self.position_buffer_y) / len(self.position_buffer_y)

        if not self.initialized:
            self.initialize_center_position(smooth_x, smooth_y)
            return self.last_position

        offset_x = (smooth_x - self.center_offset_x) * self.horizontal_scale_factor
        offset_y =  smooth_y - self.center_offset_y

        cursor_x = self.screen_width  / 2 + offset_x * self.movement_scale * self.screen_width
        cursor_y = self.screen_height / 2 + offset_y * self.movement_scale * self.screen_height

        cursor_x = max(0, min(self.screen_width,  cursor_x))
        cursor_y = max(0, min(self.screen_height, cursor_y))

        cursor_x = 0.7 * cursor_x + 0.3 * self.last_position[0]
        cursor_y = 0.7 * cursor_y + 0.3 * self.last_position[1]

        self.last_position = (cursor_x, cursor_y)
        return cursor_x, cursor_y

    def calculate_ear(self, eye_points):
        pts = np.array(eye_points, dtype=float)
        A   = np.linalg.norm(pts[1] - pts[5])
        B   = np.linalg.norm(pts[2] - pts[4])
        C   = np.linalg.norm(pts[0] - pts[3])
        if C == 0:
            return 0.3
        return (A + B) / (2.0 * C)

    def detect_blinks_and_winks(self, left_eye, right_eye):
        left_ear  = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)

        blink_threshold = 0.19
        wink_threshold  = 0.18
        open_threshold  = 0.23

        blink      = left_ear  < blink_threshold and right_ear < blink_threshold
        left_wink  = left_ear  < wink_threshold  and right_ear > open_threshold
        right_wink = right_ear < wink_threshold  and left_ear  > open_threshold

        now = time.time()
        if blink and not self.is_blinking:
            self.is_blinking      = True
            self.blink_start_time = now
        elif not blink:
            self.is_blinking = False

        long_blink = self.is_blinking and (now - self.blink_start_time) > self.long_blink_duration
        return blink, left_wink, right_wink, left_ear, right_ear, long_blink

    def detect_eyebrow_raise(self, mesh, img_h):
        left_brow_y  = float(np.mean([mesh[i][1] for i in self.LEFT_BROW]))
        right_brow_y = float(np.mean([mesh[i][1] for i in self.RIGHT_BROW]))
        left_eye_y   = float(np.mean([mesh[i][1] for i in self.LEFT_EYE]))
        right_eye_y  = float(np.mean([mesh[i][1] for i in self.RIGHT_EYE]))

        threshold   = img_h * 0.06
        both_raised = (
            (left_eye_y  - left_brow_y  > threshold) and
            (right_eye_y - right_brow_y > threshold)
        )
        return both_raised, (left_eye_y - left_brow_y), (right_eye_y - right_brow_y)

    def detect_mouth_open(self, mesh, img_h):
        top_lip    = mesh[self.MOUTH_TOP][1]
        bottom_lip = mesh[self.MOUTH_BOTTOM][1]
        ratio      = (bottom_lip - top_lip) / (img_h * 0.7)
        is_open    = ratio > self.mouth_open_threshold

        now = time.time()
        if is_open and not self.is_mouth_open:
            self.is_mouth_open    = True
            self.mouth_open_start = now
        elif not is_open:
            self.is_mouth_open = False

        long_open = self.is_mouth_open and (now - self.mouth_open_start) > self.mouth_open_duration
        return long_open, ratio

    # ── virtual keyboard ──────────────────────────────────────────────────────
    def open_virtual_keyboard(self):
        now = time.time()
        if self.keyboard_open and (now - self.last_action_time < self.keyboard_cooldown):
            return False
        try:
            if os.name == 'nt':
                os.system('start osk')
            else:
                for cmd in (['onboard'], ['florence'], ['kvkbd'],
                            ['open', '-a', 'Keyboard Viewer']):
                    try:
                        subprocess.Popen(cmd)
                        break
                    except FileNotFoundError:
                        continue
            self.keyboard_open    = True
            self.last_action_time = now
            return True
        except Exception as e:
            print(f"Keyboard error: {e}")
            return False

    def keyboard_shortcut_method(self):
        try:
            pyautogui.hotkey('win', 'ctrl', 'o')
            self.keyboard_open = True
            return True
        except Exception:
            return False

    def zoom_in(self):  pyautogui.hotkey('ctrl', '+')
    def zoom_out(self): pyautogui.hotkey('ctrl', '-')

    # ── drawing ───────────────────────────────────────────────────────────────
    def draw_iris_tracking(self, frame, left_iris, right_iris,
                           left_eye, right_eye, left_center, right_center):
        cv2.polylines(frame, [np.array(left_eye)],   True, (0, 255, 0), 1)
        cv2.polylines(frame, [np.array(right_eye)],  True, (0, 255, 0), 1)
        cv2.polylines(frame, [np.array(left_iris)],  True, (255, 0, 0), 1)
        cv2.polylines(frame, [np.array(right_iris)], True, (255, 0, 0), 1)
        cv2.circle(frame, (int(left_center[0]),  int(left_center[1])),  2, (0, 0, 255), -1)
        cv2.circle(frame, (int(right_center[0]), int(right_center[1])), 2, (0, 0, 255), -1)
        for pts, ctr in [(left_eye, left_center), (right_eye, right_center)]:
            mn_x = min(p[0] for p in pts); mx_x = max(p[0] for p in pts)
            mn_y = min(p[1] for p in pts); mx_y = max(p[1] for p in pts)
            cx, cy = int(ctr[0]), int(ctr[1])
            cv2.line(frame, (mn_x, cy), (mx_x, cy), (255, 255, 0), 1)
            cv2.line(frame, (cx, mn_y), (cx, mx_y), (255, 255, 0), 1)
        return frame

    # ── main per-frame logic ──────────────────────────────────────────────────
    def process_frame(self, frame):
        img_h, img_w = frame.shape[:2]

        # Must be C-contiguous RGB for mp.Image
        frame_rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result   = self.face_landmarker.detect_for_video(mp_image, self._get_ts_ms())
        except Exception as e:
            cv2.putText(frame, f"Detection error: {e}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            return frame

        if not result.face_landmarks:
            cv2.putText(frame, "No face detected — move closer / improve lighting",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return frame

        raw         = result.face_landmarks[0]
        mesh_points = np.array(
            [[int(lm.x * img_w), int(lm.y * img_h)] for lm in raw],
            dtype=np.int32
        )
        total_lm = len(mesh_points)

        def safe_pts(indices):
            return [mesh_points[i] for i in indices if i < total_lm]

        left_iris  = safe_pts(self.LEFT_IRIS)
        right_iris = safe_pts(self.RIGHT_IRIS)
        left_eye   = safe_pts(self.LEFT_EYE)
        right_eye  = safe_pts(self.RIGHT_EYE)

        if len(left_iris) < 4 or len(right_iris) < 4 or \
           len(left_eye)  < 6 or len(right_eye)  < 6:
            cv2.putText(frame,
                        f"Only {total_lm} landmarks found (need 478 with iris). "
                        "Re-download face_landmarker.task",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            return frame

        rel_lx, rel_ly, left_center  = self.get_relative_eye_position(left_iris,  left_eye)
        rel_rx, rel_ry, right_center = self.get_relative_eye_position(right_iris, right_eye)

        frame = self.draw_iris_tracking(frame, left_iris, right_iris,
                                        left_eye, right_eye, left_center, right_center)

        cursor_x, cursor_y = self.calculate_cursor_position(
            (rel_lx, rel_ly), (rel_rx, rel_ry)
        )

        if self.initialized:
            pyautogui.moveTo(int(cursor_x), int(cursor_y))

            blink, left_wink, right_wink, left_ear, right_ear, long_blink = \
                self.detect_blinks_and_winks(left_eye, right_eye)
            mouth_open, mouth_ratio = self.detect_mouth_open(mesh_points, img_h)
            eyebrows_raised, left_dist, right_dist = \
                self.detect_eyebrow_raise(mesh_points, img_h)
            now = time.time()

            # HUD
            cv2.putText(frame, f"L-EAR:  {left_ear:.2f}",    (10, img_h - 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(frame, f"R-EAR:  {right_ear:.2f}",   (10, img_h - 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(frame, f"Mouth:  {mouth_ratio:.3f}", (10, img_h - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(frame, f"L-Brow: {left_dist:.1f}",   (10, img_h - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(frame, f"LM: {total_lm}",            (10, img_h - 125),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

            # Long blink → keyboard
            if long_blink and now - self.last_action_time > 2.0:
                self.open_virtual_keyboard() or self.keyboard_shortcut_method()
                cv2.putText(frame, "Opening Keyboard (Long Blink)", (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                self.last_action_time = now
                self.is_blinking      = False

            # Mouth open → keyboard
            elif mouth_open and now - self.last_action_time > 2.0:
                self.open_virtual_keyboard() or self.keyboard_shortcut_method()
                cv2.putText(frame, "Opening Keyboard (Mouth Open)", (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                self.last_action_time = now

            # Progress bars
            if self.is_blinking:
                prog = min(1.0, (now - self.blink_start_time) / self.long_blink_duration)
                cv2.rectangle(frame, (50, 250), (50 + int(200*prog), 270), (0, 255, 255), -1)
                cv2.rectangle(frame, (50, 250), (250, 270), (255, 255, 255), 2)
                if prog > 0.5:
                    cv2.putText(frame, "Hold...", (60, 265),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            if self.is_mouth_open:
                prog = min(1.0, (now - self.mouth_open_start) / self.mouth_open_duration)
                cv2.rectangle(frame, (50, 280), (50 + int(200*prog), 300), (0, 165, 255), -1)
                cv2.rectangle(frame, (50, 280), (250, 300), (255, 255, 255), 2)
                if prog > 0.5:
                    cv2.putText(frame, "Keep Open...", (60, 295),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Blink / wink
            if blink and not self.is_blinking and now - self.last_action_time > 0.8:
                pyautogui.click()
                self.last_action_time = now
                cv2.putText(frame, "Click", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif left_wink and now - self.last_action_time > self.zoom_cooldown:
                self.zoom_out()
                self.last_action_time = now
                cv2.putText(frame, "Zoom Out", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif right_wink and now - self.last_action_time > self.zoom_cooldown:
                self.zoom_in()
                self.last_action_time = now
                cv2.putText(frame, "Zoom In", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Eyebrow raises
            if eyebrows_raised:
                cv2.putText(frame, "Eyebrows Raised", (50, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                if now - self.last_brow_time > 0.7 and now - self.last_brow_time < self.brow_cooldown:
                    self.brow_raises   += 1
                    self.last_brow_time = now
                elif now - self.last_brow_time >= self.brow_cooldown:
                    self.brow_raises    = 1
                    self.last_brow_time = now

                cv2.putText(frame, f"Raises: {self.brow_raises}", (50, 220),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                if self.brow_raises == 2 and now - self.last_action_time > 0.8:
                    pyautogui.click(button='right')
                    cv2.putText(frame, "Right Click", (50, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    self.last_action_time = now
                elif self.brow_raises >= 3 and now - self.last_action_time > 0.8:
                    self.open_virtual_keyboard() or self.keyboard_shortcut_method()
                    cv2.putText(frame, "Opening Keyboard (Eyebrows)", (50, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                    self.last_action_time = now
                    self.brow_raises      = 0

        else:
            cv2.putText(frame, f"Calibrating: {self.init_frames}/{self.init_required_frames}",
                        (50, 50),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, "Look at the CENTER of your screen",
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        return frame


# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("Starting Eye Tracker...")
    tracker = EyeTracker()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam. Try changing index to 1 or 2.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("Webcam opened. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame — exiting.")
            break

        frame     = cv2.flip(frame, 1)
        processed = tracker.process_frame(frame)

        h, w = processed.shape[:2]
        labels = [
            "Controls:",
            "Blink           = Click",
            "Right Wink      = Zoom In",
            "Left Wink       = Zoom Out",
            "2x Eyebrow      = Right Click",
            "3x Eyebrow      = Keyboard",
            "Long Blink 1.5s = Keyboard",
            "Open Mouth 1.2s = Keyboard",
        ]
        for i, txt in enumerate(labels):
            scale = 0.7 if i == 0 else 0.55
            cv2.putText(processed, txt, (w - 340, 30 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1)

        cv2.putText(processed, "Press Q to quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Eye Controlled Mouse", processed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Exited cleanly.")


if __name__ == "__main__":
    main()
