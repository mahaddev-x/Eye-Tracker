import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import pyautogui
import math
import threading
import time
import keyboard

# ── Screen setup ──
MONITOR_WIDTH, MONITOR_HEIGHT = pyautogui.size()
CENTER_X = MONITOR_WIDTH // 2
CENTER_Y = MONITOR_HEIGHT // 2
mouse_control_enabled = True
filter_length = 8

FACE_OUTLINE_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356,
    454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109
]

# Shared mouse target position
mouse_target = [CENTER_X, CENTER_Y]
mouse_lock = threading.Lock()

# ── Head precision offset calibration ──
calibration_offset_yaw = 0
calibration_offset_pitch = 0
HEAD_PRECISION_SCALE = 15  # pixels of offset per degree of head rotation

# Buffers for smoothing
ray_directions = deque(maxlen=filter_length)
finger_positions = deque(maxlen=6)  # smooth finger positions

# ── Initialize MediaPipe ──
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Open camera
cap = cv2.VideoCapture(0)

# Face landmark indices for head orientation
LANDMARKS = {
    "left": 234,
    "right": 454,
    "top": 10,
    "bottom": 152,
    "front": 1,
}

def mouse_mover():
    while True:
        if mouse_control_enabled:
            with mouse_lock:
                x, y = mouse_target
            pyautogui.moveTo(x, y)
        time.sleep(0.01)

def landmark_to_np(landmark, w, h):
    return np.array([landmark.x * w, landmark.y * h, landmark.z * w])

# ── Blink-to-Click Setup ──
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE  = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD = 0.21
BLINK_MIN_SEC = 0.30
BLINK_MAX_SEC = 0.80
eyes_closed_since = None
blink_click_cooldown = 0

def compute_ear(landmarks, eye_indices, w, h):
    """Compute Eye Aspect Ratio for one eye."""
    pts = [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in eye_indices]
    vertical1 = np.linalg.norm(pts[1] - pts[5])
    vertical2 = np.linalg.norm(pts[2] - pts[4])
    horizontal = np.linalg.norm(pts[0] - pts[3])
    if horizontal < 1e-6:
        return 0.3
    return (vertical1 + vertical2) / (2.0 * horizontal)

threading.Thread(target=mouse_mover, daemon=True).start()

# ── Main loop ──
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame = cv2.flip(frame, 1)  # mirror so left/right feel natural
    landmarks_frame = np.zeros_like(frame)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_results = face_mesh.process(rgb)
    hand_results = hands.process(rgb)

    finger_screen_x = None
    finger_screen_y = None
    head_offset_x = 0
    head_offset_y = 0
    raw_yaw_deg = 180
    raw_pitch_deg = 180

    # ══════════════════════════════════════════
    # 1) FINGER TRACKING  (primary cursor)
    # ══════════════════════════════════════════
    if hand_results.multi_hand_landmarks:
        hand_lm = hand_results.multi_hand_landmarks[0]

        # Index finger tip = landmark 8
        idx_tip = hand_lm.landmark[8]

        # Map finger position in camera space to screen coordinates
        # Camera x is mirrored, so already flipped above
        fx = idx_tip.x  # 0..1 across frame width
        fy = idx_tip.y  # 0..1 across frame height

        finger_positions.append((fx, fy))
        avg_fx = np.mean([p[0] for p in finger_positions])
        avg_fy = np.mean([p[1] for p in finger_positions])

        finger_screen_x = int(avg_fx * MONITOR_WIDTH)
        finger_screen_y = int(avg_fy * MONITOR_HEIGHT)

        # Draw index finger on frame
        px, py = int(idx_tip.x * w), int(idx_tip.y * h)
        cv2.circle(frame, (px, py), 10, (0, 255, 0), -1)
        cv2.putText(frame, "INDEX", (px + 15, py - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw all hand landmarks lightly
        for lm in hand_lm.landmark:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 2, (200, 200, 200), -1)

    # ══════════════════════════════════════════
    # 2) HEAD ROTATION  (precision offset)
    # ══════════════════════════════════════════
    if face_results.multi_face_landmarks:
        face_landmarks = face_results.multi_face_landmarks[0].landmark

        # Draw face landmarks
        for i, landmark in enumerate(face_landmarks):
            pt = landmark_to_np(landmark, w, h)
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < w and 0 <= y < h:
                color = (155, 155, 155) if i in FACE_OUTLINE_INDICES else (255, 25, 10)
                cv2.circle(landmarks_frame, (x, y), 3, color, -1)

        # Compute head orientation
        key_points = {}
        for name, idx in LANDMARKS.items():
            pt = landmark_to_np(face_landmarks[idx], w, h)
            key_points[name] = pt
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(frame, (x, y), 4, (0, 0, 0), -1)

        left = key_points["left"]
        right = key_points["right"]
        top = key_points["top"]
        bottom = key_points["bottom"]
        front = key_points["front"]

        right_axis = (right - left)
        right_axis /= np.linalg.norm(right_axis)
        up_axis = (top - bottom)
        up_axis /= np.linalg.norm(up_axis)
        forward_axis = np.cross(right_axis, up_axis)
        forward_axis /= np.linalg.norm(forward_axis)
        forward_axis = -forward_axis

        center = (left + right + top + bottom + front) / 5

        # Wireframe cube
        half_width = np.linalg.norm(right - left) / 2
        half_height = np.linalg.norm(top - bottom) / 2
        half_depth = 80

        def corner(x_sign, y_sign, z_sign):
            return (center
                    + x_sign * half_width * right_axis
                    + y_sign * half_height * up_axis
                    + z_sign * half_depth * forward_axis)

        cube_corners = [
            corner(-1, 1, -1), corner(1, 1, -1),
            corner(1, -1, -1), corner(-1, -1, -1),
            corner(-1, 1, 1), corner(1, 1, 1),
            corner(1, -1, 1), corner(-1, -1, 1)
        ]

        def project(pt3d):
            return int(pt3d[0]), int(pt3d[1])

        cube_corners_2d = [project(pt) for pt in cube_corners]
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        for i, j in edges:
            cv2.line(frame, cube_corners_2d[i], cube_corners_2d[j], (255, 125, 35), 2)

        # Smooth head direction
        ray_directions.append(forward_axis)
        avg_direction = np.mean(ray_directions, axis=0)
        avg_direction /= np.linalg.norm(avg_direction)

        # Compute yaw/pitch angles
        reference_forward = np.array([0, 0, -1])

        xz_proj = np.array([avg_direction[0], 0, avg_direction[2]])
        xz_proj /= np.linalg.norm(xz_proj)
        yaw_rad = math.acos(np.clip(np.dot(reference_forward, xz_proj), -1.0, 1.0))
        if avg_direction[0] < 0:
            yaw_rad = -yaw_rad

        yz_proj = np.array([0, avg_direction[1], avg_direction[2]])
        yz_proj /= np.linalg.norm(yz_proj)
        pitch_rad = math.acos(np.clip(np.dot(reference_forward, yz_proj), -1.0, 1.0))
        if avg_direction[1] > 0:
            pitch_rad = -pitch_rad

        yaw_deg = np.degrees(yaw_rad)
        pitch_deg = np.degrees(pitch_rad)

        if yaw_deg < 0:
            yaw_deg = abs(yaw_deg)
        elif yaw_deg < 180:
            yaw_deg = 360 - yaw_deg

        if pitch_deg < 0:
            pitch_deg = 360 + pitch_deg

        raw_yaw_deg = yaw_deg
        raw_pitch_deg = pitch_deg

        # Head offset = deviation from calibrated center * sensitivity
        yaw_offset_deg = yaw_deg + calibration_offset_yaw - 180
        pitch_offset_deg = pitch_deg + calibration_offset_pitch - 180

        head_offset_x = int(yaw_offset_deg * HEAD_PRECISION_SCALE)
        head_offset_y = int(-pitch_offset_deg * HEAD_PRECISION_SCALE)

        # Draw head direction ray
        ray_length = 2.5 * half_depth
        ray_end = np.mean(list(deque(maxlen=filter_length)), axis=0) if len(ray_directions) == 0 else center - avg_direction * ray_length
        cv2.line(frame, project(center), project(ray_end), (15, 255, 0), 3)
        cv2.line(landmarks_frame, project(center), project(ray_end), (15, 255, 0), 3)

        # ── Blink-to-Click Detection ──
        ear_left  = compute_ear(face_landmarks, LEFT_EYE, w, h)
        ear_right = compute_ear(face_landmarks, RIGHT_EYE, w, h)
        ear_avg   = (ear_left + ear_right) / 2.0

        now = time.time()
        if blink_click_cooldown > 0:
            blink_click_cooldown -= 1

        if ear_avg < EAR_THRESHOLD:
            if eyes_closed_since is None:
                eyes_closed_since = now
            hold_time = now - eyes_closed_since
            bar_frac = min(hold_time / BLINK_MIN_SEC, 1.0)
            bar_color = (0, 255, 0) if bar_frac >= 1.0 else (0, 200, 255)
            cv2.rectangle(frame, (10, 10), (10 + int(200 * bar_frac), 30), bar_color, -1)
            cv2.rectangle(frame, (10, 10), (210, 30), (255, 255, 255), 1)
            cv2.putText(frame, "BLINK", (220, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            if eyes_closed_since is not None:
                hold_time = now - eyes_closed_since
                if BLINK_MIN_SEC <= hold_time <= BLINK_MAX_SEC and blink_click_cooldown <= 0:
                    pyautogui.click()
                    blink_click_cooldown = 15
                    print(f"[BLINK CLICK] hold={hold_time:.2f}s")
                eyes_closed_since = None

    # ══════════════════════════════════════════
    # 3) COMBINE: finger position + head offset
    # ══════════════════════════════════════════
    if finger_screen_x is not None:
        final_x = finger_screen_x + head_offset_x
        final_y = finger_screen_y + head_offset_y

        # Clamp
        final_x = max(10, min(final_x, MONITOR_WIDTH - 10))
        final_y = max(10, min(final_y, MONITOR_HEIGHT - 10))

        print(f"Finger=({finger_screen_x},{finger_screen_y}) Head offset=({head_offset_x},{head_offset_y}) -> Screen=({final_x},{final_y})")

        if mouse_control_enabled:
            with mouse_lock:
                mouse_target[0] = final_x
                mouse_target[1] = final_y

    # ── Display ──
    cv2.imshow("Head-Aligned Cube", frame)
    cv2.imshow("Facial Landmarks", landmarks_frame)

    if keyboard.is_pressed('f7'):
        mouse_control_enabled = not mouse_control_enabled
        print(f"[Mouse Control] {'Enabled' if mouse_control_enabled else 'Disabled'}")
        time.sleep(0.3)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        calibration_offset_yaw = 180 - raw_yaw_deg
        calibration_offset_pitch = 180 - raw_pitch_deg
        print(f"[Calibrated] Head center zeroed. Offset Yaw: {calibration_offset_yaw:.1f}, Offset Pitch: {calibration_offset_pitch:.1f}")

cap.release()
cv2.destroyAllWindows()
