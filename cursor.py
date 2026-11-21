import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import math
import time

# --- НАСТРОЙКИ ---
SENSITIVITY_X = 8.0
SENSITIVITY_Y = 8.0
DEAD_ZONE = 0.01
SMOOTHING = 1

# Порог моргания и задержка (чтобы не было ложных кликов)
EYE_CLOSE_THRESHOLD = 0.012
CLICK_COOLDOWN = 1.0  # Секунд между кликами

# --- ИНИЦИАЛИЗАЦИЯ ---
model_path = "face_landmarker.task"
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

screen_w, screen_h = pyautogui.size()
pyautogui.FAILSAFE = False

# Переменные
calibrated = False
center_x, center_y = 0.0, 0.0
prev_mouse_x, prev_mouse_y = screen_w / 2, screen_h / 2
smoothed_head_x, smoothed_head_y = 0.0, 0.0

# Переменные для моргания
blink_start_time = 0
is_blinking = False
last_click_time = 0


def distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


print("=== КАЛИБРОВКА ===")
print("1. Смотрите прямо в центр экрана.")
print("2. Нажмите ПРОБЕЛ (Space), чтобы запомнить центр.")
print("3. Управляйте головой. Закройте глаза на 0.5 сек для клика.")
print("Нажмите 'q' для выхода.")

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect(mp_image)

    h, w, _ = frame.shape
    current_time = time.time()

    if result.face_landmarks:
        face_landmarks = result.face_landmarks[0]

        # --- 1. УПРАВЛЕНИЕ ГОЛОВОЙ ---
        nose = face_landmarks[1]
        raw_head_x, raw_head_y = nose.x, nose.y

        if smoothed_head_x == 0.0:
            smoothed_head_x, smoothed_head_y = raw_head_x, raw_head_y
        else:
            smoothed_head_x = smoothed_head_x * 0.8 + raw_head_x * 0.2
            smoothed_head_y = smoothed_head_y * 0.8 + raw_head_y * 0.2

        nx, ny = int(smoothed_head_x * w), int(smoothed_head_y * h)
        cv2.circle(frame, (nx, ny), 5, (0, 255, 0), -1)

        # --- КАЛИБРОВКА ---
        if not calibrated:
            cv2.putText(frame, "PRESS SPACE TO CALIBRATE", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            if cv2.waitKey(1) & 0xFF == 32:
                center_x, center_y = smoothed_head_x, smoothed_head_y
                calibrated = True
                print(f"✅ Калибровка выполнена!")
        else:
            # Движение мыши
            dx = smoothed_head_x - center_x
            dy = smoothed_head_y - center_y

            if abs(dx) < DEAD_ZONE: dx = 0
            if abs(dy) < DEAD_ZONE: dy = 0

            target_x = screen_w / 2 + dx * screen_w * SENSITIVITY_X
            target_y = screen_h / 2 + dy * screen_h * SENSITIVITY_Y
            target_x = max(0, min(screen_w - 1, target_x))
            target_y = max(0, min(screen_h - 1, target_y))

            current_mouse_x = prev_mouse_x + (target_x - prev_mouse_x) * SMOOTHING
            current_mouse_y = prev_mouse_y + (target_y - prev_mouse_y) * SMOOTHING
            pyautogui.moveTo(int(current_mouse_x), int(current_mouse_y), duration=0.01)
            prev_mouse_x, prev_mouse_y = current_mouse_x, current_mouse_y

            # --- 2. МОРГАНИЕ ДЛЯ КЛИКА ---
            left_eye_open = distance(face_landmarks[159], face_landmarks[145])
            right_eye_open = distance(face_landmarks[386], face_landmarks[374])
            avg_eye_open = (left_eye_open + right_eye_open) / 2

            if avg_eye_open < EYE_CLOSE_THRESHOLD:
                if not is_blinking:
                    is_blinking = True
                    blink_start_time = current_time
            else:
                if is_blinking:
                    blink_duration = current_time - blink_start_time
                    is_blinking = False

                    # Если глаза были закрыты больше 0.3 сек и не прошло время кулдауна
                    if blink_duration > 0.3 and (current_time - last_click_time) > CLICK_COOLDOWN:
                        pyautogui.click()
                        last_click_time = current_time
                        print(f"🖱️ КЛИК! (закрыты {blink_duration:.2f} сек)")

            # Визуализация статуса
            cv2.putText(frame, f"Mouse: {int(current_mouse_x)}x{int(current_mouse_y)}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            status_text = "Eyes OPEN" if not is_blinking else "CLICKING..."
            status_color = (0, 255, 0) if not is_blinking else (0, 0, 255)
            cv2.putText(frame, status_text, (50, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    cv2.imshow("Head Mouse with Blink Click", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()