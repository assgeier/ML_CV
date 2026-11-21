import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import subprocess

# --- НАСТРОЙКИ ---
hand_model = "hand_landmarker.task"
face_model = "face_landmarker.task"

# Инициализация детектора рук
hand_base_options = python.BaseOptions(model_asset_path=hand_model)
hand_options = vision.HandLandmarkerOptions(
    base_options=hand_base_options, num_hands=1,
    min_hand_detection_confidence=0.6, min_tracking_confidence=0.6
)
hand_detector = vision.HandLandmarker.create_from_options(hand_options)

# Инициализация детектора лица
face_base_options = python.BaseOptions(model_asset_path=face_model)
face_options = vision.FaceLandmarkerOptions(
    base_options=face_base_options, num_faces=1,
    min_face_detection_confidence=0.6, min_tracking_confidence=0.6
)
face_detector = vision.FaceLandmarker.create_from_options(face_options)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Программа запущена. Держи руку в кадре, чтобы задать громкость, и моргни для применения. 'q' для выхода.")

# Переменные
current_smoothed_volume = 50
last_applied_volume = 50
is_blinking = False
blink_cooldown = 0
hand_is_present = False  # Флаг: рука в кадре?


def distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def set_mac_volume(vol):
    vol = max(0, min(100, int(vol)))
    script = f"set volume output volume {vol}"
    subprocess.run(["osascript", "-e", script], capture_output=True)
    return vol


while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # --- 1. ОБРАБОТКА РУК ---
    hand_result = hand_detector.detect(mp_image)

    # Сбрасываем флаг в начале каждого кадра
    hand_is_present = False

    if hand_result.hand_landmarks:
        hand_is_present = True  # Рука найдена!
        hand_landmarks = hand_result.hand_landmarks[0]
        h, w, _ = frame.shape

        for lm in hand_landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

        thumb_tip = hand_landmarks[4]
        index_tip = hand_landmarks[8]
        tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)
        ix, iy = int(index_tip.x * w), int(index_tip.y * h)

        dist = math.hypot(ix - tx, iy - ty)
        cv2.line(frame, (tx, ty), (ix, iy), (255, 0, 255), 3)

        dist_clamped = max(30, min(300, dist))
        raw_volume = ((dist_clamped - 30) / (300 - 30)) * 100

        # Обновляем громкость только если рука есть
        current_smoothed_volume = current_smoothed_volume * 0.8 + raw_volume * 0.2

    # --- 2. ОБРАБОТКА ЛИЦА (МОРГАНИЕ) ---
    face_result = face_detector.detect(mp_image)

    if face_result.face_landmarks:
        face_landmarks = face_result.face_landmarks[0]

        left_eye_open = distance(face_landmarks[159], face_landmarks[145])
        right_eye_open = distance(face_landmarks[386], face_landmarks[374])
        avg_eye_open = (left_eye_open + right_eye_open) / 2

        blink_threshold = 0.012

        if avg_eye_open < blink_threshold:
            if not is_blinking and blink_cooldown == 0:
                is_blinking = True
                blink_cooldown = 15

                # 🔥 ГЛАВНОЕ ИСПРАВЛЕНИЕ:
                # Применяем громкость, ТОЛЬКО если рука прямо сейчас в кадре
                if hand_is_present and abs(current_smoothed_volume - last_applied_volume) > 2:
                    last_applied_volume = set_mac_volume(current_smoothed_volume)
                    print(f"👁️ Моргание (с рукой)! Громкость применена: {last_applied_volume}%")

                    cv2.circle(frame, (w - 100, 100), 30, (0, 255, 0), -1)
                    cv2.putText(frame, "SET", (w - 130, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        else:
            is_blinking = False

    if blink_cooldown > 0:
        blink_cooldown -= 1

    # --- 3. ВИЗУАЛИЗАЦИЯ ---
    # Цвет текста зависит от того, есть ли рука (зеленый = можно моргать, красный = нельзя)
    ui_color = (0, 255, 0) if hand_is_present else (0, 0, 255)

    cv2.rectangle(frame, (40, 40), (340, 120), (0, 0, 0), -1)
    cv2.putText(frame, f"Target Vol: {int(current_smoothed_volume)}%", (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, ui_color, 3)

    bar_x, bar_y, bar_w, bar_h = 50, 100, 280, 10
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), -1)
    fill_w = int((current_smoothed_volume / 100) * bar_w)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), ui_color, -1)

    if hand_is_present:
        cv2.putText(frame, "Blink to Apply", (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Show hand first!", (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Volume Control by Blink", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()