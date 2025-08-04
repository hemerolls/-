import cv2
import os
import numpy as np
from pygame import mixer
import time
import mediapipe as mp
from keras.models import load_model
import logging
import sys


def resource_path(relative_path):
    """ Получает абсолютный путь к ресурсу, работает для dev и для PyInstaller """
    try:

        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)



logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


try:

    model_path = resource_path(os.path.join('models', 'mine.h5'))

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at expected path: {model_path}. Current working directory: {os.getcwd()}")
    model = load_model(model_path)
    logger.info(f"Model loaded successfully from: {model_path}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")

    sys.exit(1)


try:

    sound_path = resource_path('alarm.wav')

    if not os.path.exists(sound_path):
        raise FileNotFoundError(f"Sound file not found at expected path: {sound_path}. Current working directory: {os.getcwd()}")
    mixer.init()
    sound = mixer.Sound(sound_path)
    logger.info(f"Sound loaded successfully from: {sound_path}")
except Exception as e:
    logger.error(f"Failed to load sound: {e}")

    sound = None
# ---


LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


blink_count = 0
last_blink_time = time.time()

no_blink_threshold = 45.0
eye_closed_start_time = None
eye_closed_duration = 0
min_blink_duration = 0.15
max_blink_duration = 0.4

# Для отслеживания состояния глаз
previous_eye_state = "open"
current_eye_state = "open"

def calculate_eye_aspect_ratio(eye_coords):
    """
    Вычисляет соотношение сторон глаза (EAR)
    """
    if len(eye_coords) < 6:
        return 0
    # Вертикальные расстояния
    A = np.linalg.norm(np.array(eye_coords[1]) - np.array(eye_coords[5]))
    B = np.linalg.norm(np.array(eye_coords[2]) - np.array(eye_coords[4]))
    # Горизонтальное расстояние
    C = np.linalg.norm(np.array(eye_coords[0]) - np.array(eye_coords[3]))
    # EAR = (A + B) / (2.0 * C)
    ear = (A + B) / (2.0 * C) if C > 0 else 0
    return ear

def is_eye_closed(eye_coords, threshold=0.25):
    """
    Определяет, закрыт ли глаз
    """
    ear = calculate_eye_aspect_ratio(eye_coords)
    return ear < threshold

def detect_blink(eye_state, previous_state):
    """
    Обнаруживает моргание
    """
    global blink_count, last_blink_time, previous_eye_state, eye_closed_start_time
    current_time = time.time()

    if previous_state == "open" and eye_state == "closed":
        eye_closed_start_time = current_time

    elif previous_state == "closed" and eye_state == "open":
        if eye_closed_start_time is not None:
            blink_duration = current_time - eye_closed_start_time

            if min_blink_duration <= blink_duration <= max_blink_duration:
                blink_count += 1
                last_blink_time = current_time
                eye_closed_start_time = None
    previous_eye_state = eye_state

def get_eye_coords(eye_indices, face_landmarks, width, height):
    """
    Получает координаты глаза
    """
    coords = []
    for i in eye_indices:
        pt = face_landmarks.landmark[i]
        x = int(pt.x * width)
        y = int(pt.y * height)
        coords.append((x, y))
    return coords

def crop_eye(eye_coords, frame):
    """
    Обрезает область вокруг глаза
    """
    if len(eye_coords) == 0:
        return None
    x_min = min([pt[0] for pt in eye_coords])
    x_max = max([pt[0] for pt in eye_coords])
    y_min = min([pt[1] for pt in eye_coords])
    y_max = max([pt[1] for pt in eye_coords])
    # Добавляем небольшой отступ
    padding = 5
    x_min = max(0, x_min - padding)
    x_max = min(frame.shape[1], x_max + padding) # Используем ширину кадра
    y_min = max(0, y_min - padding)
    y_max = min(frame.shape[0], y_max + padding) # Используем высоту кадра
    if x_max > x_min and y_max > y_min:
        return frame[y_min:y_max, x_min:x_max]
    return None


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    logger.error("Cannot open camera")
    sys.exit(1)

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0
thicc = 2
rpred = [99]
lpred = [99]

while True:
    ret, frame = cap.read()
    if not ret:
        logger.warning("Can't receive frame (stream end?). Exiting ...")
        break
    height, width = frame.shape[:2]

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)

    cv2.rectangle(frame, (0, height - 100), (300, height), (0, 0, 0), thickness=cv2.FILLED)


    face_detected = results.multi_face_landmarks is not None

    left_eye_img = None
    right_eye_img = None

    if face_detected:
        for face_landmarks in results.multi_face_landmarks:

            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())


            left_eye_coords = get_eye_coords(LEFT_EYE, face_landmarks, width, height)
            right_eye_coords = get_eye_coords(RIGHT_EYE, face_landmarks, width, height)


            left_eye_img = crop_eye(left_eye_coords, frame)
            right_eye_img = crop_eye(right_eye_coords, frame)

            # Рисуем прямоугольники вокруг глаз (опционально)
            if left_eye_img is not None and len(left_eye_coords) > 0:
                x_min = min([pt[0] for pt in left_eye_coords]) - 5
                x_max = max([pt[0] for pt in left_eye_coords]) + 5
                y_min = min([pt[1] for pt in left_eye_coords]) - 5
                y_max = max([pt[1] for pt in left_eye_coords]) + 5
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
            if right_eye_img is not None and len(right_eye_coords) > 0:
                x_min = min([pt[0] for pt in right_eye_coords]) - 5
                x_max = max([pt[0] for pt in right_eye_coords]) + 5
                y_min = min([pt[1] for pt in right_eye_coords]) - 5
                y_max = max([pt[1] for pt in right_eye_coords]) + 5
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
    else:

        cv2.putText(frame, "Face Not Detected", (10, height - 80), font, 1, (0, 0, 255), 1, cv2.LINE_AA)


    if right_eye_img is not None:
        r_eye = cv2.cvtColor(right_eye_img, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (24, 24))
        r_eye = r_eye / 255.0
        r_eye = r_eye.reshape(24, 24, -1)
        r_eye = np.expand_dims(r_eye, axis=0)

        if model is not None:
            try:
                prediction = model.predict(r_eye, verbose=0)
                rpred = np.argmax(prediction, axis=1)
            except Exception as e:
                logger.error(f"Error during right eye prediction: {e}")
                rpred = [0]
        else:
            rpred = [0]
    else:
        rpred = [0]


    if left_eye_img is not None:
        l_eye = cv2.cvtColor(left_eye_img, cv2.COLOR_BGR2GRAY)
        count = count + 1
        l_eye = cv2.resize(l_eye, (24, 24))
        l_eye = l_eye / 255.0
        l_eye = l_eye.reshape(24, 24, -1)
        l_eye = np.expand_dims(l_eye, axis=0)

        if model is not None:
            try:
                prediction = model.predict(l_eye, verbose=0)
                lpred = np.argmax(prediction, axis=1)
            except Exception as e:
                logger.error(f"Error during left eye prediction: {e}")
                lpred = [0]
        else:
            lpred = [0]
    else:
        lpred = [0]


    left_eye_closed = False
    right_eye_closed = False
    if face_detected and results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_eye_coords = get_eye_coords(LEFT_EYE, face_landmarks, width, height)
            right_eye_coords = get_eye_coords(RIGHT_EYE, face_landmarks, width, height)
            left_eye_closed = is_eye_closed(left_eye_coords)
            right_eye_closed = is_eye_closed(right_eye_coords)


    if not face_detected:

        pass
    elif (rpred[0] == 0 and lpred[0] == 0) or (left_eye_closed and right_eye_closed):
        current_eye_state = "closed"
        score = score + 0.5
        cv2.putText(frame, "Closed", (10, height - 80), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
    else:
        current_eye_state = "open"
        score = max(0, score - 1)
        cv2.putText(frame, "Open", (10, height - 80), font, 1, (0, 255, 0), 1, cv2.LINE_AA)


    if face_detected:
        detect_blink(current_eye_state, previous_eye_state)


    current_time = time.time()
    if face_detected:
        time_since_last_blink = current_time - last_blink_time
        no_blink_alert = time_since_last_blink > no_blink_threshold
    else:
        time_since_last_blink = current_time - last_blink_time
        no_blink_alert = False


    cv2.putText(frame, 'Score: ' + str(score), (10, height - 60), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, 'Blinks: ' + str(blink_count), (10, height - 40), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    if face_detected:
        cv2.putText(frame, 'No blink: {:.1f}s'.format(time_since_last_blink), (10, height - 20), font, 1,
                    (255, 255, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, 'No blink: --', (10, height - 20), font, 1, (128, 128, 128), 1, cv2.LINE_AA)


    alert_triggered = False
    if face_detected and (score > 15 or no_blink_alert):
        alert_triggered = True

        try:
            cv2.imwrite(resource_path('image.jpg'), frame)
        except Exception as e:
             cv2.imwrite(os.path.join(os.getcwd(), 'image.jpg'), frame) # fallback

        if sound is not None:
            try:
                sound.play()
            except Exception as e:
                 logger.error(f"Error playing sound: {e}")
                 pass
        else:
             logger.warning("Sound not loaded, skipping alert sound.")

        if thicc < 16:
            thicc = thicc + 2
        else:
            thicc = thicc - 2
            if thicc < 2:
                thicc = 2
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)

        if score > 15:
            cv2.putText(frame, 'ALERT: Eyes Closed!', (width // 2 - 150, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        elif no_blink_alert:
            cv2.putText(frame, 'ALERT: No Blinking!', (width // 2 - 150, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)


    if not alert_triggered and face_detected and score > 0:
        score = max(0, score - 0.1)

    cv2.imshow('Drowsiness Detection', frame)


    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()