import cv2
import mediapipe as mp
import math

# Инициализация Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Функция для вычисления расстояния между двумя точками
def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

# Запуск видеозахвата
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Не удалось захватить изображение.")
            break

        # Перевернем изображение, чтобы было как в зеркале
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Обработка изображения Mediapipe
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Отрисовка скелета руки
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Координаты ключевых точек
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

                # Вычисляем расстояния
                distance_thumb_index = calculate_distance(thumb_tip, index_tip)
                distance_thumb_middle = calculate_distance(thumb_tip, middle_tip)

                # Проверка на касание (пороговое значение)
                if distance_thumb_index < 0.05:  # Настраиваемый порог
                    print("А")
                elif distance_thumb_middle < 0.05:
                    print("Б")

        # Отображение видео
        cv2.imshow('Hand Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
