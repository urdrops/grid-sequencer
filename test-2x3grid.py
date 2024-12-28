import cv2
import mediapipe as mp
import pygame
import math
import time
import os

class GridSequencer:
    def __init__(self):
        # --- MediaPipe для распознавания рук ---
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

        # --- Pygame mixer ---
        pygame.mixer.init()
        pygame.mixer.set_num_channels(6)  # 6 каналов под 6 ячеек

        # Папки со звуками (соответствие ячеек)
        #  0= Bass, 1=Drums, 2=Fx, 3=Guitar, 4=Synth, 5=Vox
        self.cell_folders = [
            "bits/Bass",
            "bits/Drums",
            "bits/Fx",
            "bits/Guitar",
            "bits/Synth",
            "bits/Vox"
        ]

        # Текстовые метки для ячеек (то, что будет отображаться на экране)
        self.cell_labels = [
            "Bass",    # cell 0
            "Drums",   # cell 1
            "Fx",      # cell 2
            "Guitar",  # cell 3
            "Synth",   # cell 4
            "Vox"      # cell 5
        ]

        # Загрузим mp3 в память. Каждая ячейка → список из 4 треков
        self.sounds = []
        for folder in self.cell_folders:
            mp3_files = sorted([f for f in os.listdir(folder) if f.endswith(".mp3")])
            cell_sounds = []
            for mp3file in mp3_files[:4]:  # только первые 4, если их больше
                path = os.path.join(folder, mp3file)
                snd = pygame.mixer.Sound(path)
                cell_sounds.append(snd)
            self.sounds.append(cell_sounds)

        # Активный трек в каждой ячейке (или -1, если не играет)
        self.active_track_in_cell = [-1]*6

        # Цвета для подсветки ячеек (BGR)
        self.cell_colors = [
            (255, 0, 0),   # Bass   (cell 0)
            (0, 255, 0),   # Drums  (cell 1)
            (0, 0, 255),   # Fx     (cell 2)
            (255, 255, 0), # Guitar (cell 3)
            (255, 0, 255), # Synth  (cell 4)
            (0, 255, 255)  # Vox    (cell 5)
        ]

        # Дебаунс (предыдущие касания четырёх пальцев)
        self.prev_touches = [False]*4

        # Сетка (будет заполнена в create_grid)
        self.cells = []

        # Для сеточного запуска по 1.6 с
        self.schedule = []    # (time, action, cell_idx, track_idx)
        self.start_time = time.time()

        # Прозрачность при отрисовке цветных прямоугольников
        self.overlay_alpha = 0.3

        # Порог для определения "касания" (пиксели)
        self.touch_threshold = 40

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Камера недоступна.")
            return

        while True:
            success, frame = cap.read()
            if not success:
                break

            # Зеркалим кадр, переводим в RGB (для MediaPipe)
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Обрабатываем кадр
            results = self.hands.process(rgb)

            # Создаём/рисуем сетку
            self.create_grid(frame)

            # Если найдена (хотя бы одна) рука
            if results.multi_hand_landmarks:
                for hand_lm in results.multi_hand_landmarks:
                    # Рисуем скелет руки (для наглядности)
                    self.mp_draw.draw_landmarks(frame, hand_lm, self.mp_hands.HAND_CONNECTIONS)
                    # Обновляем логику (какая ячейка, какой палец, включаем/выключаем)
                    self.update_logic(frame, hand_lm)

            # Подсветим активные ячейки (и выведем названия/номера треков)
            self.highlight_active_cells(frame)

            # Проверяем расписание (запуск/остановка по 1.6)
            self.update_scheduler()

            # Показываем на экране
            cv2.imshow("Grid Sequencer", frame)

            # Выход по нажатию 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Закрываем ресурсы
        cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()

    # ----------------------------------------------------------------
    #      ЛОГИКА ЯЧЕЕК И ЖЕСТОВ
    # ----------------------------------------------------------------

    def create_grid(self, frame):
        """Отрисовываем сетку 2×3 и запоминаем координаты 6 ячеек."""
        h, w = frame.shape[:2]
        cell_h = h // 2
        cell_w = w // 3

        # рисуем вертикальные линии
        cv2.line(frame, (cell_w, 0), (cell_w, h), (255, 255, 255), 2)
        cv2.line(frame, (cell_w*2, 0), (cell_w*2, h), (255, 255, 255), 2)
        # рисуем горизонтальную
        cv2.line(frame, (0, cell_h), (w, cell_h), (255, 255, 255), 2)

        self.cells = [
            [(0, 0),       (cell_w, cell_h)],       # cell 0
            [(cell_w, 0),  (cell_w*2, cell_h)],      # cell 1
            [(cell_w*2, 0),(w, cell_h)],             # cell 2
            [(0, cell_h),       (cell_w, h)],        # cell 3
            [(cell_w, cell_h),  (cell_w*2, h)],      # cell 4
            [(cell_w*2, cell_h),(w, h)]              # cell 5
        ]

    def update_logic(self, frame, hand_landmarks):
        """
        1) Определяем, в какую ячейку попало запястье (landmark[0])
        2) Проверяем, какие пальцы касаются большого (thumb=4).
        3) При "переходе" (False->True) запускаем toggle_track по ячейке и номеру пальца.
        """
        h, w = frame.shape[:2]

        # Координаты запястья
        wrist = hand_landmarks.landmark[9]
        wx, wy = int(wrist.x*w), int(wrist.y*h)
        cell_idx = self.get_cell_index(wx, wy)

        # Какие пальцы касаются большого
        touches_now = self.detect_finger_thumb_touch(frame, hand_landmarks)

        if 0 <= cell_idx < 6:
            for i in range(4):
                if touches_now[i] and not self.prev_touches[i]:
                    # "Нажатие" пальца i в ячейке cell_idx
                    self.toggle_track(cell_idx, i)

        # Обновляем состояние для дебаунса
        self.prev_touches = touches_now

    def detect_finger_thumb_touch(self, frame, hand_landmarks):
        """
        Проверяем 4 пальца (tip: 8,12,16,20) на близость к thumb tip (4).
        Возвращаем массив [False,False,False,False] или True, если касание < self.touch_threshold.
        """
        h, w = frame.shape[:2]
        thumb = hand_landmarks.landmark[4]
        tx, ty = int(thumb.x*w), int(thumb.y*h)

        tip_ids = [8, 12, 16, 20]  # указательный, средний, безымянный, мизинец
        touches = [False]*4
        for i, tip_id in enumerate(tip_ids):
            f = hand_landmarks.landmark[tip_id]
            fx, fy = int(f.x*w), int(f.y*h)
            dist = ((fx - tx)**2 + (fy - ty)**2)**0.5
            if dist < self.touch_threshold:
                touches[i] = True
        return touches

    def get_cell_index(self, x, y):
        """Определяем, в какой из 6 ячеек попали координаты (x,y)."""
        for i, ((x1,y1),(x2,y2)) in enumerate(self.cells):
            if x1 <= x < x2 and y1 <= y < y2:
                return i
        return -1

    # ----------------------------------------------------------------
    #      ЛОГИКА ВКЛ/ВЫКЛ ТРЕКОВ С РАСПИСАНИЕМ (1.6 С)
    # ----------------------------------------------------------------

    def next_multiple_1_6(self):
        """Возвращает ближайшее время, кратное 1.6,
           которое не меньше текущего момента."""
        now = time.time() - self.start_time
        n = math.ceil(now / 1.6)
        return n * 1.6

    def schedule_action(self, action, cell_idx, track_idx):
        """Запланировать start или stop на ближайшем 'слоте' 1.6."""
        t = self.next_multiple_1_6()
        self.schedule.append((t, action, cell_idx, track_idx))
        print(f"[SCHEDULE] At {t:.2f}s → {action} cell={cell_idx}, track={track_idx}")

    def update_scheduler(self):
        """Каждый кадр вызываем: проверяем, не пора ли исполнить запланированное."""
        now = time.time() - self.start_time
        done = []
        for event in self.schedule:
            t, action, c_idx, t_idx = event
            if now >= t:
                if action == 'start':
                    self.start_sound(c_idx, t_idx)
                elif action == 'stop':
                    self.stop_sound(c_idx, t_idx)
                done.append(event)
        # Удаляем отработанные
        for ev in done:
            self.schedule.remove(ev)

    def toggle_track(self, cell_idx, track_idx):
        """
        Если текущая ячейка уже играет этот же трек → планируем stop.
        Иначе планируем stop старого трека (если был) и start нового.
        """
        current = self.active_track_in_cell[cell_idx]
        print(f"[TOGGLE] cell={cell_idx}, finger/track={track_idx}, current={current}")

        if current == track_idx:
            # Останавливаем
            self.schedule_action('stop', cell_idx, track_idx)
        else:
            # Останавливаем предыдущий, если был
            if current != -1:
                self.schedule_action('stop', cell_idx, current)
            # Стартуем новый
            self.schedule_action('start', cell_idx, track_idx)

    def start_sound(self, cell_idx, track_idx):
        """
        Реальный запуск звука (без зацикливания для Fx = cell_idx=2).
        """
        snd = self.sounds[cell_idx][track_idx]
        ch = pygame.mixer.Channel(cell_idx)
        # Fx (ячейка 2) играет один раз (loops=0), остальные - в loop
        if cell_idx == 2:
            loops_count = 0
        else:
            loops_count = -1

        ch.play(snd, loops=loops_count)
        self.active_track_in_cell[cell_idx] = track_idx
        print(f"[START] cell={cell_idx}, track={track_idx}, loops={loops_count}")

    def stop_sound(self, cell_idx, track_idx):
        """Останавливаем канал ячейки cell_idx, если там играется track_idx."""
        ch = pygame.mixer.Channel(cell_idx)
        ch.stop()
        if self.active_track_in_cell[cell_idx] == track_idx:
            self.active_track_in_cell[cell_idx] = -1
        print(f"[STOP] cell={cell_idx}, track={track_idx}")

    # ----------------------------------------------------------------
    #      ВИЗУАЛЬНАЯ ПОДСВЕТКА И ТЕКСТ
    # ----------------------------------------------------------------

    def highlight_active_cells(self, frame):
        """
        1) Создаём overlay и закрашиваем ячейки, в которых
           сейчас играет хотя бы один трек (у нас максимум один трек на ячейку).
        2) Пишем текст: Название ячейки + номер трека (или Off).
        """
        overlay = frame.copy()
        for i, ((x1,y1),(x2,y2)) in enumerate(self.cells):
            track_idx = self.active_track_in_cell[i]
            # Если ячейка активна (track_idx != -1), заливаем цветом
            if track_idx != -1:
                color = self.cell_colors[i]
                cv2.rectangle(overlay, (x1,y1), (x2,y2), color, -1)

        # Наложение с прозрачностью
        cv2.addWeighted(overlay, self.overlay_alpha, frame, 1 - self.overlay_alpha, 0, frame)

        # Теперь поверх (уже финального кадра) пишем текст
        for i, ((x1,y1),(x2,y2)) in enumerate(self.cells):
            # Точка, где будем писать текст (примерно левый верх)
            text_x, text_y = x1 + 10, y1 + 30

            # Название ячейки
            label = self.cell_labels[i]

            # Активный трек
            track_idx = self.active_track_in_cell[i]
            if track_idx == -1:
                track_text = "Off"
            else:
                # track_idx - это 0..3, показываем как 1..4
                track_text = f"Track {track_idx + 1}"

            # Собираем строку вида: "Bass (Track 2)" или "Fx (Off)"
            full_text = f"{label} ({track_text})"

            # Выводим текст белым цветом
            cv2.putText(
                frame,
                full_text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,            # масштаб
                (255, 255, 255),# цвет текста (B,G,R)
                2,              # толщина
                cv2.LINE_AA
            )


if __name__ == "__main__":
    seq = GridSequencer()
    seq.run()
