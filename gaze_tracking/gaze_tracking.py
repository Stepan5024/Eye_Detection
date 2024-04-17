from __future__ import division
import os
import cv2
import dlib
#from .eye import Eye
from .eye_mp import EyeMP
from .calibration import Calibration
import mediapipe as mp
import numpy as np

# Инициализация MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils  # Для визуализации результатов

face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                                  min_detection_confidence=0.5)

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class GazeTracking(object):
    """
    Этот класс отслеживает взгляд пользователя.
    Он предоставляет полезную информацию, такую как положение глаз
    и зрачков, и позволяет узнать, открыты глаза или закрыты
    """

    def __init__(self):
        """
        Инициализирует объект GazeTracking

        Атрибуты:
            frame (numpy.ndarray): Текущий кадр для анализа взгляда.
            eye_left (Eye): Объект, представляющий левый глаз.
            eye_right (Eye): Объект, представляющий правый глаз.
            calibration (Calibration): Объект, представляющий данные калибровки для определения размера глаза.
            _face_detector: Объект детектора лиц из библиотеки dlib.
            _predictor: Объект предиктора лицевых ориентиров из библиотеки dlib.
        """
        self.frame = None # Текущий кадр для анализа взгляда.
        self.eye_left = None # Объект, представляющий левый глаз.
        self.eye_right = None  # Объект, представляющий правый глаз.
        self.calibration = Calibration() # Объект, представляющий данные калибровки для определения размера глаза.

        # _face_detector используется для обнаружения лиц
        self._face_detector = dlib.get_frontal_face_detector()
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False,
                                                    max_num_faces=1,
                                                    refine_landmarks=True,
                                                    min_detection_confidence=0.5,
                                                    min_tracking_confidence=0.5)
    @property
    def pupils_located(self):
        """Проверка расположения зрачков"""
        try:
            # Проверяет, что координаты зрачков обоих глаз являются целыми числами.
       
            int(self.eye_left.pupil.x)
            int(self.eye_left.pupil.y)
            int(self.eye_right.pupil.x)
            int(self.eye_right.pupil.y)
            return True
        except Exception:
             # Возвращает False, если хотя бы для одного зрачка координаты не являются целыми числами
        
            return False

    def _analyze(self):
        """Распознает лицо и инициализирует объекты для глаз
         Преобразует кадр в оттенки серого и использует детектор лиц для обнаружения лиц.
    Затем использует предиктор для получения точек Landmarks лица и инициализирует объекты Eye
    для левого и правого глаза, передавая им необходимые параметры. В случае ошибки (отсутствия
    обнаруженных лиц) устанавливает значения eye_left и eye_right в None.

    Замечание:
        Метод ожидает, что на кадре присутствует ровно одно обнаруженное лицо.

    Исключения:
        IndexError: Вызывается, если в списке faces нет обнаруженных лиц.
        """
         # Преобразует кадр в оттенки серого для использования в детекции лиц.
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
         # Использует детектор лиц для обнаружения лиц на кадре.
        faces = self._face_detector(frame)
        #print(f"faces dlib {faces}")
        results = face_detection.process(self.frame)  # Здесь frame должен быть вашим изображением в RGB
        # Проверка наличия лиц в результате
        if results.detections:
            faces_mp = []  # Список для хранения координат лиц
            for detection in results.detections:
                # Получение координат ограничивающего прямоугольника (bounding box) лица
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = self.frame.shape  # Размеры изображения
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                faces_mp.append(bbox)
                #print(f"faces mp {faces_mp}")

            # Теперь в переменной faces_mp хранится список кортежей с координатами ограничивающих прямоугольников лиц
            # Каждый кортеж представляет собой (x, y, width, height)
            else:
                faces_mp = None  # Лица не обнаружены
        
        try:
            # Использует предиктор для получения точек landmarks для первого обнаруженного лица.
            #landmarks = self._predictor(frame, faces[0])
            #print(f"landmarks dlib {landmarks}")
            # Вывод количества точек
            #print(f"Количество ключевых точек: {landmarks.num_parts}")
            # Обработка изображения с помощью Face Mesh
            results_face = face_mesh.process(self.frame)
            # Переменная для хранения ключевых точек лица в формате, аналогичном dlib
            landmarks_mp = []
            # Извлечение ключевых точек лица, если они обнаружены
            if results_face.multi_face_landmarks:
                for face_landmarks in results_face.multi_face_landmarks:
                    for landmark in face_landmarks.landmark:
                        x = int(landmark.x * self.frame.shape[1])
                        y = int(landmark.y * self.frame.shape[0])
                        landmarks_mp.append(Point(x, y))
            # Итерация по всем точкам и вывод их координат
            #for i in range(landmarks.num_parts):
            #    point = landmarks.part(i)
            #    print(f"Точка {i}: (x={point.x}, y={point.y})")
             # Инициализирует объекты Eye для левого и правого глаза с использованием landmarks.
            self.eye_left = EyeMP(self.frame, landmarks_mp, 0, self.calibration)
            self.eye_right = EyeMP(self.frame, landmarks_mp, 1, self.calibration)

            #self.eye_left = Eye(frame, landmarks, 0, self.calibration)
            #self.eye_right = Eye(frame, landmarks, 1, self.calibration)

        except IndexError:
             # В случае отсутствия обнаруженных лиц, устанавливает значения eye_left и eye_right в None.
        
            self.eye_left = None
            self.eye_right = None

    def refresh(self, frame):
        """Обновляет кадр и анализирует его.
        Трехмерный массив обычно используется для представления цветных изображений с 3 каналами (например, RGB), а двумерный - для черно-белых изображений.
        Аргументы:
            фрейм (numpy.ndarray): Фрейм для анализа
             Этот массив может быть либо двумерным (shape: [высота, ширина]) для черно-белых изображений, либо трехмерным (shape: [высота, ширина, каналы]) для цветных изображений, где каналы обычно равны 3 (для RGB).
        """
        #print(f"frame {type(frame)} {frame}")
        """<class 'numpy.ndarray'> [[[101  98 104]
  [102  99 105]
  [101  98 104]
  ...
  [100  98  93]
  [101  99  94]
  [101  99  94]]
  """
         # Обновляет текущий кадр объекта VideoAnalyzer.
        self.frame = frame
          # Вызывает метод _analyze для обработки и анализа нового кадра.
        self._analyze()

    def pupil_left_coords(self):
        """Возвращает координаты левого зрачка
         Возвращает координаты левого зрачка относительно исходной точки (origin) левого глаза.
    
        Возвращаемые значения:
        tuple or None: Кортеж с координатами (x, y) левого зрачка,
        или None, если координаты зрачка не были обнаружены.
                       """
        # Проверяет, были ли обнаружены координаты зрачков обоих глаз.
        if self.pupils_located:
             # Рассчитывает абсолютные координаты левого зрачка относительно исходной точки левого глаза.
        
            x = self.eye_left.origin[0] + self.eye_left.pupil.x
            y = self.eye_left.origin[1] + self.eye_left.pupil.y
            return (x, y)
        else:
        # Возвращает None, если координаты зрачка не были обнаружены.
            return None

    def pupil_right_coords(self):
        """Возвращает координаты правого зрачка
        Возвращает координаты правого зрачка относительно исходной точки (origin) правого глаза.
    
        Возвращаемые значения:
        tuple or None: Кортеж с координатами (x, y) правого зрачка,
                       или None, если координаты зрачка не были обнаружены.
        """
        if self.pupils_located:
            x = self.eye_right.origin[0] + self.eye_right.pupil.x
            y = self.eye_right.origin[1] + self.eye_right.pupil.y
            return (x, y)

    def horizontal_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        horizontal direction of the gaze. The extreme right is 0.0,
        the center is 0.5 and the extreme left is 1.0
        """
        # Проверяет, были ли обнаружены координаты зрачков обоих глаз.
        if self.pupils_located:
             # Рассчитывает абсолютные координаты правого зрачка относительно исходной точки правого глаза.
        
            pupil_left = self.eye_left.pupil.x / (self.eye_left.center[0] * 2 - 10)
            pupil_right = self.eye_right.pupil.x / (self.eye_right.center[0] * 2 - 10)
            return (pupil_left + pupil_right) / 2
        else:
            # Возвращает None, если координаты зрачка не были обнаружены.
            return None

    def vertical_ratio(self):
        """Возвращает число от 0,0 до 1,0, указывающее
        вертикальное направление взгляда. Крайняя верхняя точка равна 0,0,
        центр равен 0,5, а крайняя нижняя точка равна 1,0

         Возвращаемые значения:
        float or None: Значение отражает вертикальное направление взгляда. 
                       Возвращает None, если координаты зрачков не были обнаружены.

        """
        # Проверяет, были ли обнаружены координаты зрачков обоих глаз.
        if self.pupils_located:
              # Рассчитывает относительное вертикальное положение левого и правого зрачков.
            pupil_left = self.eye_left.pupil.y / (self.eye_left.center[1] * 2 - 10)
            pupil_right = self.eye_right.pupil.y / (self.eye_right.center[1] * 2 - 10)
             # Возвращает среднее значение относительных вертикальных положений зрачков.
            return (pupil_left + pupil_right) / 2
        else:
            # Возвращает None, если координаты зрачка не были обнаружены.
            return None

    def is_right(self):
        """Возвращает значение true, если пользователь смотрит вправо"""
        if self.pupils_located:
            return self.horizontal_ratio() <= 0.40

    def is_left(self):
        """Возвращает значение true, если пользователь смотрит влево"""
        if self.pupils_located:
            return self.horizontal_ratio() >= 0.50

    def is_center(self):
        """Возвращает значение true, если пользователь смотрит по центру"""
        if self.pupils_located:
            return self.is_right() is not True and self.is_left() is not True
    @staticmethod
    def get_eye_points(self, face_landmarks, LEFT_EYE_INDICES, RIGHT_EYE_INDICES, frame_shape):
        # Собираем координаты точек для левого и правого глаз
        left_eye_points = [(int(face_landmarks.landmark[index].x * frame_shape[1]), 
                            int(face_landmarks.landmark[index].y * frame_shape[0])) 
                           for index in LEFT_EYE_INDICES]
        right_eye_points = [(int(face_landmarks.landmark[index].x * frame_shape[1]), 
                             int(face_landmarks.landmark[index].y * frame_shape[0])) 
                            for index in RIGHT_EYE_INDICES]
        return left_eye_points, right_eye_points
    
    def determine_eye_sector(pupil, eye_points):
        # Считаем средние координаты для левой, правой, верхней и нижней точек глаза
        leftmost = min(eye_points, key=lambda p: p[0])[0]
        rightmost = max(eye_points, key=lambda p: p[0])[0]
        topmost = min(eye_points, key=lambda p: p[1])[1]
        bottommost = max(eye_points, key=lambda p: p[1])[1]

        eye_center = ((leftmost + rightmost) / 2, (topmost + bottommost) / 2)

        # Нормализуем положение зрачка относительно центра глаза
        dx = pupil[0] - eye_center[0]
        dy = pupil[1] - eye_center[1]

        # Вычисляем процентное отклонение зрачка от центра глаза
        # в диапазоне от -1 (левый/верхний край) до 1 (правый/нижний край)
        relative_dx = dx / (rightmost - eye_center[0]) if dx > 0 else dx / (eye_center[0] - leftmost)
        relative_dy = dy / (bottommost - eye_center[1]) if dy > 0 else dy / (eye_center[1] - topmost)

        # Выбираем пороговые значения для определения сектора
        horizontal_zones = [-0.5, -0.25, 0.25, 0.5]
        vertical_zones = [-0.5, -0.25, 0.25, 0.5]

        # Определяем сектор на основе относительного отклонения
        horizontal_sector = sum(relative_dx > x for x in horizontal_zones)
        vertical_sector = sum(relative_dy > y for y in vertical_zones)

        # Преобразуем в индекс сетки от 0 до 3
        horizontal_sector = max(0, min(horizontal_sector, 3))
        vertical_sector = max(0, min(vertical_sector, 3))

        return horizontal_sector, vertical_sector


    def is_blinking(self):
        """Возвращает значение true, если пользователь закрыл глаза"""
        if self.pupils_located:
            blinking_ratio = (self.eye_left.blinking + self.eye_right.blinking) / 2
            return blinking_ratio > 3.8
    @staticmethod
    def draw_eye_contours(self, frame, left_eye_points, right_eye_points):
        # Рисуем контур вокруг левого глаза
        cv2.polylines(frame, [np.array(left_eye_points, dtype=np.int32)], 
                      isClosed=True, color=(0, 255, 0), thickness=1)
        # Рисуем контур вокруг правого глаза
        cv2.polylines(frame, [np.array(right_eye_points, dtype=np.int32)], 
                      isClosed=True, color=(0, 255, 0), thickness=1)
    
    def annotated_frame(self):
        """Возвращает основной кадр с выделенными зрачками
         Возвращаемые значения:
        numpy.ndarray: Копия исходного кадра с добавленными линиями, обозначающими положение зрачков.
        """
         # Создает копию исходного кадра.
        frame = self.frame.copy()
        # Проверяет, были ли обнаружены координаты зрачков обоих глаз.
        if self.pupils_located:
            # Задает цвет линий для выделения зрачков (зеленый).
            color = (0, 255, 0)
             # Получает координаты левого и правого зрачков.
            x_left, y_left = self.pupil_left_coords()
            x_right, y_right = self.pupil_right_coords()
             # Рисует линии, обозначающие положение левого и правого зрачков.
            cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
            cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
            cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
            cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color)
        
        # Индексы ключевых точек для левого и правого глаз по документации Mediapipe
        LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        results = face_mesh.process(frame)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Получаем координаты точек для глаз
                left_eye_points, right_eye_points = GazeTracking.get_eye_points(self, face_landmarks, LEFT_EYE_INDICES, RIGHT_EYE_INDICES, frame.shape)
                # Рисуем контуры вокруг глаз
                GazeTracking.draw_eye_contours(self, frame, left_eye_points, right_eye_points)
                
                
        left_pupil = GazeTracking.pupil_left_coords(self)
        right_pupil = GazeTracking.pupil_right_coords(self)

        left_eye_points, right_eye_points = GazeTracking.get_eye_points(self, face_landmarks, LEFT_EYE_INDICES, RIGHT_EYE_INDICES, frame.shape)
       
        #left_eye_zone = GazeTracking.determine_eye_sector(left_pupil, left_eye_points)
        #right_eye_zone = GazeTracking.determine_eye_sector(right_pupil, right_eye_points)

        #print(f"Левый глаз смотрит в вертикальную зону {left_eye_zone[0]}, горизонтальную зону {left_eye_zone[1]}")
        #print(f"Правый глаз смотрит в вертикальную зону {right_eye_zone[0]}, горизонтальную зону {right_eye_zone[1]}")
        return frame
