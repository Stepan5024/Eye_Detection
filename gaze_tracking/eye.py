import math
import numpy as np
import cv2
from .pupil import Pupil


class Eye(object):
    """
   Этот класс создает новую рамку для выделения глаза и
инициирует обнаружение зрачка.
    """
# Индексы точек лица, соответствующих левому и правому глазу в landmarks.
    LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41]
    RIGHT_EYE_POINTS = [42, 43, 44, 45, 46, 47]

    def __init__(self, original_frame, landmarks, side, calibration):
        """Инициализирует объект Eye

        Аргументы:
            original_frame (numpy.ndarray): Исходный кадр с изображением лица
            landmarks (list): Список точек landmarks, определяющих особенности лица
            side (int): Сторона глаза (0 для левого, 1 для правого)
            calibration (tuple): Кортеж с коэффициентами калибровки для определения размера глаза
        """
        self.frame = None # Кадр для выделения глаза будет храниться здесь.
        self.origin = None # Начальные координаты глаза (верхний левый угол).
        self.center = None  # Координаты центра глаза.
        self.pupil = None # Объект, представляющий зрачок.
        self.landmark_points = None # Точки landmarks, определяющие глаз.
        # Вызывает метод _analyze для обработки и анализа глаза.
        self._analyze(original_frame, landmarks, side, calibration)

    @staticmethod
    def _middle_point(p1, p2):
        """Возвращает среднюю точку (x,y) между двумя точками

        Аргументы:
            p1 (dlib.point): Первая точка
            p2 (dlib.point): Вторая точка
        """
        x = int((p1.x + p2.x) / 2)
        y = int((p1.y + p2.y) / 2)
        return (x, y)

    def _isolate(self, frame, landmarks, points):
        """Выделите глаз, чтобы получилась рамка без другой части лица.

         Аргументы:
        frame (numpy.ndarray): Кадр, содержащий лицо
        landmarks (dlib.full_object_detection): Ориентиры лица для области лица
        points (list): Точки зрения (из 68 различных ориентиров)
        """
         # Создание массива точек для области лица на основе переданных ориентиров.
    
        region = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in points])
        region = region.astype(np.int32)
        self.landmark_points = region

       # Применение маски для выделения только глаза.
        height, width = frame.shape[:2]
        black_frame = np.zeros((height, width), np.uint8)
        mask = np.full((height, width), 255, np.uint8)
        cv2.fillPoly(mask, [region], (0, 0, 0))
        eye = cv2.bitwise_not(black_frame, frame.copy(), mask=mask)
        
        # Обрезка области глаза.
        margin = 5
        min_x = np.min(region[:, 0]) - margin
        max_x = np.max(region[:, 0]) + margin
        min_y = np.min(region[:, 1]) - margin
        max_y = np.max(region[:, 1]) + margin
 
        self.frame = eye[min_y:max_y, min_x:max_x]
        self.origin = (min_x, min_y)
        # Вычисление центра глаза на основе обрезанной рамки.
        height, width = self.frame.shape[:2]
        self.center = (width / 2, height / 2)

    def _blinking_ratio(self, landmarks, points):
        """Вычисляет соотношение, которое может указывать, закрыт глаз или нет.
        Это деление ширины глаза на его высоту.

         Аргументы:
        landmarks (dlib.full_object_detection): Ориентиры лица для области лица
        points (list): Точки зрения (из 68 различных ориентиров)

        Возвращаемые значения:
        float or None: Вычисленное соотношение. Возвращает None, если произошло деление на ноль.
        """
         # Получение координат углов глаза и средних точек верхней и нижней части глаза.
    
        left = (landmarks.part(points[0]).x, landmarks.part(points[0]).y)
        right = (landmarks.part(points[3]).x, landmarks.part(points[3]).y)
        top = self._middle_point(landmarks.part(points[1]), landmarks.part(points[2]))
        bottom = self._middle_point(landmarks.part(points[5]), landmarks.part(points[4]))
        # Вычисление ширины и высоты глаза с использованием теоремы Пифагора.
    
        eye_width = math.hypot((left[0] - right[0]), (left[1] - right[1]))
        eye_height = math.hypot((top[0] - bottom[0]), (top[1] - bottom[1]))

        try:
            # Вычисление соотношения ширины к высоте.
            ratio = eye_width / eye_height
        except ZeroDivisionError:
             # В случае деления на ноль возвращает None.
            ratio = None

        return ratio

    def _analyze(self, original_frame, landmarks, side, calibration):
        """Обнаруживает и выделяет глаз в новом кадре, отправляет данные на калибровку
        и инициализирует объект зрачка.

        Аргументы:
        original_frame (numpy.ndarray): Кадр, переданный пользователем
        landmarks (dlib.full_object_detection): Ориентиры лица для области лица
        side (int): Указывает, является ли это левым глазом (0) или правым глазом (1)
        calibration (calibration.Calibration): Объект, управляющий пороговым значением бинаризации
        """
         # Определение точек зрения в зависимости от стороны глаза.
        if side == 0:
            points = self.LEFT_EYE_POINTS
        elif side == 1:
            points = self.RIGHT_EYE_POINTS
        else:
            return
 # Вычисление соотношения моргания и выделение области глаза.   
        self.blinking = self._blinking_ratio(landmarks, points)
        self._isolate(original_frame, landmarks, points)
  # Если калибровка не завершена, оценка данных для калибровки.
        if not calibration.is_complete():
            calibration.evaluate(self.frame, side)
  # Получение порогового значения для бинаризации и инициализация объекта зрачка.
    
        threshold = calibration.threshold(side)
        self.pupil = Pupil(self.frame, threshold)
