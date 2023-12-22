from __future__ import division
import os
import cv2
import dlib
from .eye import Eye
from .calibration import Calibration


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

        # _predictor используется для получения лицевых ориентиров данного лица
        cwd = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.abspath(os.path.join(cwd, "trained_models/shape_predictor_68_face_landmarks.dat"))
        self._predictor = dlib.shape_predictor(model_path)

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

        try:
            # Использует предиктор для получения точек landmarks для первого обнаруженного лица.
            landmarks = self._predictor(frame, faces[0])
             # Инициализирует объекты Eye для левого и правого глаза с использованием landmarks.
            self.eye_left = Eye(frame, landmarks, 0, self.calibration)
            self.eye_right = Eye(frame, landmarks, 1, self.calibration)

        except IndexError:
             # В случае отсутствия обнаруженных лиц, устанавливает значения eye_left и eye_right в None.
        
            self.eye_left = None
            self.eye_right = None

    def refresh(self, frame):
        """Обновляет кадр и анализирует его.

        Аргументы:
            фрейм (numpy.ndarray): Фрейм для анализа
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
            return self.horizontal_ratio() <= 0.65

    def is_left(self):
        """Возвращает значение true, если пользователь смотрит влево"""
        if self.pupils_located:
            return self.horizontal_ratio() >= 0.65

    def is_center(self):
        """Возвращает значение true, если пользователь смотрит по центру"""
        if self.pupils_located:
            return self.is_right() is not True and self.is_left() is not True

    def is_blinking(self):
        """Возвращает значение true, если пользователь закрыл глаза"""
        if self.pupils_located:
            blinking_ratio = (self.eye_left.blinking + self.eye_right.blinking) / 2
            return blinking_ratio > 3.8

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

        return frame
