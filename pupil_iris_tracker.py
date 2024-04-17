import cv2
import mediapipe as mp
from tkinter import *
from PIL import Image, ImageTk
import threading
from gaze_tracking import GazeTracking

gaze = GazeTracking()



# Функция для обновления изображения в виджете Label
def update_frame():
    ret, frame = cap.read()
    if ret:
        # Переворачиваем изображение для натурального отображения
        frame = cv2.flip(frame, 1)

        # We send this frame to GazeTracking to analyze it
        gaze.refresh(frame)

        frame = gaze.annotated_frame()
        text = ""

        if gaze.is_blinking():
            text = "Blinking"
        elif gaze.is_right():
            text = "Looking right"
        elif gaze.is_left():
            text = "Looking left"
        elif gaze.is_center():
            text = "Looking center"
        
        cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
        left_pupil = gaze.pupil_left_coords()
        right_pupil = gaze.pupil_right_coords()
        cv2.putText(frame, "Left pupil:  " + str(left_pupil), (30, 90), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        cv2.putText(frame, "Right pupil: " + str(right_pupil), (30, 135), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

        # Конвертируем изображение из BGR в RGB
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(10, update_frame)

# Функция для завершения работы с камерой и закрытия окна
def on_closing():   
    global cap
    cap.release()
    cv2.destroyAllWindows()
    window.destroy()

# Инициализируем камеру
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Ошибка: Не удалось инициализировать камеру.")
    exit()

# Создаем главное окно Tkinter
window = Tk()
window.title("Видеопоток с анализом лицевых ориентиров")

# Виджет для отображения видео
lmain = Label(window)
lmain.pack()

# Запускаем обновление кадров в отдельном потоке, чтобы избежать блокировки GUI
thread = threading.Thread(target=update_frame)
thread.start()

# Устанавливаем обработчик закрытия окна
window.protocol("WM_DELETE_WINDOW", on_closing)

window.mainloop()
