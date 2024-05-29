from flask import Flask, Response, render_template
from flask_socketio import SocketIO, emit

import cv2
import json
import base64
import numpy as np
import PIL
from PIL import Image
from hsemotion.facial_emotions import HSEmotionRecognizer

import time
from time import sleep
# from IPython.display import clear_output

app = Flask(__name__)
socketio = SocketIO(app)

def load_model():
    model_name='enet_b0_8_best_afew'
    fer=HSEmotionRecognizer(model_name=model_name,device='cpu')
    return fer

# Глобальная переменная для контроля активности gen_frames
is_running = False
# Загружаем модель
model = load_model()

def gen_frames():
    # Загружаем модель
    # model = load_model()

    global is_running
    global model

    # Захватываем видео с камеры
    cap = cv2.VideoCapture(0)
    
    timestamp = 0.01
    last_checkpoint_time = time.time()

    while(is_running):
        success, frame = cap.read()
        
        if not success:
            break
        else:
            # Преобразуем изображение в RGB (кажется, это и не требуется)
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # убрать в будущем
            cv2.imshow('Video', frame)

            if time.time() - last_checkpoint_time > timestamp:
                last_checkpoint_time = time.time()

                # Предсказываем эмоцию
                emotion, _ = model.predict_emotions(frame)

                # clear_output(wait=True)
                # print(emotion)
                _, buffer = cv2.imencode('.jpg', frame)
                frame = base64.b64encode(buffer).decode('utf-8')
                # Создание словаря с данными для отправки
                data_to_send = {
                    "frame": f"data:image/jpeg;base64,{frame}",
                    "emotion": emotion,
                }

                # Преобразование словаря в JSON строку
                json_data = json.dumps(data_to_send)

                # Отправка JSON строки через веб-сокет
                # await websocket.send(json_data)
                socketio.emit('emotion', json_data)
                # sleep(0.001)
            # cv2.imshow('frame',gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def test_connect():
    print('Client connected')

@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected')

@socketio.on('get_emotion')
def handle_get_emotion():
    global is_running
    is_running = True
    gen_frames()
    # model = load_model()
    # cap = cv2.VideoCapture(0)
    # success, frame = cap.read()
    # if success:
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # emotion, _ = model.predict_emotions(gray)
        # socketio.emit('emotion', {'emotion': emotion})

@socketio.on('end_emotion')
def handle_get_emotion():
    global is_running
    is_running = False

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8000)