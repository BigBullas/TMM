# ПРИМЕР

from flask import Flask, Response, render_template
from flask_socketio import SocketIO, emit

import cv2
import numpy as np
from PIL import Image
from hsemotion.facial_emotions import HSEmotionRecognizer

import time

app = Flask(__name__)
socketio = SocketIO(app)

def load_model():
    model_name='enet_b0_8_best_afew'
    fer=HSEmotionRecognizer(model_name=model_name,device='cpu')
    return fer

# Глобальная переменная для контроля активности gen_frames
is_running = False

def gen_frames():
    global is_running
    model = load_model()
    cap = cv2.VideoCapture(0)
    
    while is_running:
        success, frame = cap.read()
        if not success:
            break
        else:
            if time.time() - last_checkpoint_time > timestamp:
                last_checkpoint_time = time.time()

                emotion, _ = model.predict_emotions(frame)
                print(emotion)
                socketio.emit('emotion', {'emotion': emotion})

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

@socketio.on('end_emotion')
def handle_end_emotion():
    global is_running
    is_running = False

if __name__ == '__main__':
    socketio.run(app, host='localhost', port=8000)
