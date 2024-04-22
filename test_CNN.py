# -*- coding: utf-8 -*-
import cv2
import torch
import torch.nn as nn
import numpy as np
from statistics import mode
import time


# Define a class for the application
class EmotionDetectionApp:
    def __init__(self):
        self.detection_model_path = 'dataset/haarcascade_frontalface_default.xml'
        self.classification_model_path = './model/model_cnn.pkl'
        self.frame_window = 10
        self.emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}
        self.emotion_window = []
        self.last_detection_time = None
        self.face_detection = cv2.CascadeClassifier(self.detection_model_path)
        self.emotion_classifier = torch.load(self.classification_model_path)
        self.video_capture = cv2.VideoCapture(0)
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.startWindowThread()
        cv2.namedWindow('window_frame')

    def preprocess_input(self, images):
        return images / 255.0

    def capture_frame(self):
        ret, frame = self.video_capture.read()
        if not ret:
            return None
        frame = frame[:, ::-1, :]  # Horizontal flip for self-portrait view
        return frame.copy()

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.face_detection.detectMultiScale(gray, 1.1, 5)

    def display_emotions(self, frame):
        faces = self.detect_faces(frame)
        for (x, y, w, h) in faces:
            if self.last_detection_time is None or time.time() - self.last_detection_time > 2:
                face = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, (48, 48))
                face = np.expand_dims(np.expand_dims(face, 0), 0)
                face = self.preprocess_input(face)
                face_tensor = torch.from_numpy(face).float().requires_grad_(False)
                emotion_pred = self.emotion_classifier(face_tensor)
                emotion_arg = torch.argmax(emotion_pred.detach()).item()
                emotion = self.emotion_labels[emotion_arg]
                self.emotion_window.append(emotion)
                self.last_detection_time = time.time()

            cv2.rectangle(frame, (x, y), (x+w, y+h), (84, 255, 159), 2)
            emotion_mode = mode(self.emotion_window) if self.emotion_window else ''
            cv2.putText(frame, emotion_mode, (x, y - 30), self.font, .7, (0, 0, 255), 1, cv2.LINE_AA)

    def run(self):
        while True:
            frame = self.capture_frame()
            if frame is None:
                continue
            self.display_emotions(frame)
            cv2.imshow('window_frame', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite('screenshot.png', frame)  # Save screenshot

        self.video_capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    app = EmotionDetectionApp()
    app.run()
