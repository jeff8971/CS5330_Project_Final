# -*- coding: utf-8 -*-
"""
Filename: test_MTCNN_VGG.py
Author: Yuan Zhao
Email: zhao.yuan2@northeastern.edu
Description: This script combines MTCNN for real-time face detection with a VGG-style convolutional neural network for emotion recognition.
             It operates on live video streams, identifying faces and classifying their emotional expressions into categories such as anger, happiness, and sadness.
             Detected emotions are displayed as overlays on the video feed with corresponding labels.
Date: 2024-04-09
"""


import cv2
import torch
import torch.nn as nn
import numpy as np
from facenet_pytorch import MTCNN

def preprocess_input(images):
    """ Normalize image pixels from 0-255 to 0-1 """
    return images / 255.0

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 7)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=True, device='cpu')

# Initialize the VGG-based emotion classifier
emotion_classifier = VGG()
emotion_classifier.load_state_dict(torch.load('./model/model_vgg.pkl', map_location=torch.device('cpu')))
emotion_classifier.eval()

# Emotion labels
emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}

# Setup video capture
video_capture = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.namedWindow('window_frame')

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Horizontal flip

    # Detect faces using MTCNN
    boxes, _ = mtcnn.detect(frame)
    for box in boxes:
        x, y, x2, y2 = box
        w = x2 - x
        h = y2 - y
        if w > 0 and h > 0:
            face = frame[int(y):int(y2), int(x):int(x2)]
            face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face_resized = cv2.resize(face_gray, (48, 48))
            face_normalized = preprocess_input(np.array(face_resized, dtype=np.float32)).reshape(1, 1, 48, 48)
            face_tensor = torch.from_numpy(face_normalized).type(torch.FloatTensor)

            with torch.no_grad():
                emotion_pred = emotion_classifier(face_tensor)
                emotion_arg = torch.argmax(emotion_pred).item()
                emotion = emotion_labels[emotion_arg]
                cv2.rectangle(frame, (int(x), int(y)), (int(x2), int(y2)), (84, 255, 159), 2)
                cv2.putText(frame, emotion, (int(x), int(y) - 10), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('window_frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
