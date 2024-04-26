# -*- coding: utf-8 -*-

"""
Filename: test_MTCNN_CNN.py
Author: Yuan Zhao
Email: zhao.yuan2@northeastern.edu
Description: This script leverages the MTCNN for real-time face detection and a custom CNN model for emotion recognition from video streams.
             It features live emotion classification, the ability to record segments of the video, and real-time tracking of facial features using OpenCV's KCF tracker.
Date: 2024-04-09
"""

import cv2
import torch
import torch.nn as nn
import numpy as np
from facenet_pytorch import MTCNN
from statistics import mode
from datetime import datetime, timedelta

# Normalize facial data by mapping pixel values from 0-255 to 0-1
def preprocess_input(images):
    return images / 255.0

def gaussian_weights_init(m):
    if 'Conv' in m.__class__.__name__:
        m.weight.data.normal_(0.0, 0.04)

class FaceCNN(nn.Module):
    # Initialize network structure
    def __init__(self):
        super(FaceCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv1.apply(gaussian_weights_init)
        self.conv2.apply(gaussian_weights_init)
        self.conv3.apply(gaussian_weights_init)
        self.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=256 * 6 * 6, out_features=4096),
            nn.RReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=1024),
            nn.RReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=256),
            nn.RReLU(inplace=True),
            nn.Linear(in_features=256, out_features=7),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        y = self.fc(x)
        return y

# Initialize models
mtcnn = MTCNN(keep_all=True, device='cpu')
emotion_classifier = torch.load('./model/model_cnn.pkl', map_location=torch.device('cpu'))
emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}
tracker = cv2.TrackerKCF_create()

# Setup video capture
video_capture = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.namedWindow('window_frame')
bbox = None

# Video recording variables
record_flag = False
video_writer = None
start_time = None

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    if bbox is not None:
        # Use KCF Tracker
        ok, bbox = tracker.update(frame)
        if ok:
            x, y, w, h = map(int, bbox)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (84, 255, 159), 2)
    else:
        # Detect faces
        boxes, _ = mtcnn.detect(frame)
        if boxes is not None and len(boxes) > 0:
            bbox = boxes[0]
            x, y, x2, y2 = bbox
            w = int(x2 - x)
            h = int(y2 - y)
            x, y = int(x), int(y)
            bbox = (x, y, w, h)
            tracker.init(frame, bbox)

    if bbox is not None and w > 0 and h > 0:
        face = frame[y:y + h, x:x + w]
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, (48, 48), interpolation=cv2.INTER_AREA)
        face_preprocessed = preprocess_input(np.array(face_resized, dtype=np.float32)).reshape(1, 1, 48, 48)
        face_tensor = torch.from_numpy(face_preprocessed).type(torch.FloatTensor)

        with torch.no_grad():
            emotion_pred = emotion_classifier(face_tensor)
            emotion_arg = torch.argmax(emotion_pred).item()
            emotion = emotion_labels[emotion_arg]
            cv2.putText(frame, emotion, (x, y - 10), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    if record_flag:
        if video_writer is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_writer = cv2.VideoWriter(f'video_{timestamp}.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10.0, (frame.shape[1], frame.shape[0]))
            start_time = datetime.now()
        video_writer.write(frame)
        if datetime.now() - start_time >= timedelta(seconds=5):
            record_flag = False
            video_writer.release()
            video_writer = None

    cv2.imshow('window_frame', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('r'):
        record_flag = True
    elif key == ord('s'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f'screenshot_{timestamp}.png', frame)

video_capture.release()
if video_writer:
    video_writer.release()
cv2.destroyAllWindows()