# -*- coding: utf-8 -*-

"""
Filename: test_MTCNN_RESNET.py
Author: Yuan Zhao
Email: zhao.yuan2@northeastern.edu
Description: This script implements real-time facial expression recognition using MTCNN for face detection and a ResNet model for classification.
             It processes video input to detect faces, then classifies each detected face into one of seven emotion categories using the trained ResNet model.
             Results are displayed in real-time with labeled bounding boxes around detected faces. The system is designed to run on CPU with optimizations for real-time performance.
Date: 2024-04-09
"""

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from facenet_pytorch import MTCNN
from statistics import mode

# Normalize facial data by mapping pixel values from 0-255 to 0-1
def preprocess_input(images):
    return images / 255.0

# Define a Global Average Pooling 2D layer
class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])

# Residual Block Definition
class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride) if use_1x1conv else None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)

# Assemble ResNet using defined blocks
def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)

# Define the full ResNet model
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            resnet_block(64, 64, 2, first_block=True),
            resnet_block(64, 128, 2),
            resnet_block(128, 256, 2),
            resnet_block(256, 512, 2),
            GlobalAvgPool2d(),  # Global Average Pooling
            nn.Flatten(),
            nn.Linear(512, 7)  # Assuming 7 emotion categories
        )

    def forward(self, x):
        return self.network(x)

# Initialize the face detection model (MTCNN)
mtcnn = MTCNN(keep_all=True, device='cpu')

# Load the emotion recognition model (ResNet)
emotion_classifier = ResNet()
emotion_classifier.load_state_dict(torch.load('model/model_resnet.pkl', map_location=torch.device('cpu')))
emotion_classifier.eval()

# Emotion labels
emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}

# Initialize video capture
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
    if boxes is not None:
        for box in boxes:
            x, y, x2, y2 = map(int, box)
            w = x2 - x
            h = y2 - y
            face = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (48, 48))
            face = preprocess_input(np.array(face, dtype=np.float32)).reshape(1, 1, 48, 48)
            face_tensor = torch.from_numpy(face).type(torch.FloatTensor)

            with torch.no_grad():
                emotion_pred = emotion_classifier(face_tensor)
                emotion_arg = torch.argmax(emotion_pred).item()
                emotion = emotion_labels[emotion_arg]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (84, 255, 159), 2)
                cv2.putText(frame, emotion, (x, y - 10), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('window_frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
