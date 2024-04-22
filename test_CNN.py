# -*- coding: utf-8 -*-
import cv2
import torch
import torch.nn as nn
import numpy as np
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

        # First convolution and pooling
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1,
                      padding=1),  # Convolution layer
            nn.BatchNorm2d(num_features=64),  # Normalization
            nn.RReLU(inplace=True),  # Activation function
            nn.MaxPool2d(kernel_size=2, stride=2),  # Max pooling
        )

        # Second convolution and pooling
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Third convolution and pooling
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Apply Gaussian weight initialization
        self.conv1.apply(gaussian_weights_init)
        self.conv2.apply(gaussian_weights_init)
        self.conv3.apply(gaussian_weights_init)

        # Fully connected layers
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

    # Forward propagation
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # Flatten the data
        x = x.view(x.shape[0], -1)
        y = self.fc(x)
        return y


# Initialize models and load pretrained weights
face_detection = cv2.CascadeClassifier(
    'dataset/haarcascade_frontalface_default.xml')
emotion_classifier = torch.load('./model/model_cnn.pkl',
                                map_location=torch.device('cpu'))
emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad',
                  5: 'surprise', 6: 'neutral'}

# Setup video capture
video_capture = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.namedWindow('window_frame')

# Emotion tracking
emotion_window = []
last_display_time = datetime.now()

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Horizontal flip
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray, scaleFactor=1.1,
                                            minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = cv2.resize(gray[y:y + h, x:x + w], (48, 48))
        face = preprocess_input(face).reshape(1, 1, 48, 48)
        face_tensor = torch.from_numpy(face).type(torch.FloatTensor)

        with torch.no_grad():
            emotion_pred = emotion_classifier(face_tensor)
            emotion_arg = torch.argmax(emotion_pred).item()
            emotion = emotion_labels[emotion_arg]

        emotion_window.append(emotion)

        # Ensure the detection box displays for 2 seconds
        if datetime.now() - last_display_time < timedelta(seconds=2):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (84, 255, 159), 2)
            cv2.putText(frame, emotion, (x, y - 30), font, 0.7, (0, 0, 255), 1,
                        cv2.LINE_AA)

    last_display_time = datetime.now()

    cv2.imshow('window_frame', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('s'):  # Save screenshot when 's' key is pressed
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f'screenshot_{timestamp}.png', frame)

video_capture.release()
cv2.destroyAllWindows()