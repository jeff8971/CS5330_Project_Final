# -*- coding: utf-8 -*-
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from statistics import mode


# Normalize facial data by mapping pixel values from 0-255 to 0-1
def preprocess_input(images):
    """ Preprocess input by dividing the pixel values by 255
    # Arguments: images or image of any shape
    # Returns: images or image with normalized pixel values
    """
    images = images / 255.0
    return images


class ResNet(nn.Module):
    def __init__(self, *args):
        super(ResNet, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class GlobalAvgPool2d(nn.Module):
    # Global average pooling layer can be achieved by setting the pool window shape to the input's height and width
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])


# Residual neural network block
class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                   stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)


def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels  # The number of channels in the first block must match the input channels
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True,
                                stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)


resnet = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
resnet.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
resnet.add_module("resnet_block2", resnet_block(64, 128, 2))
resnet.add_module("resnet_block3", resnet_block(128, 256, 2))
resnet.add_module("resnet_block4", resnet_block(256, 512, 2))
resnet.add_module("global_avg_pool",
                  GlobalAvgPool2d())  # Output of GlobalAvgPool2d: (Batch, 512, 1, 1)
resnet.add_module("fc", nn.Sequential(ResNet(), nn.Linear(512, 7)))

# Built-in OpenCV facial recognition classifier
detection_model_path = 'model/haarcascade_frontalface_default.xml'

# Path to the trained model for facial expression recognition
classification_model_path = 'model/model_resnet.pkl'

# Load the face detection model
face_detection = cv2.CascadeClassifier(detection_model_path)

# Load the emotion recognition model
emotion_classifier = torch.load(classification_model_path)

frame_window = 10

# Emotion labels
emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad',
                  5: 'surprise', 6: 'neutral'}

emotion_window = []

# Start the webcam, 0 is for the default webcam
video_capture = cv2.VideoCapture(0)
# For recognizing from a video file
# video_capture = cv2.VideoCapture("video/example_dsh.mp4")
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.startWindowThread()
cv2.namedWindow('window_frame')

while True:
    # Read one frame
    _, frame = video_capture.read()
    frame = frame[:, ::-1, :]  # Horizontal flip, to suit selfie habit
    frame = frame.copy()
    # Get a grayscale image and create an image object in memory
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect all faces in the current frame
    faces = face_detection.detectMultiScale(gray, 1.3, 5)
    # For each detected face
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face, (255,0,0) is the color, 2 is the line width
        cv2.rectangle(frame, (x, y), (x + w, y + h), (84, 255, 159), 2)

        # Extract the face image
        face = gray[y:y + h, x:x + w]

        try:
            # Resize to (48, 48)
            face = cv2.resize(face, (48, 48))
        except:
            continue

        # Expand dimensions, reshape from (1, 48, 48, 1) to (1, 1, 48, 48)
        face = np.expand_dims(face, 0)
        face = np.expand_dims(face, 0)

        # Normalize facial data by mapping pixel values from 0-255 to 0-1
        face = preprocess_input(face)
        new_face = torch.from_numpy(face)
        new_face = new_face.float().requires_grad_(False)

        # Use the trained emotion recognition model to predict classification
        emotion_arg = np.argmax(
            emotion_classifier.forward(new_face).detach().numpy())
        emotion = emotion_labels[emotion_arg]

        emotion_window.append(emotion)

        if len(emotion_window) >= frame_window:
            emotion_window.pop(0)

        try:
            # Get the most frequently appearing emotion
            emotion_mode = mode(emotion_window)
        except:
            continue

        # Display the emotion label above the rectangle
        cv2.putText(frame, emotion_mode, (x, y - 30), font, 0.7, (0, 0, 255),
                    1, cv2.LINE_AA)

    try:
        # Display the image from memory to the screen
        cv2.imshow('window_frame', frame)
    except:
        continue

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
