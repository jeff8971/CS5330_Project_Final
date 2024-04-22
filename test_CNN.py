# -*- coding: utf-8 -*-
import cv2
import torch
import torch.nn as nn
import numpy as np
from statistics import mode


# Normalize facial data by mapping pixel values from 0-255 to 0-1
def preprocess_input(images):
    """ Preprocess input by subtracting the train mean.
    # Arguments: images or image of any shape
    # Returns: images or image with subtracted train mean (129)
    """
    images = images / 255.0
    return images


def gaussian_weights_init(m):
    classname = m.__class__.__name__
    # Use string find; if not found, returns -1, indicating the substring does not exist
    if classname.find('Conv') != -1:
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


# Built-in OpenCV face recognition classifier
detection_model_path = 'dataset/haarcascade_frontalface_default.xml'
classification_model_path = './model/model_cnn.pkl'

# Load face detection model
face_detection = cv2.CascadeClassifier(detection_model_path)

# Load emotion recognition model
emotion_classifier = torch.load(classification_model_path)

frame_window = 10

# Emotion labels
emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad',
                  5: 'surprise', 6: 'neutral'}

emotion_window = []

# Activate the camera, 0 is the built-in laptop camera
video_capture = cv2.VideoCapture(0)
# Recognize from a video file
# video_capture = cv2.VideoCapture("video/example_dsh.mp4")
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.startWindowThread()
cv2.namedWindow('window_frame')

while True:
    # Read one frame
    _, frame = video_capture.read()
    frame = frame[:, ::-1, :]  # Horizontal flip, fits self-portrait habit
    frame = frame.copy()
    # Obtain the grayscale image and create an image object in memory
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect all faces in the current frame
    faces = face_detection.detectMultiScale(gray, 1.1, 5)
    # For all detected faces
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face, (255,0,0) is the color, 2 is the line width
        cv2.rectangle(frame, (x, y), (x + w, y + h), (84, 255, 159), 2)

        # Extract the face image
        face = gray[y:y + h, x:x + w]

        try:
            # Resize to (48,48)
            face = cv2.resize(face, (48, 48))
        except:
            continue

        # Expand dimensions, reshape from (1,48,48,1) to (1,1,48,48)
        face = np.expand_dims(face, 0)
        face = np.expand_dims(face, 0)

        # Normalize facial data by mapping pixel values from 0-255 to 0-1
        face = preprocess_input(face)
        new_face = torch.from_numpy(face)
        new_new_face = new_face.float().requires_grad_(False)

        # Use the trained emotion recognition model to predict the class
        emotion_arg = np.argmax(
            emotion_classifier.forward(new_new_face).detach().numpy())
        emotion = emotion_labels[emotion_arg]

        emotion_window.append(emotion)

        if len(emotion_window) >= frame_window:
            emotion_window.pop(0)

        try:
            # Get the most frequently appearing class
            emotion_mode = mode(emotion_window)
        except:
            continue

        # Display the classification text above the rectangle
        cv2.putText(frame, emotion_mode, (x, y - 30), font, .7, (0, 0, 255), 1,
                    cv2.LINE_AA)

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
