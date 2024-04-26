#!/usr/bin/env python3
"""
Filename: model_CNN.py
Author: Yuan Zhao
Email: zhao.yuan2@northeastern.edu
Description: This script defines and trains a convolutional neural network (CNN) for facial expression recognition. It includes
             the entire pipeline from loading and preprocessing image data to training the model and validating its performance.
             The script leverages PyTorch for model construction and training, and utilizes custom data loaders for managing
             image datasets.
Date: 2024-04-09
"""


import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import cv2


# Weight initialization function
def gaussian_weights_init(m):
    classname = m.__class__.__name__
    # Use string find; if not found, returns -1, so checking not equal to -1 indicates the substring exists
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.04)


# Function to validate model accuracy on the validation set
def validate(model, dataset, batch_size):
    val_loader = data.DataLoader(dataset, batch_size)
    result, num = 0.0, 0
    for images, labels in val_loader:
        pred = model.forward(images)
        pred = np.argmax(pred.data.numpy(), axis=1)
        labels = labels.data.numpy()
        result += np.sum((pred == labels))
        num += len(images)
    acc = result / num
    return acc


# Custom data loader class named FaceDataset by inheriting Dataset class
class FaceDataset(data.Dataset):
    '''
    The first thing to do is the class initialization. The previously created image-emotion table is needed when loading data.
    Therefore, in the initialization process, we need to read the data from the image-emotion table.
    Data is read using the pandas library, and then the read data is placed into a list or numpy array for easy indexing later.
    '''
    # Initialization
    def __init__(self, root):
        super(FaceDataset, self).__init__()
        self.root = root
        df_path = pd.read_csv(root + '\\image_emotion.csv', header=None, usecols=[0])
        df_label = pd.read_csv(root + '\\image_emotion.csv', header=None, usecols=[1])
        self.path = np.array(df_path)[:, 0]
        self.label = np.array(df_label)[:, 0]

    '''
    Next, we need to override the __getitem__() function, which loads the data.
    We have already obtained all the image addresses in the initialization part, and in this function, we will read data using these addresses.
    Since we are reading image data, we still rely on the OpenCV library.
    It should be noted that the previous visualization part restored the pixel values to face images and saved them as 3-channel gray images (all channels are exactly the same).
    Here we only need a single channel, so during image reading, even if the original image is already gray, we still need to add a parameter from cv2.COLOR_BGR2GRAY,
    to ensure the data read out is single-channel. After reading out, some basic image processing operations can be considered,
    such as noise reduction through Gaussian blur, image enhancement through histogram equalization (experiments have shown that histogram equalization is not very useful in this project, and Gaussian noise reduction even reduces accuracy, probably because the image resolution is originally low, and blurring makes almost nothing clear).
    The read data is 48X48, and the subsequent convolutional neural network in nn.Conv2d() API accepts data formats as (batch_size, channel, width, height), the channel of the image this time is 1, so we need to reshape 48X48 to 1X48X48.
    '''

    # Read a specific image, item is the index number
    def __getitem__(self, item):
        face = cv2.imread(self.root + '\\' + self.path[item])
        # Read single channel grayscale image
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        # Gaussian blur
        # face_Gus = cv2.GaussianBlur(face_gray, (3,3), 0)
        # Histogram equalization
        face_hist = cv2.equalizeHist(face_gray)
        # Pixel normalization
        face_normalized = face_hist.reshape(1, 48, 48) / 255.0 # To match the design of PyTorch's convolutional neural network API, need to reshape the original image
        # Data for training must be of tensor type
        face_tensor = torch.from_numpy(face_normalized) # Convert Python's numpy data type to PyTorch's tensor data type
        face_tensor = face_tensor.type('torch.FloatTensor') # Specify as 'torch.FloatTensor' type, otherwise it will cause a data type mismatch error when sent into the model
        label = self.label[item]
        return face_tensor, label

    '''
    Finally, the len() function is overridden to get the size of the dataset.
    self.path stores all the image names, and the size of the first dimension of self.path is the size of the dataset.
    '''
    # Get the number of samples in the dataset
    def __len__(self):
        return self.path.shape[0]

class FaceCNN(nn.Module):
    # Initialize network structure
    def __init__(self):
        super(FaceCNN, self).__init__()

        # First convolution, pooling
        self.conv1 = nn.Sequential(
            # Input channel number in_channels, output channel number (i.e., the number of channels of the convolutional kernel) out_channels, kernel size kernel_size, stride stride, symmetric zero-padding number padding
            # input:(batch_size, 1, 48, 48), output:(batch_size, 64, 48, 48), (48-3+2*1)/1+1 = 48
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1), # Convolutional layer
            nn.BatchNorm2d(num_features=64), # Normalization
            nn.RReLU(inplace=True), # Activation function
            # output(batch_size, 64, 24, 24)
            nn.MaxPool2d(kernel_size=2, stride=2), # Max pooling
        )

        # Second convolution, pooling
        self.conv2 = nn.Sequential(
            # input:(batch_size, 64, 24, 24), output:(batch_size, 128, 24, 24), (24-3+2*1)/1+1 = 24
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.RReLU(inplace=True),
            # output:(batch_size, 128, 12, 12)
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Third convolution, pooling
        self.conv3 = nn.Sequential(
            # input:(batch_size, 128, 12, 12), output:(batch_size, 256, 12, 12), (12-3+2*1)/1+1 = 12
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.RReLU(inplace=True),
            # output:(batch_size, 256, 6, 6)
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Weight initialization
        self.conv1.apply(gaussian_weights_init)
        self.conv2.apply(gaussian_weights_init)
        self.conv3.apply(gaussian_weights_init)

        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=256*6*6, out_features=4096),
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
        # Flatten data
        x = x.view(x.shape[0], -1)
        y = self.fc(x)
        return y

def train(train_dataset, val_dataset, batch_size, epochs, learning_rate, wt_decay):
    # Load data and split into batches
    train_loader = data.DataLoader(train_dataset, batch_size)
    # Build the model
    model = FaceCNN()
    # Loss function
    loss_function = nn.CrossEntropyLoss()
    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=wt_decay)
    # Learning rate decay
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    # Train by epoch
    for epoch in range(epochs):
        # Record loss value
        loss_rate = 0
        # scheduler.step() # Learning rate decay
        model.train() # Model training
        for images, emotion in train_loader:
            # Clear gradients
            optimizer.zero_grad()
            # Forward propagation
            output = model.forward(images)
            # Calculate error
            loss_rate = loss_function(output, emotion)
            # Backward propagation of the error
            loss_rate.backward()
            # Update parameters
            optimizer.step()

        # Print the loss for each epoch
        print('After {} epochs, the loss_rate is: '.format(epoch+1), loss_rate.item())
        if epoch % 5 == 0:
            model.eval() # Model evaluation
            acc_train = validate(model, train_dataset, batch_size)
            acc_val = validate(model, val_dataset, batch_size)
            print('After {} epochs, the acc_train is: '.format(epoch+1), acc_train)
            print('After {} epochs, the acc_val is: '.format(epoch+1), acc_val)

    return model


def main():
    # Instantiate the dataset (create dataset)
    train_dataset = FaceDataset(root='/dataset/img/train_set')
    test_dataset = FaceDataset(root='./dataset/img/test_set')
    # Hyperparameters can be specified by yourself
    model = train(train_dataset, test_dataset, batch_size=128, epochs=100, learning_rate=0.1, wt_decay=0)
    # Save the model
    torch.save(model, '../facial-expression-recognition-hexiang/model/model_cnn.pkl')


if __name__ == '__main__':
    main()
