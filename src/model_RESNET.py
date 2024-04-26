#!/usr/bin/env python3
"""
Filename: model_RESNET.py
Author: Yuan Zhao
Email: zhao.yuan2@northeastern.edu
Description: This script implements a ResNet model for facial expression recognition. It loads image data with specified transformations,
             constructs a ResNet architecture, and trains the model using stochastic gradient descent. The training and validation
             progress is monitored with loss and accuracy metrics, and the model state is saved upon completion. The script is
             configured to run on the CPU and uses a custom data loading strategy for handling image datasets.
Date: 2024-04-09
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

BATCH_SIZE = 128
LR = 0.01
EPOCH = 60
DEVICE = torch.device('cpu')

path_train = 'face_images/resnet_train_set'
path_valid = 'face_images/resnet_valid_set'

# Define transformations for the training dataset
transforms_train = transforms.Compose([
    transforms.Grayscale(),  # ImageFolder default is to expand as three channels, converting back to grayscale
    transforms.RandomHorizontalFlip(),  # Randomly flip the images horizontally
    transforms.ColorJitter(brightness=0.5, contrast=0.5),  # Randomly adjust brightness and contrast
    transforms.ToTensor()
])

# Define transformations for the validation dataset
transforms_valid = transforms.Compose([
    transforms.Grayscale(),  # Convert to grayscale
    transforms.ToTensor()
])

# Load datasets
data_train = torchvision.datasets.ImageFolder(root=path_train, transform=transforms_train)
data_valid = torchvision.datasets.ImageFolder(root=path_valid, transform=transforms_valid)

# Data loaders for training and validation sets
train_set = torch.utils.data.DataLoader(dataset=data_train, batch_size=BATCH_SIZE, shuffle=True)
valid_set = torch.utils.data.DataLoader(dataset=data_valid, batch_size=BATCH_SIZE, shuffle=False)


class ResNet(nn.Module):
    def __init__(self, *args):
        super(ResNet, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class GlobalAvgPool2d(nn.Module):
    # Global average pooling layer can be implemented by setting the pool window shape to the input's height and width
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])


# Residual block for ResNet
class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
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
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)

# Constructing the ResNet model
resnet = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
resnet.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
resnet.add_module("resnet_block2", resnet_block(64, 128, 2))
resnet.add_module("resnet_block3", resnet_block(128, 256, 2))
resnet.add_module("resnet_block4", resnet_block(256, 512, 2))
resnet.add_module("global_avg_pool", GlobalAvgPool2d())  # Output of GlobalAvgPool2d: (Batch, 512, 1, 1)
resnet.add_module("fc", nn.Sequential(ResNet(), nn.Linear(512, 7)))

model = resnet
model.to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
criterion = nn.CrossEntropyLoss()

train_loss = []
train_acc = []
valid_loss = []
valid_acc = []
y_pred = []

def train(model, device, dataset, optimizer, epoch):
    model.train()
    correct = 0
    for i, (x, y) in tqdm(enumerate(dataset)):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(y.view_as(pred)).sum().item()
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

    train_acc.append(correct / len(data_train))
    train_loss.append(loss.item())
    print(f"Epoch {epoch} Loss {loss:.4f} Accuracy {correct}/{len(data_train)} ({100 * correct / len(data_train):.0f}%)")

def valid(model, device, dataset):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i, (x, y) in tqdm(enumerate(dataset)):
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)
            pred = output.max(1, keepdim=True)[1]
            global y_pred
            y_pred += pred.view(pred.size()[0]).cpu().numpy().tolist()
            correct += pred.eq(y.view_as(pred)).sum().item()

    valid_acc.append(correct / len(data_valid))
    valid_loss.append(loss.item())
    print(f"Test Loss {loss:.4f} Accuracy {correct}/{len(data_valid)} ({100. * correct / len(data_valid):.0f}%)")

def RUN():
    for epoch in range(1, EPOCH + 1):
        train(model, device=DEVICE, dataset=train_set, optimizer=optimizer, epoch=epoch)
        valid(model, device=DEVICE, dataset=valid_set)
        torch.save(model, 'model/model_resnet.pkl')

if __name__ == '__main__':
    RUN()
