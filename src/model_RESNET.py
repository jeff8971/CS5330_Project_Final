#!/usr/bin/env python3
"""
Filename: model_RESNET.py
Author: Yuan Zhao
Email: zhao.yuan2@northeastern.edu
Description: This script implements a ResNet model for facial expression recognition. It includes loading image data with transformations,
             constructing a ResNet architecture, training the model with stochastic gradient descent, and monitoring training and validation
             progress with loss and accuracy metrics. The model state is saved upon completion. The script runs on the CPU and uses a custom
             data loading strategy for image datasets.
Date: 2024-04-09
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.nn.functional as F

# Constants
BATCH_SIZE = 128
LEARNING_RATE = 0.01
EPOCHS = 60
DEVICE = torch.device('cpu')

# Paths
TRAIN_PATH = 'face_images/resnet_train_set'
VALID_PATH = 'face_images/resnet_valid_set'

# Transforms
transforms_train = transforms.Compose([
    transforms.Grayscale(),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.5, contrast=0.5),
    transforms.ToTensor(),
])

transforms_valid = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
])

# Datasets
train_dataset = torchvision.datasets.ImageFolder(root=TRAIN_PATH,
                                                 transform=transforms_train)
valid_dataset = torchvision.datasets.ImageFolder(root=VALID_PATH,
                                                 transform=transforms_valid)

# Data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE, shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=False)


class ResidualBlock(nn.Module):
    """Residual Block as used in ResNet architectures."""

    def __init__(self, in_channels, out_channels, stride=1, use_1x1conv=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                               stride=stride) if use_1x1conv else None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.conv3 is not None:
            identity = self.conv3(identity)
        out += identity
        return F.relu(out)


def make_resnet():
    """Constructs a ResNet model using predefined Residual blocks."""
    layers = []
    layers.append(nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)))

    in_channels = 64
    num_blocks_list = [2, 2, 2, 2]
    out_channels_list = [64, 128, 256, 512]
    strides_list = [1, 2, 2, 2]

    for out_channels, num_blocks, stride in zip(out_channels_list,
                                                num_blocks_list, strides_list):
        layers.append(ResidualBlock(in_channels, out_channels, stride=stride,
                                    use_1x1conv=(in_channels != out_channels)))
        in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(in_channels, out_channels))

    layers.append(nn.Sequential(
        GlobalAvgPool2d(),
        nn.Flatten(),
        nn.Linear(512, 7)
    ))

    return nn.Sequential(*layers).to(DEVICE)


# Model
model = make_resnet()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
criterion = nn.CrossEntropyLoss()


# Training and validation
def train_epoch(model, device, train_loader, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in tqdm(train_loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    print(f"Training Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")


def validate_epoch(model, device, valid_loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(valid_loader, desc="Validating",
                                   leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(valid_loader)
    accuracy = correct / total
    print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")


def main():
    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}/{EPOCHS}")
        train_epoch(model, DEVICE, train_loader, optimizer)
        validate_epoch(model, DEVICE, valid_loader)
        torch.save(model.state_dict(), 'resnet_model.pth')


if __name__ == '__main__':
    main()
