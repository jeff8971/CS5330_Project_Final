#!/usr/bin/env python3
"""
Filename: model_VGG.py
Author: Yuan Zhao
Email: zhao.yuan2@northeastern.edu
Description: This script implements a VGG-like convolutional neural network for facial expression recognition.
             It prepares the image data with specific transformations, constructs the VGG model, and trains it
             using stochastic gradient descent with momentum. The script evaluates both training and validation
             performance, updates the model weights, and saves the trained model. Configured for CPU, this script
             is intended for use with facial image datasets organized in specified directory paths.
Date: 2024-04-09
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

# Constants for configuration
BATCH_SIZE = 128
LEARNING_RATE = 0.01
EPOCHS = 60
DEVICE = torch.device('cpu')

# Paths to datasets
TRAIN_PATH = 'face_images/vgg_train_set'
VALID_PATH = 'face_images/vgg_valid_set'

# Transformations for image preprocessing
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

# Loading datasets using ImageFolder
train_dataset = torchvision.datasets.ImageFolder(root=TRAIN_PATH, transform=transforms_train)
valid_dataset = torchvision.datasets.ImageFolder(root=VALID_PATH, transform=transforms_valid)

# DataLoader setup
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

class VGG(nn.Module):
    """
    VGG-like network for facial expression recognition.
    """
    def __init__(self, num_classes=7):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            # First VGG block
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Second VGG block
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Third VGG block
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 6 * 6, 4096), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

# Model instantiation
model = VGG()
model.to(DEVICE)

# Optimizer and loss function
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
criterion = nn.CrossEntropyLoss()

def train_epoch(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
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
    print(f'Training - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')

def validate_epoch(model, valid_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(valid_loader, desc="Validation"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / len(valid_loader)
    accuracy = correct / total
    print(f'Validation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')

def main():
    for epoch in range(1, EPOCHS + 1):
        print(f'Epoch {epoch}/{EPOCHS}')
        train_epoch(model, train_loader, optimizer, criterion)
        validate_epoch(model, valid_loader, criterion)
        # Save the model periodically or after last epoch
        if epoch == EPOCHS or epoch % 10 == 0:
            torch.save(model.state_dict(), f'model_vgg_epoch_{epoch}.pth')

if __name__ == '__main__':
    main()
