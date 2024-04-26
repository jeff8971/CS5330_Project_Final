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

def initialize_data_loaders(batch_size):
    """Initializes data loaders for training and validation datasets."""
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

    train_dataset = torchvision.datasets.ImageFolder(root='face_images/vgg_train_set', transform=transforms_train)
    valid_dataset = torchvision.datasets.ImageFolder(root='face_images/vgg_valid_set', transform=transforms_valid)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader

class VGG(nn.Module):
    """VGG-like model for facial expression recognition."""
    def __init__(self, num_classes=7):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2)
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

def train_model(model, train_loader, valid_loader, epochs, learning_rate):
    """Train the VGG model."""
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        print(f'Epoch {epoch}/{epochs}')
        train_epoch(model, train_loader, optimizer, criterion)
        validate_epoch(model, valid_loader, criterion)
        if epoch % 10 == 0 or epoch == epochs:
            torch.save(model.state_dict(), f'model_vgg_epoch_{epoch}.pth')

def train_epoch(model, train_loader, optimizer, criterion):
    """Train for one epoch."""
    model.train()
    total_loss, correct, total = 0, 0, 0
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
    print(f'Training - Loss: {total_loss / len(train_loader):.4f}, Accuracy: {correct / total:.4f}')

def validate_epoch(model, valid_loader, criterion):
    """Validate after one epoch."""
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(valid_loader, desc="Validation"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    print(f'Validation - Loss: {total_loss / len(valid_loader):.4f}, Accuracy: {correct / total:.4f}')

if __name__ == '__main__':
    DEVICE = torch.device('cpu')
    BATCH_SIZE = 128
    LEARNING_RATE = 0.01
    EPOCHS = 60
    train_loader, valid_loader = initialize_data_loaders(BATCH_SIZE)
    model = VGG().to(DEVICE)
    train_model(model, train_loader, valid_loader, EPOCHS, LEARNING_RATE)
