# CS5330_Project_Final

[Project Repository](https://github.com/jeff8971/CS5330_Project_Final)

[Representation]() 

## Overview
This repository hosts the CS5330 final project, focusing on advanced image processing models. The project explores various deep learning models for image analysis tasks, offering an in-depth approach to understanding and manipulating image data.

## System Environment
- **IDE**: PyCharm or any preferred Python IDE
- Python 3.8+

## Project Structure
- `dataset/`: Contains image data used in experiments.
- `result_img/`: Stores output images from model predictions.
- `src/`: Source code files for model implementations and testing.
  - `model_CNN.py`: CNN model for training.
  - `model_RESNET.py`: ResNet model for training.
  - `model_VGG.py`: VGG model for training.
  - `test_MTCNN_CNN.py`: application for facial recognition by using CNN 
    model.
  - `test_MTCNN_RESNET.py`: application for facial recognition by using ResNet 
    model.
  - `test_MTCNN_VGG.py`: application for facial recognition by using VGG model.


## Features
- **Model Implementations**: CNN, ResNet, and VGG models for deep learning based image processing.
- **Image to CSV Mapping**: Script to map image data to CSV format for easier manipulation.
- **Test Scripts**: Dedicated scripts for testing each model under different scenarios.

## Getting Started
### Prerequisites
- numpy
- pandas
- matplotlib
- pytorch
- scikit-learn
- opencv-python
- Pillow

## Installation
- Clone the repository:
```git clone https://github.com/jeff8971/CS5330_Project_Final.git```
- Navigate to the project directory:
```cd CS5330_Project_Final```

## Usage
- Run the `src/model_CNN.py` script to train the CNN model.
- Run the `src/model_RESNET.py` script to train the ResNet model.
- Run the `src/model_VGG.py` script to train the VGG model.
- Run the test scripts in the `test_MTCNN_CNN.py` directory to use facial recognition models.

