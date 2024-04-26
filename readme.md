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
  - `data_preprocessing.py`: Script for image data processing.
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

## Acknowledgements
The author would like to express sincere gratitude to Xiang He for the valuable resources provided through his open-source project on facial expression recognition. The contributions available at \url{https://github.com/hexiang10/facial-expression-recognition} have been instrumental in the advancement of this work. The willingness to share knowledge and tools openly has greatly facilitated the progress of this research.