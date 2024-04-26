#!/usr/bin/env python3
"""
Filename: img_data_process.py
Author: Yuan Zhao
Email: zhao.yuan2@northeastern.edu
Description: This script processes facial expression data from a CSV file, segregating images into
             training and validation datasets based on their usage tag. It creates directories for
             each emotion category and saves the images into these directories, facilitating organized
             access for training and validating machine learning models.
Date: 2024-04-09
"""

import numpy as np
import pandas as pd
from PIL import Image
import os

# Define the paths for the training and validation datasets
train_path = 'dataset/img/train/'
valid_path = 'dataset/img/test/'
data_path = 'dataset/csv/src_csv/fer2013.csv'  # Path to the source CSV file containing image data

def make_dir():
    """ Create directories for each of the 7 emotion categories in both training and validation datasets. """
    for emotion in range(7):  # Loop over each emotion category (0 to 6)
        os.makedirs(os.path.join(train_path, str(emotion)), exist_ok=True)  # Create training directory for each emotion
        os.makedirs(os.path.join(valid_path, str(emotion)), exist_ok=True)  # Create validation directory for each emotion

def save_images():
    """ Read the CSV file, convert image data from pixels to images, and save them in their respective directories. """
    df = pd.read_csv(data_path)  # Read the CSV file into a DataFrame
    train_index, valid_index = [1] * 7, [1] * 7  # Initialize counters for image filenames for both training and validation

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        emotion, image_data, usage = row['emotion'], row['pixels'], row['Usage']  # Extract emotion, image data, and usage type
        image_array = np.array(list(map(int, image_data.split()))).reshape(48, 48)  # Convert pixel string to an array and reshape
        image = Image.fromarray(image_array).convert('L')  # Convert the array to an 8-bit grayscale image

        # Save the image in the appropriate directory based on whether it's for training or validation
        if usage == 'Training':
            file_path = os.path.join(train_path, str(emotion), f'{train_index[emotion]}.jpg')  # Define file path for training image
            train_index[emotion] += 1  # Increment the file name index for the training set
        else:
            file_path = os.path.join(valid_path, str(emotion), f'{valid_index[emotion]}.jpg')  # Define file path for validation image
            valid_index[emotion] += 1  # Increment the file name index for the validation set
        image.save(file_path)  # Save the image

# Execute the directory creation and image saving functions
make_dir()
save_images()
