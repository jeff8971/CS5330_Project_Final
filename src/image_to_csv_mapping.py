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

def create_directories(base_path):
    """Create directories for each of the 7 emotion categories in the specified base path."""
    for emotion in range(7):  # There are 7 emotions, labeled 0 to 6
        os.makedirs(os.path.join(base_path, str(emotion)), exist_ok=True)

def process_and_save_images():
    """Read the CSV file, convert pixel data to images, and save them in the respective directories."""
    df = pd.read_csv(data_path)  # Load image data
    file_counters = {emotion: 1 for emotion in range(7)}  # Initialize image counters for each emotion

    for index, row in df.iterrows():
        emotion, pixels, usage = row['emotion'], row['pixels'], row['Usage']
        image_array = np.fromstring(pixels, dtype=int, sep=' ').reshape(48, 48)  # Convert pixel string to numpy array and reshape
        image = Image.fromarray(image_array).convert('L')  # Convert array to grayscale image

        # Determine the directory based on usage and increment the respective counter
        directory = train_path if usage == 'Training' else valid_path
        file_path = os.path.join(directory, str(emotion), f'{file_counters[emotion]}.jpg')
        file_counters[emotion] += 1  # Increment the file name index for the respective emotion and usage

        image.save(file_path)  # Save the image file

def main():
    # Create directories for training and validation datasets
    create_directories(train_path)
    create_directories(valid_path)
    # Process and save images
    process_and_save_images()

if __name__ == "__main__":
    main()
