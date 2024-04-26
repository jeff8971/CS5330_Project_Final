#!/usr/bin/env python3
"""
Filename: image_to_csv_mapping.py
Author: Yuan Zhao
Email: zhao.yuan2@northeastern.edu
Description: This script organizes images into specific directories and maps them to emotional labels from a CSV file.
             It is designed to facilitate the preparation of image datasets for facial expression recognition tasks.
Date: 2024-04-09
"""

import os
import pandas as pd
import shutil

def separate_images(source_dir, dest_dir1, dest_dir2):
    """Separate images into two different directories based on their index."""
    os.makedirs(dest_dir1, exist_ok=True)  # Ensure the destination directory 1 exists
    os.makedirs(dest_dir2, exist_ok=True)  # Ensure the destination directory 2 exists

    filenames = os.listdir(source_dir)  # List all files in the source directory
    for filename in filenames:
        index = int(os.path.splitext(filename)[0])  # Extract index from the filename
        if 0 <= index <= 23999:
            shutil.move(os.path.join(source_dir, filename), os.path.join(dest_dir1, filename))
        elif 24000 <= index <= 28708:
            shutil.move(os.path.join(source_dir, filename), os.path.join(dest_dir2, filename))

def image_emotion_mapping(path):
    """Map each image in a directory to its corresponding emotion from a CSV file."""
    df_emotion = pd.read_csv('./dataset/csv/rst_csv/emotion.csv', header=None)  # Load emotion data
    files_dir = os.listdir(path)  # List all files in the specified folder
    path_list = []  # Initialize list to store image paths
    emotion_list = []  # Initialize list to store corresponding emotions

    for file_dir in files_dir:
        if os.path.splitext(file_dir)[1] == ".jpg":  # Process only jpg images
            path_list.append(file_dir)
            index = int(os.path.splitext(file_dir)[0])  # Extract the index
            emotion_list.append(df_emotion.iat[index, 0])  # Append emotion to list

    df = pd.DataFrame({'path': path_list, 'emotion': emotion_list})  # Create DataFrame
    df.to_csv(os.path.join(path, 'image_emotion.csv'), index=False, header=False)  # Save DataFrame to CSV

def main():
    source_directory = './dataset/face_images'
    train_set_path = './dataset/img/train_set'
    test_set_path = './dataset/img/test_set'
    separate_images(source_directory, train_set_path, test_set_path)  # Separate images into training and test directories
    image_emotion_mapping(train_set_path)  # Map training images to emotions
    image_emotion_mapping(test_set_path)  # Map test images to emotions

if __name__ == "__main__":
    main()
