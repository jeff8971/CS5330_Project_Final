#!/usr/bin/env python3
"""
Filename: csv_view.py
Author: Yuan Zhao
Email: zhao.yuan2@northeastern.edu
Description: This script loads pixel data from a CSV file and saves it as individual JPEG images.
             It is particularly useful for processing machine learning datasets where images are stored as flattened pixel arrays.
Date: 2024-04-09
"""

import cv2
import numpy as np
import os


def save_images_from_csv(csv_file_path, output_path):
    """
    Loads pixel data from a CSV file and saves it as individual JPEG images.

    Args:
    csv_file_path (str): Path to the CSV file containing flattened pixel arrays.
    output_path (str): Directory where JPEG images will be saved.
    """
    # Ensure the directory exists; create it if it does not
    os.makedirs(output_path, exist_ok=True)

    # Load pixel data from the CSV file
    data = np.loadtxt(csv_file_path)

    # Loop through each row of the data, each row represents one image
    for i in range(data.shape[0]):
        # Reshape the flat array into a 48x48 image, assuming each image is 48x48 pixels
        face_array = data[i].reshape((48, 48))
        # Construct the file path where the image will be saved
        file_path = os.path.join(output_path, f'{i}.jpg')
        # Save the image as a JPEG file
        cv2.imwrite(file_path, face_array)


# Example usage
csv_path = './dataset/csv/rst_csv/pixels.csv'
images_path = './dataset/face_images'
save_images_from_csv(csv_path, images_path)
