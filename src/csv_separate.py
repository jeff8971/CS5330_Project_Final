#!/usr/bin/env python3
"""
Filename: csv_separate.py
Author: Yuan Zhao
Email: zhao.yuan2@northeastern.edu
Description: This file is used to separate the emotion data and the pixels data from the train.csv file
             and save them as emotion.csv and pixels.csv respectively.
Date: 2024-04-09
"""

import pandas as pd

def separate_data(csv_path, output_dir):
    """
    Separates emotion and pixels data from a CSV file and saves them into separate files.

    Args:
    csv_path (str): Path to the source CSV file containing the training data.
    output_dir (str): Directory where the separated CSV files will be saved.
    """
    # Read data
    df = pd.read_csv(csv_path)

    # Get the emotion data
    df_y = df[['emotion']]
    # Get the pixels data
    df_x = df[['pixels']]

    # Save emotion data into emotion.csv
    emotion_output_path = f'{output_dir}/emotion.csv'
    df_y.to_csv(emotion_output_path, index=False, header=False)

    # Save pixel data into pixels.csv
    pixels_output_path = f'{output_dir}/pixels.csv'
    df_x.to_csv(pixels_output_path, index=False, header=False)

# Example usage
train_csv_path = 'dataset/csv/src_csv/train.csv'
output_directory = 'dataset/csv/rst_csv'
separate_data(train_csv_path, output_directory)
