import numpy as np
import pandas as pd
from PIL import Image
import os

# Paths for the training and validation datasets
train_path = 'dataset/img/train/'
valid_path = 'dataset/img/test/'
data_path = 'dataset/csv/src_csv/fer2013.csv'

def make_dir():
    # Create directories for each of the 7 emotion categories in training and validation
    for emotion in range(7):
        os.makedirs(os.path.join(train_path, str(emotion)), exist_ok=True)
        os.makedirs(os.path.join(valid_path, str(emotion)), exist_ok=True)

def save_images():
    df = pd.read_csv(data_path)
    train_index, valid_index = [1] * 7, [1] * 7  # Initialize counters for image filenames

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        emotion, image_data, usage = row['emotion'], row['pixels'], row['Usage']
        image_array = np.array(list(map(int, image_data.split()))).reshape(48, 48)
        image = Image.fromarray(image_array).convert('L')  # Convert array to an 8-bit grayscale image

        # Define path based on usage and save the image
        if usage == 'Training':
            file_path = os.path.join(train_path, str(emotion), f'{train_index[emotion]}.jpg')
            train_index[emotion] += 1
        else:
            file_path = os.path.join(valid_path, str(emotion), f'{valid_index[emotion]}.jpg')
            valid_index[emotion] += 1
        image.save(file_path)

# Create directories and save images
make_dir()
save_images()
