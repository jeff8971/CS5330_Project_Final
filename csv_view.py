import cv2
import numpy as np
import os

# Set the path for saving the images
path = './dataset/face_images'
os.makedirs(path, exist_ok=True)  # Ensure the directory exists

# Load pixel data from a CSV file
data = np.loadtxt('./dataset/csv/rst_csv/pixels.csv')

# Loop through each row of the data
for i in range(data.shape[0]):
    # Reshape the flat array into a 48x48 image
    face_array = data[i].reshape((48, 48))
    # Construct the file path
    file_path = os.path.join(path, f'{i}.jpg')
    # Save the image
    cv2.imwrite(file_path, face_array)

