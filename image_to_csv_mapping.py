import os
import pandas as pd
import shutil


def separate_images(source_dir, dest_dir1, dest_dir2):
    # Create destination directories if they do not exist
    os.makedirs(dest_dir1, exist_ok=True)
    os.makedirs(dest_dir2, exist_ok=True)

    # List all files in the source directory
    filenames = os.listdir(source_dir)

    # Iterate through each file
    for filename in filenames:
        # Extract the base index from the filename assuming the format "number.jpg"
        index = int(os.path.splitext(filename)[0])

        # Check the index range and move the file to the appropriate directory
        if 0 <= index <= 23999:
            shutil.move(os.path.join(source_dir, filename),
                        os.path.join(dest_dir1, filename))
        elif 24000 <= index <= 28708:
            shutil.move(os.path.join(source_dir, filename),
                        os.path.join(dest_dir2, filename))


def image_emotion_mapping(path):
    # Read the emotion data from CSV file
    df_emotion = pd.read_csv('./dataset/csv/rst_csv/emotion.csv', header=None)
    # List all files in the specified folder
    files_dir = os.listdir(path)
    # Lists to store image filenames and corresponding emotions
    path_list = []
    emotion_list = []

    # Iterate over all files in the folder
    for file_dir in files_dir:
        # Check if the file is an image and process it
        if os.path.splitext(file_dir)[1] == ".jpg":
            path_list.append(file_dir)
            index = int(os.path.splitext(file_dir)[0])
            emotion_list.append(df_emotion.iat[index, 0])

    # Combine lists into a DataFrame and save to CSV
    df = pd.DataFrame({'path': path_list, 'emotion': emotion_list})
    df.to_csv(os.path.join(path, 'image_emotion.csv'), index=False, header=False)


def main():
    # Paths for the source and destination folders
    source_directory = './dataset/face_images'
    train_set_path = './dataset/img/train_set'
    test_set_path = './dataset/img/test_set'
    # Separate images into two folders
    separate_images(source_directory, train_set_path, test_set_path)

    image_emotion_mapping(train_set_path)
    image_emotion_mapping(test_set_path)


if __name__ == "__main__":
    main()
