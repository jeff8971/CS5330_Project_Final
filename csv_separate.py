# separate the emotion data and the pixels data from the train.csv file
# and save them as emotion.csv and pixels.csv respectively

import pandas as pd

train_csv_path = 'dataset/csv/src_csv/train.csv'
# read data
df = pd.read_csv(train_csv_path)
# get the emotion data
df_y = df[['emotion']]
# get the pixels data
df_x = df[['pixels']]
# separate emotion data into emotion.csv
df_y.to_csv('dataset/csv/rst_csv/emotion.csv', index=False, header=False)
# separate pixel data into pixels.csv
df_x.to_csv('dataset/csv/rst_csv/pixels.csv', index=False, header=False)
