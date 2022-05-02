import os
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import argparse

np.random.seed(42)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    '--data_file_name',
    type=str,
    default = ''
)
args = arg_parser.parse_args()

DATA_FILE_NAME = args.data_file_name

os.chdir('../')

total_data = pd.read_csv(os.path.join('data', 'aita', DATA_FILE_NAME))

train_dev_data, test_data = train_test_split(total_data, test_size=0.1)
train_data, dev_data = train_test_split(train_dev_data, test_size=0.1)

train_data.to_csv(os.path.join('data', 'aita', f'train_{DATA_FILE_NAME}'), header=True, index=False)
dev_data.to_csv(os.path.join('data', 'aita', f'dev_{DATA_FILE_NAME}'), header=True, index=False)
test_data.to_csv(os.path.join('data', 'aita', f'test_{DATA_FILE_NAME}'), header=True, index=False)

train_data[:16].to_csv(os.path.join('data', 'aita', f'mini_{DATA_FILE_NAME}'), header=True, index=False)

print(train_data.shape, dev_data.shape, test_data.shape)