import os
from sklearn.model_selection import train_test_split
import pandas as pd

DATA_FILE_NAME = 'balanced_aita_clean.csv'

os.chdir('../')

total_data = pd.read_csv(os.path.join('data', 'aita', DATA_FILE_NAME))

blank_post_ids = []
for i in range(total_data.shape[0]):
    if type(total_data.loc[i].values.tolist()[3]) != str:
        blank_post_ids.append(i)
total_data = total_data.drop(blank_post_ids)

train_dev_data, test_data = train_test_split(total_data, test_size=0.1)
train_data, dev_data = train_test_split(train_dev_data, test_size=0.1)

train_data.to_csv(os.path.join('data', 'aita', f'train_{DATA_FILE_NAME}'), header=True, index=False)
dev_data.to_csv(os.path.join('data', 'aita', f'dev_{DATA_FILE_NAME}'), header=True, index=False)
test_data.to_csv(os.path.join('data', 'aita', f'test_{DATA_FILE_NAME}'), header=True, index=False)