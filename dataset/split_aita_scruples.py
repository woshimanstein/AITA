import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

os.chdir('../')

DATA_FILE_NAME = 'scruples-anecdotes.jsonl'
TRAIN_DATA_FILE_NAME = os.path.join('data', 'anecdotes', 'train.scruples-anecdotes.jsonl')
DEV_DATA_FILE_NAME = os.path.join('data', 'anecdotes', 'dev.scruples-anecdotes.jsonl')
TEST_DATA_FILE_NAME = os.path.join('data', 'anecdotes', 'test.scruples-anecdotes.jsonl')

train_df = pd.read_json(TRAIN_DATA_FILE_NAME, lines=True)
dev_df = pd.read_json(DEV_DATA_FILE_NAME, lines=True)
test_df = pd.read_json(TEST_DATA_FILE_NAME, lines=True)

total_df = pd.concat([train_df, dev_df, test_df]).reset_index()

def drop_low_agreemnet(total_df):
    ids_to_be_dropped = []
    for i in range(total_df.shape[0]):
        binarized_count = total_df.iloc[i].tolist()[-2]
        if sum(binarized_count.values()) < 10 or max(binarized_count['RIGHT'] / sum(binarized_count.values()), binarized_count['WRONG'] / sum(binarized_count.values())) < 0.9:
            ids_to_be_dropped.append(i)
    return total_df.drop(ids_to_be_dropped).reset_index()


def split_and_write(total_df, balanced: bool):
    if balanced:
        is_the_asshole_ids = []
        not_the_asshole_ids = []
        
        for i in range(total_df.shape[0]):
            if total_df.iloc[i].tolist()[-1] == 'WRONG':
                is_the_asshole_ids.append(i)
            elif total_df.iloc[i].tolist()[-1] == 'RIGHT':
                not_the_asshole_ids.append(i)
        
        selected_not_the_asshole_ids = np.random.choice(not_the_asshole_ids, len(is_the_asshole_ids))
        ids_to_be_dropped = set(range(total_df.shape[0])).difference(set(is_the_asshole_ids).union(set(selected_not_the_asshole_ids)))
        total_df = total_df.drop(ids_to_be_dropped)
    
    train_dev_data, test_data = train_test_split(total_df, test_size=0.1)
    train_data, dev_data = train_test_split(train_dev_data, test_size=0.2)

    train_data.to_json(os.path.join('data', 'aita', f'train.{DATA_FILE_NAME}'), orient='records', lines=True)
    dev_data.to_json(os.path.join('data', 'aita', f'dev.{DATA_FILE_NAME}'), orient='records', lines=True)
    test_data.to_json(os.path.join('data', 'aita', f'test.{DATA_FILE_NAME}'), orient='records', lines=True)

filtered_df = drop_low_agreemnet(total_df)
split_and_write(filtered_df, balanced=True)