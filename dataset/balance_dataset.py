import os
import pandas as pd
import numpy as np
import argparse

np.random.seed(42)

os.chdir('../')

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    '--data_file_name',
    type=str,
    default = ''
)
args = arg_parser.parse_args()

DATA_FILE_NAME = args.data_file_name

total_data = pd.read_csv(os.path.join('data', 'aita', DATA_FILE_NAME))

target_size = pd.read_csv(os.path.join('data', 'aita', 'aita_custom_agr_0.9_comment_20.csv')).shape[0]
total_data = total_data.sample(n=target_size, replace=False).reset_index()

is_the_asshole_ids = []
not_the_asshole_ids = []

for i in range(total_data.shape[0]):
    if total_data.loc[i].values.tolist()[-1] == 1:
        is_the_asshole_ids.append(i)
    else:
        not_the_asshole_ids.append(i)

selected_not_the_asshole_ids = np.random.choice(not_the_asshole_ids, len(is_the_asshole_ids))

ids_to_be_dropped = set(range(total_data.shape[0])).difference(set(is_the_asshole_ids).union(set(selected_not_the_asshole_ids)))
balanced_data = total_data.drop(ids_to_be_dropped)

balanced_data.to_csv(os.path.join('data', 'aita', f'balanced_{DATA_FILE_NAME}'), header=True, index=False)