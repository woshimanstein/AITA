import os
import linecache
import json

import torch
from torch.utils.data import Dataset
import pandas as pd

class BinaryClassificationDataset(Dataset):
    def __init__(self, data='train', balanced=True):
        if balanced:
            self.file_name = os.path.join('data', 'aita', f'{data}_balanced_aita_clean.csv')
        else:
            self.file_name = os.path.join('data', 'aita', f'{data}_aita_clean.csv')
        self.data_df = pd.read_csv(self.file_name)
        

    def __len__(self):
        return self.data_df.shape[0]

    def __getitem__(self, idx):
        return {k: v for k, v in zip(['title', 'body', 'isasshole'], [self.data_df.loc[idx].values.tolist()[2], self.data_df.loc[idx].values.tolist()[3], self.data_df.loc[idx].values.tolist()[8]])}


class SocialIQaDataset(Dataset):
    def __init__(self, data='train'):
        self.file_name = os.path.join('data', 'socialiqa-train-dev', f'{data}.jsonl')
        data_file = open(self.file_name)
        self.length = len(data_file.readlines())
        data_file.close()

        self.labels = open(os.path.join('data', 'socialiqa-train-dev', f'{data}-labels.lst')).readlines()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        line = linecache.getline(self.file_name, idx + 1)
        data_dict = json.loads(line)
        data_dict['label'] = int(self.labels[idx])
        return data_dict