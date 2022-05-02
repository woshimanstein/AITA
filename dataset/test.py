import os
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize

os.chdir('../')

DATA_FILE_NAME = 'train.jsonl'

total_data = pd.read_json(os.path.join('data', 'socialiqa-train-dev', DATA_FILE_NAME), lines=True)

length_list = []
for i in range(total_data.shape[0]):
    length = len(word_tokenize(total_data.iloc[i]['context']))
    length_list.append(length)

length_array = np.array(length_list)
print(length_array.mean())
print(np.median(length_array))    