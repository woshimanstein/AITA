import os
import pandas as pd

os.chdir('../')

data_df = pd.read_csv(os.path.join('data', 'social_chem', 'social-chem-101.v1.0.tsv'), sep='\t')
data_df_not_null = data_df.notnull()

train_data = []
for i in range(data_df.shape[0]):
    if data_df.loc[i].values.tolist()[2] == 'train' and data_df_not_null.loc[i].values.tolist()[9] and data_df_not_null.loc[i].values.tolist()[11] and data_df.loc[i].values.tolist()[11] != 0:
        train_data.append(data_df.loc[i].values.tolist())

dev_data = []
for i in range(data_df.shape[0]):
    if data_df.loc[i].values.tolist()[2] == 'dev' and data_df_not_null.loc[i].values.tolist()[9] and data_df_not_null.loc[i].values.tolist()[11] and data_df.loc[i].values.tolist()[11] != 0:
        dev_data.append(data_df.loc[i].values.tolist())

test_data = []
for i in range(data_df.shape[0]):
    if data_df.loc[i].values.tolist()[2] == 'test' and data_df_not_null.loc[i].values.tolist()[9] and data_df_not_null.loc[i].values.tolist()[11] and data_df.loc[i].values.tolist()[11] != 0:
        test_data.append(data_df.loc[i].values.tolist())

train_df = pd.DataFrame(train_data, columns=data_df.columns)
dev_df = pd.DataFrame(dev_data, columns=data_df.columns)
test_df = pd.DataFrame(test_data, columns=data_df.columns)
mini_df = pd.DataFrame(train_data[0:100], columns=data_df.columns)

train_df.to_csv(os.path.join('data', 'social_chem', f'train_social_chem.csv'), header=True, index=False)
dev_df.to_csv(os.path.join('data', 'social_chem', f'dev_social_chem.csv'), header=True, index=False)
test_df.to_csv(os.path.join('data', 'social_chem', f'test_social_chem.csv'), header=True, index=False)
mini_df.to_csv(os.path.join('data', 'social_chem', f'mini_social_chem.csv'), header=True, index=False)