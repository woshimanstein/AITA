import os
import pickle
import pandas as pd

from analysis_util import find_similar, str_minus

os.chdir('../')

aita_df = pd.read_csv(os.path.join('data', 'aita', 'aita_clean.csv'))
social_chem_df = pd.read_csv(os.path.join('data', 'social_chem', 'social-chem-101.v1.0.tsv'), sep='\t')

aita_titles = aita_df['title'].tolist()
social_chem_df = social_chem_df.loc[social_chem_df['area'] == 'amitheasshole']
social_chem_situations = social_chem_df['situation'].tolist()
print(len(aita_titles), len(social_chem_situations))

for i in range(len(aita_titles)):
    if 'aita for' in str.lower(aita_titles[i]):
        aita_titles[i] = str_minus(str.lower(aita_titles[i]), 'aita for')
    elif 'wibta if' in str.lower(aita_titles[i]):
        aita_titles[i] = str_minus(str.lower(aita_titles[i]), 'wibta if')
    elif 'aita if' in str.lower(aita_titles[i]):
        aita_titles[i] = str_minus(str.lower(aita_titles[i]), 'aita if')
    elif 'wibta for' in str.lower(aita_titles[i]):
        aita_titles[i] = str_minus(str.lower(aita_titles[i]), 'wibta for')
    else:
        aita_titles[i] = str.lower(aita_titles[i])

if not os.path.exists(os.path.join('dataset_analysis', 'result')):
    with open(os.path.join('dataset_analysis', 'result'), 'wb') as f:
        result = find_similar(aita_titles, social_chem_situations)
        pickle.dump(result, f)
else:
    with open(os.path.join('dataset_analysis', 'result'), 'rb') as f:
        result = pickle.load(f)

print(result)
