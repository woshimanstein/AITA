import os
import pandas as pd
from nltk.metrics.distance import edit_distance

os.chdir('../')

aita_custom_df = pd.read_csv(os.path.join('data', 'aita', 'aita_custom.csv'))
titles_custom = aita_custom_df['title'].tolist()

aita_df = pd.read_csv(os.path.join('data', 'aita', 'aita_clean.csv'))
titles = aita_df['title'].tolist()

for title0 in titles_custom:
    min_distance = 10000
    min_distance_title = ''
    for title1 in titles:
        distance = edit_distance(title0.lower(), title1.lower())
        if distance < min_distance:
            min_distance = distance
            min_distance_title = title1
    print(title0, title1)
