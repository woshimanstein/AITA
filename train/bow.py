import os
import json
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import bow_utils

os.chdir('../')

'''
hyperparameter
'''
TF_IDF = False
DO_PCA = True
TOLERANCE = 1e-4
PENALTY = 'l2'
C = 0.001
MAX_ITER = 100000

'''
data setup
'''
# load data
train_df = pd.read_csv(os.path.join('data', 'aita', 'train_balanced_aita_clean.csv'))
dev_df = pd.read_csv(os.path.join('data', 'aita', 'dev_balanced_aita_clean.csv'))

# construct vocab
comments = train_df['body'].to_list()
if os.path.exists(os.path.join('train', 'word2dix.json')):
    word2idx = json.load(open(os.path.join('train', 'word2dix.json')))
else:
    vocab = bow_utils.generate_vocab(comments)
    vocab.append('[UNK]')
    word2idx = {k: v for v, k in enumerate(vocab)}
    with open(os.path.join('train', 'word2dix.json'), 'w') as f:
        f.write(json.dumps(word2idx))

# construct bow vectors
# if os.path.exists(os.path.join('train', 'train_bow_vectors.npy')):
#     train_bow_vectors_file = open(os.path.join('train', 'train_bow_vectors.npy'), 'rb')
#     train_bow_vectors = np.load(train_bow_vectors_file)
#     train_bow_vectors_file.close()
# else:
#     train_bow_vectors = np.zeros((len(comments), len(word2idx)))
#     for i in tqdm(range(len(comments))):
#         words = bow_utils.word_extraction(comments[i])
#         bow_vector = np.zeros(len(word2idx))
#         for w in words:
#             if w in word2idx.keys():
#                 bow_vector[word2idx[w]] += 1
#             else:
#                 bow_vector[word2idx['[UNK]']] += 1
#         train_bow_vectors[i] = bow_vector
#     with open(os.path.join('train', 'train_bow_vectors.npy'), 'wb') as f:
#         np.save(f, train_bow_vectors)

# if os.path.exists(os.path.join('train', 'dev_bow_vectors.npy')):
#     dev_bow_vectors_file = open(os.path.join('train', 'dev_bow_vectors.npy'), 'rb')
#     dev_bow_vectors = np.load(dev_bow_vectors_file)
#     dev_bow_vectors_file.close()
# else:
#     dev_comments = dev_df['body'].to_list()
#     dev_bow_vectors = np.zeros((len(dev_comments), len(word2idx)))
#     for i in tqdm(range(len(dev_comments))):
#         words = bow_utils.word_extraction(dev_comments[i])
#         bow_vector = np.zeros(len(word2idx))
#         for w in words:
#             if w in word2idx.keys():
#                 bow_vector[word2idx[w]] += 1
#             else:
#                 bow_vector[word2idx['[UNK]']] += 1
#         dev_bow_vectors[i] = bow_vector
#     with open(os.path.join('train', 'dev_bow_vectors.npy'), 'wb') as f:
#         np.save(f, dev_bow_vectors)

# tf-idf
if TF_IDF:
    print('Applying TF-IDF')
    tfidf = TfidfTransformer()
    tfidf.fit(train_bow_vectors)
    train_bow_vectors = tfidf.transform(train_bow_vectors)
    dev_bow_vectors = tfidf.transform(dev_bow_vectors)


if DO_PCA:
    print('Applying PCA')
    if os.path.exists(os.path.join('train', 'train_bow_vectors_pca_100.npy')) and os.path.exists(os.path.join('train', 'dev_bow_vectors_pca_100.npy')):
        train_bow_vectors_pca_100_file = open(os.path.join('train', 'train_bow_vectors_pca_100.npy'), 'rb')
        dev_bow_vectors_pca_100_file = open(os.path.join('train', 'dev_bow_vectors_pca_100.npy'), 'rb')
        train_bow_vectors = np.load(train_bow_vectors_pca_100_file)
        dev_bow_vectors = np.load(dev_bow_vectors_pca_100_file)
        train_bow_vectors_pca_100_file.close()
        dev_bow_vectors_pca_100_file.close()
    else:
        pca = IncrementalPCA(n_components=100, batch_size=1000)
        pca.fit(train_bow_vectors)
        train_bow_vectors = pca.transform(train_bow_vectors)
        dev_bow_vectors = pca.transform(dev_bow_vectors)
        with open(os.path.join('train', 'train_bow_vectors_pca_100.npy'), 'wb') as f:
            np.save(f, train_bow_vectors)
        with open(os.path.join('train', 'dev_bow_vectors_pca_100.npy'), 'wb') as f:
            np.save(f, dev_bow_vectors)

train_labels = np.array(train_df['is_asshole'].to_list())
dev_labels = np.array(dev_df['is_asshole'].to_list())


'''
train and evaluation
'''
model = LogisticRegression(solver='saga', tol=TOLERANCE, penalty=PENALTY, C=C, max_iter=MAX_ITER, verbose=1)
print('Fitting')
print(train_bow_vectors.shape)
model.fit(train_bow_vectors, train_labels)
dev_pred = model.predict(dev_bow_vectors)
print(np.sum(dev_pred))
# dev_pred = np.random.randint(1, 2, size=4367)

accuracy = accuracy_score(dev_labels, dev_pred)
precision = precision_score(dev_labels, dev_pred)
recall = recall_score(dev_labels, dev_pred)
f1 = f1_score(dev_labels, dev_pred)

print(f'TF-IDF: {TF_IDF}')
print(f'tol: {TOLERANCE}')
print(f'penalty: {PENALTY}')
print(f'C: {C:.2e}')
print(f'Accuracy: {accuracy:.3f}')
print(f'Recall: {recall:.3f}')
print(f'Precision: {precision:.3f}')
print(f'F1: {f1:.3f}')