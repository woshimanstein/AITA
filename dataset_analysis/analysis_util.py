from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer, util

def find_similar(query, key):
    model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
    key = model.encode(key)
    query = model.encode(query)
    # cos_sim[i][j] is cos_sim(query[i], key[j])
    max_sim_ids = []
    for i in tqdm(range(query.shape[0])):
        cos_sim = util.cos_sim(query[i], key)
        max_sim_ids.append(torch.argmax(cos_sim))
    
    return max_sim_ids

def str_minus(x, y):
    result = ''
    ids = []
    if y in x:
        for i in range(0, len(x) - len(y) + 1):
            if x[i:i + len(y)] == y:
                ids.append(i)
    i = 0
    while i < len(x):
        if i not in ids:
            result += x[i]
            i += 1
        else:
            i += len(y)

    return result