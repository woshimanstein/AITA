import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def word_extraction(sentence):
    words = nltk.word_tokenize(sentence)
    # words= [word for word in words if word.isalnum()]
    ignore = set(stopwords.words('english'))

    stemmer = PorterStemmer()
    cleaned_text = [stemmer.stem(w.lower()) for w in words if w not in ignore and '/' not in w]

    return cleaned_text

def generate_vocab(sentences):
    words = []

    for sentence in sentences:
        w = word_extraction(sentence)
        words.extend(w)

    words = sorted(list(set(words)))
    return words