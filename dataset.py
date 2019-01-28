import json
import numpy as np
from spacy.lang.en import English
import datetime, pickle, os, codecs, re, string
from tqdm import tqdm
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk import tokenize
from nltk.stem import SnowballStemmer

nltk.download('punkt')
nltk.download('stopwords')


def clean_str(string):
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    string = string.replace('_', '')

    return string.strip().lower()

"""
Tratamento nos textos, como remoção de stopwords
"""

#STOP_WORDS = ['the', 'a', 'an']
STOP_WORDS = stopwords.words('english')
nlp = English()
nlp.add_pipe(nlp.create_pipe('sentencizer'))


def normalize(text):
    text = text.lower().strip()
    doc = nlp(text)
    filtered_sentences = []
    for sentence in tqdm(doc.sents):
        filtered_tokens = list()
        for i, w in enumerate(sentence):
            s = w.string.strip()
            if len(s) == 0 or s in string.punctuation and i < len(doc) - 1:
                continue
            if s not in STOP_WORDS:
                s = s.replace(',', '.')
                filtered_tokens.append(s)
        filtered_sentences.append(' '.join(filtered_tokens))
    return filtered_sentences


def chunk_to_arrays(chunk):
	x = chunk['text_tokens'].values
	y = chunk['stars'].values
	return x, y


def to_one_hot(labels, dim=5):
	results = np.zeros((len(labels), dim))
	for i, label in enumerate(labels):
		results[i][label - 1] = 1
	return results

def load_data_yelp(path, train_ratio=1, size=1569264):
    print('loading Yelp reviews in {0}...'.format(path))    
    
    df = pd.read_json(path, lines=True)
    
    text_tokens = []
    for row in tqdm(df['text']):    
        text_tokens.append(normalize(row))  
    
    df['text_tokens'] = text_tokens
    
    del text_tokens
    ### 
    
    dim = 5
    
    train_size = round(size * train_ratio)
    test_size = size - train_size;

    # training + validation set
    train_x = np.empty((0,))
    train_y = np.empty((0,))

    train_set = df[0:train_size].copy()
    train_set['len'] = train_set['text_tokens'].apply(lambda x: len(x))
    # train_set.sort_values('len', inplace=True, ascending=True)
    train_x, train_y = chunk_to_arrays(train_set)
    train_y = to_one_hot(train_y, dim=dim)

    test_set = df[train_size:]
    test_x, test_y = chunk_to_arrays(test_set)
    test_y = to_one_hot(test_y)
    print('finished loading Yelp reviews')

    return (train_x, train_y), (test_x, test_y)