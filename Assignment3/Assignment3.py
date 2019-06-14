import sys
import nltk
import numpy
from nltk.corpus import stopwords
import string
import pandas as pd
import numpy as np
import multiprocessing
from gensim.models import Word2Vec

## Student Name: Guohao Ma
## Student Number: 20676560

def token_with_punc(text):
    # seperate punctuations and letters by adding space before and after each punctuation
    translate_table = dict((ord(char), ' ' + char + ' ') for char in string.punctuation)
    # create a map to get tokens
    text = text[0].translate(translate_table)
    # return the tokens after splitting by whitespace
    return text.lower().split()

# remove punctuations
def punc_rem(text):
    # create a string of punctuations that need to be removed
    punc_list = ',!"#$%&()*+/:;<=>@[\\]^`{|}~\t\n'
    # remove punctuations as required
    text = [w for w in text if w not in punc_list]
    return text

# remove stopwords
def stop_word_rem(text):
    # using stopwords offered by NLTK
    text = [w for w in text if w not in stopwords.words('english')]
    return text

# return a list with stopwords
def clean_with_stopwords(text):
    text = token_with_punc(text)
    text = punc_rem(text)
    return text

# return a list without stopwords
def clean_no_stopwords(text):
    text = clean_with_stopwords(text)
    text = stop_word_rem(text)
    return text

# read the file and return a numpy array
df_pos = pd.read_csv("./pos.txt",header=None, sep="\n")
df_neg = pd.read_csv("./neg.txt",header=None, sep="\n")
frames = [df_pos,df_neg]
df = pd.concat(frames).values

token_with_stopwords = []
for doc in df:
    doc = clean_with_stopwords(doc)
    token_with_stopwords.append(doc)

# word2vec parameters
w2v_model = Word2Vec(min_count=20,
                     window=2,
                     size=300,
                     sample=6e-5,
                     alpha=0.03,
                     min_alpha=0.0007,
                     negative=20,
                     workers =2
                    )
# train a model
w2v_model.build_vocab(token_with_stopwords, progress_per=10000)
w2v_model.train(token_with_stopwords, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
w2v_model.init_sims(replace=True)



# get the most similar words to a given word
print('20 most similar words to “good” are:')
print(w2v_model.wv.most_similar(positive=["good"],topn=20))
print("-------------------------------")
print('20 most similar words to “bad” are:')
print(w2v_model.wv.most_similar(positive=["bad"],topn=20))