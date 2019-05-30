import sys
import nltk
import numpy
from nltk.corpus import stopwords
import string
import pandas as pd
import numpy as np
from random import shuffle
import csv


# tokenize the sentences into words with punctuation
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
    text = [ w for w in text if w not in stopwords.words('english')]
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

# load the file
def readtxt(file):
    df = pd.read_csv(file, header=None, sep="\n")
    return df.values

def data_split(text):
    shuffle(text)
    text = pd.DataFrame(text)
    text.fillna(value="", inplace=True)
    text.dropna(axis='columns')
    segement1 = int(0.8 * len(text))
    segement2 = int(0.9 * len(text))
    train = np.array(text[:segement1])
    validation = np.array(text[segement1:segement2])
    test = np.array(text[segement2:])
    return train, validation, test


if __name__ == "__main__":
    input_path = sys.argv[1]
    text = readtxt(input_path)
    # Tokenize the input file here
    token_with_stopwords = []
    token_no_stopwords = []
    for sent in text:
        # create a list with stopwords
        sent_with_stopwords = clean_with_stopwords(sent)
        token_with_stopwords.append(sent_with_stopwords)
        # create a list without stopwords
        sent_no_stopwords = clean_no_stopwords(sent)
        token_no_stopwords.append(sent_no_stopwords)
    # Create train, val, and test sets
    train_list, val_list, test_list = data_split(token_with_stopwords)
    train_list_no_stopword, val_list_no_stopword, test_list_no_stopword = data_split(token_with_stopwords)

    # save the train, validation and tes data as csv files
    np.savetxt("train.csv", train_list, delimiter=",", fmt='%s')
    np.savetxt("val.csv", val_list, delimiter=",", fmt='%s')
    np.savetxt("test.csv", test_list, delimiter=",", fmt='%s')

    np.savetxt("train_no_stopword.csv", train_list_no_stopword,
               delimiter=",", fmt='%s')
    np.savetxt("val_no_stopword.csv", val_list_no_stopword,
               delimiter=",", fmt='%s')
    np.savetxt("test_no_stopword.csv", test_list_no_stopword,
               delimiter=",", fmt='%s')
