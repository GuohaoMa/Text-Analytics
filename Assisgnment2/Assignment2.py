import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import sys
import csv
from nltk.util import ngrams
from sklearn.naive_bayes import MultinomialNB
from scipy.stats import randint
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.metrics import classification_report

def read_file(filepath):
    '''
    read the csv files that passed in
    :param filepath:
    :return: a list, each element represents one document(line)
    '''
    with open(filepath, 'r') as f:
         reader = csv.reader(f)
         file_list = list(reader)
    return file_list

def remove_bracket(file):
    '''
    remove the bracket at the beginning and the end of each doc
    :param file: for all the lines
    :return:  "[' like'" -> " 'like' "
    '''
    for j in range(len(file)):
        file[j][0] = file[j][0].replace("['","'")
        file[j][-1] = file[j][-1].replace("']","'")
    return file


def join_doc(listfile):
    '''
    for each line, join the tokens together to get one string for each document
    for preparation of bag of words
    :param listfile: all the lines
    :return: a string for each line
    '''
    for i in range(len(listfile)):
        listfile[i]="".join(listfile[i])
    return listfile


def bag_of_words(train_file, val_file, test_file, ngrams):
    '''
    count the number of each token
    :param ngrams: unigrams or bigrams
    :return: a count matrix
    '''
    vectorizer = CountVectorizer(ngram_range=ngrams)
    train_file = vectorizer.fit_transform(train_file)
    val_file= vectorizer.transform(val_file)
    test_file = vectorizer.transform(test_file)
    feature_list = vectorizer.get_feature_names()
    return train_file, val_file, test_file, feature_list


def gridsearchMNB(X_train,X_test,y_train,y_test):
    '''
    gridsearch the best alpha for MNB based on the accuracy on validation data
    :param y_train: validation dataset
    :param y_test:  validation dataset label
    :return: the best alpha found
    '''
    tuned_parameters ={'alpha': list(range(20))}
    clf=GridSearchCV(MultinomialNB(),tuned_parameters,scoring="accuracy")
    clf.fit(X_train,y_train)
    print("Parameter tuned: alpha")
    print("Search space: alpha = ", list(range(20)))
    #print(clf.cv_results_['params'])
    print("Best parameters set found:",clf.best_params_)
    print("Optimized accuracy on validation set:",clf.score(X_test,y_test))
    print("Detailed classification report:")
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    return clf.best_params_['alpha']


def model_accuracy(X_train, X_val, X_test, y_train, y_val, y_test, ngrams):
    '''
    :param X_train: training or training_no_stopword
    :param X_val:   validation or validation_no_stopword
    :param X_test:  test or test_no_stopword
    :param y_train: training_label or training_no_stopword_label
    :param y_val:   validation_label or validation_no_stopword
    :param y_test:  test_label or test_no_stopword_label
    :param ngrams:  unigrams, bigrams or both
    :return: the classification accuracy and report on test data based on the
             MNB classifier trained on training data and tuned on validation data
    '''
    X_train, X_val, X_test, feature_list = bag_of_words(X_train, X_val, X_test, ngrams)
    best_alpha = gridsearchMNB(X_train, X_val, y_train, y_val)
    nb = MultinomialNB(alpha=best_alpha)
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    print("----------------------------------------")
    print("Model accuracy: ", metrics.accuracy_score(y_test, y_pred))
    print("Confusion matrix: ")
    print(metrics.confusion_matrix(y_test, y_pred))
    print("Detailed classification report:")
    print(classification_report(y_test, y_pred))

#  create a dictionary of text features
text_features = {'unigrams':(1,1),'bigrams':(2,2),'unigrams+bigrams':(1,2)}


if __name__ == "__main__":
    # file order: training_pos, training_neg, validation_pos, validation_neg, test_pos, test_neg

    # remove the brackets for each doc
    train_pos_list = remove_bracket(read_file(sys.argv[1]))
    train_neg_list = remove_bracket(read_file(sys.argv[2]))
    val_pos_list = remove_bracket(read_file(sys.argv[3]))
    val_neg_list = remove_bracket(read_file(sys.argv[4]))
    test_pos_list = remove_bracket(read_file(sys.argv[5]))
    test_neg_list = remove_bracket(read_file(sys.argv[6]))

    # merge the positive and negative reviews
    train_list = train_neg_list + train_pos_list
    val_list = val_neg_list + val_pos_list
    test_list = test_neg_list + test_pos_list
    # 3 labels
    train_label = ['neg'] * len(train_neg_list) + ['pos'] * len(train_pos_list)
    val_label = ['neg'] * len(val_neg_list) + ['pos'] * len(val_pos_list)
    test_label = ['neg'] * len(test_neg_list) + ['pos'] * len(test_pos_list)

    # print the final accuracy
    for i in text_features:
         print("*************************************************************")
         print("Text features: ",i)
         model_accuracy(join_doc(train_list), join_doc(val_list), join_doc(test_list),train_label,val_label,test_label,text_features[i])