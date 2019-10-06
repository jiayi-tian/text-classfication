# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 16:51:53 2019

@author: 12642
"""

import spacy

from sklearn.datasets import fetch_20newsgroups  # import packages which help us download dataset and load intp python
from sklearn.pipeline import Pipeline

import numpy as np  # numpy package is for fast numerical computation in Python
twenty_train = fetch_20newsgroups(subset='train', shuffle=True, download_if_missing=True)  
twenty_test = fetch_20newsgroups(subset='test', shuffle=True, download_if_missing=True)
#print(twenty_train.description)
print(twenty_train.data[0])
print(twenty_train.target_names)
# Extracting features from text files

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
print(f'Shape of Term Frequency Matrix: {X_train_counts.shape}')

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(f'Shape of TFIDF Matrix: {X_train_tfidf.shape}')
print(X_train_counts)
print(X_train_tfidf)
print(twenty_train.target)#??????

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)
text_nb_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
text_nb_clf = text_nb_clf.fit(twenty_train.data, twenty_train.target)
predicted = text_nb_clf.predict(twenty_test.data)
naivebayes_clf_accuracy = np.mean(predicted == twenty_test.target) * 100.
print(f'Test Accuracy is {naivebayes_clf_accuracy} %')

from sklearn.linear_model import LogisticRegression as LR
text_lr_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf',LR())])
text_lr_clf = text_lr_clf.fit(twenty_train.data, twenty_train.target)
lr_predicted = text_lr_clf.predict(twenty_test.data)
lr_clf_accuracy = np.mean(lr_predicted == twenty_test.target) * 100.
print(f'Test Accuracy is {lr_clf_accuracy}')

from sklearn.metrics import confusion_matrix
cf = confusion_matrix(y_true=twenty_test.target, y_pred=lr_predicted)

import json
print(json.dumps(cf.tolist(), indent=2))#tolist将数组或矩阵转为列表 json.dumps 对数据进行编码

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))
ax = sns.heatmap(cf, annot=True,linewidths=.5, center = 90, vmax = 200)

nlp = spacy.load('en_core_web_sm')
from spacy.lang.en.stop_words import STOP_WORDS
STOP_WORDS

f'There are {len(STOP_WORDS)} stopwords in spaCy'
STOP_WORDS.add("your_additional_stop_word_here")
f'After adding your own stop words, spaCy will use {len(STOP_WORDS)} stopwords'
doc = nlp("I am learning the most important ideas Natural Language Processing ideas using Python")
print(doc) 

for token in doc:
    print(token)
    
simplified_doc = [token for token in doc if not token.is_punct | token.is_stop]
simplified_doc

for token in simplified_doc:
    print(f'Token:{token.orth_}\tLemmatized:{token.lemma_}\tPart-of-Speech-Tag:{token.pos_}')
from spacy.lang.en import English
tokenizer = English().Defaults.create_tokenizer(nlp)
def spacy_tokenizer(document):
     return [token.orth_ for token in tokenizer(document)]
 
text_lr_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf',LR())])
text_lr_clf = text_lr_clf.fit(twenty_train.data, twenty_train.target)

def calc_print_accuracy(text_clf, test):
    predictions = text_clf.predict(test.data)
    clf_accuracy = np.mean(predictions == test.target) * 100.
    print(f'Test Accuracy is {clf_accuracy}')
    return clf_accuracy

calc_print_accuracy(text_lr_clf, twenty_test)

text_lr_clf = Pipeline([('vect', CountVectorizer(tokenizer=spacy_tokenizer, stop_words=list(STOP_WORDS))), ('tfidf', TfidfTransformer()), ('clf',LR())])
text_lr_clf = text_lr_clf.fit(twenty_train.data, twenty_train.target)
calc_print_accuracy(text_lr_clf, twenty_test)
url = 'http://www.gutenberg.org/ebooks/1661.txt.utf-8'
file_name = 'sherlock.txt'

import urllib.request
# Download the file from `url` and save it locally under `file_name`:

with urllib.request.urlopen(url) as response:
    with open(file_name, 'wb') as out_file:
        data = response.read() # a `bytes` object
        out_file.write(data)

#let's the load data to RAM 
text=text.requests.get(url) # note that I add an encoding='utf-8' parameter to preserve information
print(text[:5])
