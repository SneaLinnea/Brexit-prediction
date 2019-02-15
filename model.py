# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import sklearn
import nltk
import re
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer
def read_data(filename):
    data = pd.read_csv(filename, sep="\t")
    
    #labels = data['label'].astype('str').values
    labels = data.iloc[:,0].astype('str').values
    removeRows = []
    
    for i in range(len(labels)):
        labels[i] = labels[i].replace('/','')
        
        if(labels[i].count('1') > len(labels[i])/2):
            labels[i] = 1
        elif(labels[i].count('0') > len(labels[i])/2):
            labels[i] = 0
        else: 
            removeRows.append(i)

    data = data.replace(data.iloc[:,0].values, labels)
    
    #drop ambigous answes
    for i in range(len(removeRows)):
        data = data.drop([removeRows[i]])
    
    return data

df = read_data("a2_train_final.tsv")

Xall = df.iloc[:,1]
Yall = df.iloc[:,0].astype(int)  
print(df.head(2))
"""
acc = 0
nbrOfIterations = 10
for i in range (nbrOfIterations):
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xall, Yall, train_size=0.9)
    
    
    parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 
                    'tfidf__use_idf': (True, False),'clf-svm__alpha': (1e-2, 1e-3),
                    'clf-svm__loss': ('hinge', 'log', 'squared_hinge', 'perceptron')}
    
    text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,2))),
                         ('tfidf', TfidfTransformer(use_idf=True, norm='l2')),
                         ('clf-svm', SGDClassifier(loss='hinge', 
                                               alpha=1e-3, n_iter=5, random_state=42)),
     ])
    

    gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
    gs_clf = gs_clf.fit(Xtrain, Ytrain)
    predicted_svm = gs_clf.predict(Xtest)
    acc += np.mean(predicted_svm == Ytest)
    print(gs_clf.best_score_)
    print(gs_clf.best_params_)
    
print(acc/nbrOfIterations)
    

"""


