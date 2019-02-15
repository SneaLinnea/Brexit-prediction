# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 10:04:43 2019

@author: svens
"""

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
from sklearn.linear_model import Perceptron

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer
def read_data(filename):
    data = pd.read_csv(filename, sep="\t")
    
    labels = data['label'].astype('str').values
    removeRows = []
    
    for i in range(len(labels)):
        labels[i] = labels[i].replace('/','')
        
        if(labels[i].count('1') > len(labels[i])/2):
            labels[i] = 1
        elif(labels[i].count('0') > len(labels[i])/2):
            labels[i] = 0
        else: 
            removeRows.append(i)

    data = data.replace(data['label'].values, labels)
    
    #drop ambigous answes
    for i in range(len(removeRows)):
        data = data.drop([removeRows[i]])
    
    return data
df = read_data("a2_train_final.tsv")


Xall = df['comments']
Yall = df['label'].astype(int)  



#count_vec = CountVectorizer(min_df=1, tokenizer=nltk.word_tokenize)
#X_train_counts = count_vec.fit_transform(Xall)
#print(X_train_counts.shape)

#tfidf_transformer = TfidfTransformer()
#train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
"""
pipeline = make_pipeline(
        CountVectorizer(),
        LogisticRegression()
        )
"""
acc = 0

Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xall, Yall, train_size=0.9)


#clf = MultinomialNB().fit(Xtrain, Ytrain)
#print(cross_validate(pipeline, Xtrain, Ytrain))


parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 
                'tfidf__use_idf': (True, False),'perc__alpha': (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3),
												   'perc__n_iter': (5, 10, 15, 20, 50),
									  'perc__penalty':('l2','l1',None,'elasticnet')}






text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,2))),
                     ('tfidf', TfidfTransformer(use_idf=True)),
                     ('perc', Perceptron()),
 ])

print('hej')



   # text_clf = text_clf.fit(Xtrain, Ytrain)
gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(Xtrain, Ytrain)
predicted_svm = text_clf.predict(Xtest)
print(np.mean(predicted_svm == Ytest))


    
#print(gs_clf.predict(["Brexit is a shit sandwich"]))
#print(gs_clf.best_score_)
#print(gs_clf.best_params_)
#Yguess = clf.predict(Xtest)
#print(sklearn.metrics.accuracy_score(Ytest, Yguess))

#pipeline.fit(Xtrain, Ytrain)
#Yguess = pipeline.predict(Xtest)
#accuracy = train_model(linear_model.LogisticRegression(), xtrain_counts, Ytrain, xvalid_count)
#print "LR, Count Vectors: ", accuracy

#print(accuracy_score(Ytest, Yguess))



