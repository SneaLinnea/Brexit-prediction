# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score

from sklearn.dummy import DummyClassifier
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
Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xall, Yall, train_size=0.9)
    
pipeline = make_pipeline(
    DummyClassifier()
)
print(cross_validate(pipeline, Xtrain, Ytrain))

pipeline.fit(Xtrain, Ytrain)
Yguess = pipeline.predict(Xtest)
print(accuracy_score(Ytest, Yguess))



