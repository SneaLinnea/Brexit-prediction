# -*- coding: utf-8 -*-
import pandas as pd

from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC, SVC, LinearSVR
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

from sklearn.feature_extraction.text import CountVectorizer

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

nmbrOfTest = 50
accTest = 0
for i in range(nmbrOfTest):
    Xall = df['comments']
    Yall = df['label'].astype(int)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xall, Yall, train_size=0.9)
    #Xtrain = [x.lower() for x in Xtrain]
    #Xtest = [x.lower() for x in Xtest]

    pipeline = make_pipeline(
        #CountVectorizer(max_features=1500, min_df=5, max_df=0.7),
        CountVectorizer(),
        TfidfTransformer(),
        #DummyClassifier(),
        #LinearRegression(), # array([-300.76192553, -402.20824967, -205.53098974])
        #Ridge(), #array([0.27201949, 0.25454593, 0.26855444])
        #RandomForestRegressor() #array([0.15411908, 0.17543855, 0.15572508]),
        #GradientBoostingRegressor() #array([0.21263021, 0.21315019, 0.21473426]),
        #LogisticRegression() # array([0.75988858, 0.73452315, 0.76575572])
        #DecisionTreeClassifier()
        #KNeighborsClassifier()
        #LinearSVC()
        #LinearSVR()
        #MLPClassifier(alpha=1)
        #SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
        SGDClassifier(loss="log", max_iter=5)
)


    pipeline.fit(Xtrain, Ytrain)
    Yguess = pipeline.predict(Xtest)
    accScore = accuracy_score(Ytest, Yguess)
    print("Run", (i+1), "got score:", accScore)
    accTest += accScore

    print("Current average:", accTest/(i+1))
    print("-------------------------------------")

#print(cross_validate(pipeline, Xtrain, Ytrain))
#print(accuracy_score(Ytest, Yguess))
#print(confusion_matrix(Ytest, Yguess))

print("Final result: ", accTest/nmbrOfTest)



