# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
def read_data(filename):
    data = pd.read_csv(filename, sep="\t")
    
    labels = data['label'].astype('str').values
    removeRows = []
    
    for i in range(len(labels)):
        labels[i] = labels[i].replace('/','')
        if(('0' in labels[i]) and ('1' in labels[i])):
                removeRows.append(i)
        labels[i] = labels[i][0]   
    

    data = data.replace(data['label'].values, labels)

    for i in range(len(removeRows)):
        data = data.drop([removeRows[i]])

read_data("a2_train_final.tsv")