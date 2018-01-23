#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()

### We need to chunk this array to prevent Memory Error. 
### If you experience this kind of problem you should reduce to 
### a lower number (Try 500 for example)
n = 8000

t0 = time()
classifier.fit(features_train[0:n], labels_train[0:n])
print "tempo de treinamento:", round(time()-t0, 3), "s"

t0 = time()
pred = classifier.predict(features_test[0:n])
print "tempo de treinamento:", round(time()-t0, 3), "s"

from sklearn.metrics import accuracy_score
print accuracy_score(labels_test[0:n], pred)
#########################################################


