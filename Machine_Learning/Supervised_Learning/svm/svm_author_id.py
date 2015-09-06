#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 2 (SVM) mini-project

    use an SVM to identify emails from the Enron corpus by their authors
    
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

#########################################################

from sklearn.svm import SVC

features_train = features_train[:len(features_train)]
labels_train = labels_train[:len(labels_train)]

#clf = SVC(C=1, kernel='linear')
clf = SVC(C=10000, kernel='rbf')

t0 = time()
clf.fit(features_train, labels_train)
print "Training Time ",round(time()-t0, 3),"s"
t1 = time()
prediction = clf.predict(features_test)
print "Predicting Time ",round(time()-t1, 3 ),"s"
print "Accuracy ", clf.score(features_test,labels_test)

print(prediction[10])
print(prediction[26])
print(prediction[50])

print(list(prediction).count(1))

