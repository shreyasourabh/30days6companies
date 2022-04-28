# -*- coding: utf-8 -*-
"""
Created on Oct 20 23:30:49 2021

@author: Shreya
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
dataset=pd.read_csv('creditcard.csv')

#counting the no. of 1's and 0's
count = dataset["Class"].value_counts()

#creating independent and dependent variables
X = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1:]

#splitting the data into train set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33, random_state =100)

from sklearn.metrics import roc_auc_score
roc_values = []
for feature in X_train.columns:
    clf = LogisticRegression(random_state = 0)
    clf.fit(X_train[feature].fillna(0).to_frame(), y_train)
    y_scored = clf.predict_proba(X_test[feature].fillna(0).to_frame())
    roc_values.append(roc_auc_score(y_test, y_scored[:, 1]))

roc_values = pd.Series(roc_values)
roc_values.index = X_train.columns
roc_values.sort_values(ascending=False)
roc_values.sort_values(ascending=False).plot.bar(figsize=(20, 8))
plt.title('Univariate ROC-AUC')

X_train = X_train.drop(['V22','Amount','V15','V26','V25'],axis='columns')
X_test = X_test.drop(['V22','Amount','V15','V26','V25'],axis='columns')

#feature scaling
from sklearn.preprocessing import RobustScaler
scaler= RobustScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train = pd.DataFrame(X_train_scaled,columns=X_train.columns)
X_test = pd.DataFrame(X_test_scaled,columns=X_test.columns)

#sampling
from imblearn.over_sampling import SMOTE
sm=SMOTE(random_state=100)
X_sm,y_sm=sm.fit_sample(X_train,y_train)

#training model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_sm,y_sm)

#training model
y_pred = classifier.predict(X_test).reshape(-1,1)
y_pred1 = classifier.predict(X_train)

#predicting values
from sklearn.metrics import accuracy_score
print("Accuracy of train set : " + str(accuracy_score(y_train,y_pred1)))
print('Accuracy of test set : '+str(accuracy_score(y_test,y_pred)))
print('Roc-Auc score:' +str(metrics.roc_auc_score(y_test,y_lprob)))

#roc-auc curve
from sklearn import metrics
y_lprob=classifier.predict_proba(X_test_scaled)[:,1]
auc=metrics.roc_auc_score(y_test,y_lprob)
fpr,tpr,thresholds=metrics.roc_curve(y_test,y_lprob)
plt.plot(fpr,tpr,'b', label='AUC = %0.2f'% auc)
plt.plot([0,1],[0,1],'r-.')
plt.xlim([-0.2,1.2])
plt.ylim([-0.2,1.2])
plt.title('Receiver Operating Characteristic\nLogistic Regression')
plt.legend(loc='lower right')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show() 
