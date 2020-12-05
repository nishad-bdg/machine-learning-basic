# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# dataset
dataset = pd.read_csv('Social_Network_Ads.csv')

X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train,X_test, y_train,y_test = train_test_split(X,y, test_size  = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

from sklearn.naive_bayes import GaussianNB
clf_nb = GaussianNB()
clf_nb.fit(X_train,y_train)
nb_pred = clf_nb.predict(X_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test,nb_pred)


import keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(2, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(2, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(X_train,y_train,epochs = 1000)
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

acc_neural = accuracy_score(y_test, y_pred)


