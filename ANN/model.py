# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

#dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler

labelencoder_1 = LabelEncoder()
labelencoder_2 = LabelEncoder()

X[:,1] = labelencoder_1.fit_transform(X[:,1])
X[:,2] = labelencoder_1.fit_transform(X[:,2])

onehotencoder = OneHotEncoder(categorical_features= [1])
X = onehotencoder.fit_transform(X).toarray()

from sklearn.model_selection import train_test_split
X_train,X_test, y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense,Flatten

classifier = Sequential()

classifier.add(Dense(11, activation ='relu'))
classifier.add(Dense(11, activation ='relu'))
classifier.add(Dense(1, activation ='sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics = ['accuracy'])

classifier.fit(X_train,y_train, epochs = 100)

y_pred = classifier.predict(X_test) 
y_pred = (y_pred > 0.5)
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test,y_pred)

