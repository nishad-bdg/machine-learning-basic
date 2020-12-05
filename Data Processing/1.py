import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,Imputer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

# dataset
dataset = pd.read_csv('Data.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:,0])

labelencoder_Y = LabelEncoder()
y = labelencoder_Y.fit_transform(y)

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = .2, random_state = 0)


# regressor = LinearSVC()

# regressor = regressor.fit(X_train,y_train)

# y_pred = regressor.predict(X_test)
# print(y_pred)

regressor = LinearRegression()
regressor = regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test[0:2])


