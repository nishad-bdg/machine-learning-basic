import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import pandas as pd

# dataset
dataset = pd.read_csv("Salary_data.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 1/3, random_state = 0)

# linear model 
regressor = LinearRegression()
regressor = regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

# visualization of data
plt.scatter(X_train,y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title("Experience vs Salary")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()