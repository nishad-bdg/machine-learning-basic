import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# dataset 
dataset = pd.read_csv('Salary_Data.csv')

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[: , 1].values

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 1/3, random_state = 0)

regressor = LinearRegression()
regressor = regressor.fit(X_train,Y_train)

y_pred = regressor.predict(X_test)

#visualizing the training data
plt.scatter(X_train,Y_train, color='red')
plt.plot(X_train,regressor.predict(X_train), color = 'blue')
plt.title("Salary vs Experience(Training Set)")
plt.xlabel('Years Of Experience')
plt.ylabel("Salary")
plt.show()


#visualizing the test set data
plt.scatter(X_test,Y_test, color='red')
plt.plot(X_train,regressor.predict(X_train), color = 'blue')
plt.title("Salary vs Experience(Test Set)")
plt.xlabel('Years Of Experience')
plt.ylabel("Salary")
plt.show()





