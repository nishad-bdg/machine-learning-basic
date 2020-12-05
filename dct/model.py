import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor


# dataset
dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, 1:2]
y = dataset.iloc[:, -1]


regressor = DecisionTreeRegressor(random_state = 0)
regressor = regressor.fit(X,y)

y_pred = regressor.predict(6.5)
print(y_pred)

plt.scatter(X,y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.show()
