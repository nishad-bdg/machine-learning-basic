import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR



# dataset 
dataset = pd.read_csv('Position_Salaries.csv')

x = dataset.iloc[:, 1:2]
y = dataset.iloc[:, -1]

regressor = SVR(kernel = 'rbf')
regressor = regressor.fit(x,y)

y_pred = regressor.predict(6.5)


plt.scatter(x,y, color = 'red')
plt.plot(x, regressor.predict(x), color = 'blue')
plt.show()

