import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# dataset
dataset = pd.read_csv('temp.csv')


x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


regressor = LinearRegression()
regressor = regressor.fit(x,y)

poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(x)


linear_reg = LinearRegression()
linear_reg = linear_reg.fit(x_poly, y)

pred = linear_reg.predict(poly_reg.fit_transform(50))


plt.scatter(x,y, color = 'red')
plt.plot(x, linear_reg.predict(poly_reg.fit_transform(x)), color = 'blue')
plt.show()

