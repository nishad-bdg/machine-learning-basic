import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# dataset
dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values


linear_reg = LinearRegression()
linear_reg = linear_reg.fit(X,y)

poly_reg = PolynomialFeatures(degree = 5)
X_poly = poly_reg.fit_transform(X)

poly_reg = poly_reg.fit(X_poly,y)

linear_reg2 = LinearRegression()
linear_reg2 = linear_reg2.fit(X_poly,y)

y_pred2 = linear_reg2.predict(poly_reg.fit_transform(6.5))

plt.scatter(X,y, color = 'red')
plt.plot(X, linear_reg2.predict(poly_reg.fit_transform(X)))
plt.show()

