# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# dataset
dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, 1:2]
y = dataset.iloc[:,-1]

from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg = linear_reg.fit(X,y)

y_pred = linear_reg.predict(6.5)

from sklearn.preprocessing import PolynomialFeatures


poly_reg = PolynomialFeatures(degree = 5)
X_poly = poly_reg.fit_transform(X)

poly_reg = poly_reg.fit(X_poly,y)

linear_reg2 = LinearRegression()
linear_reg2 = linear_reg2.fit(X_poly,y)

y_pred2 = linear_reg2.predict(poly_reg.fit_transform(6.5))

plt.scatter(X,y, color = 'red')
plt.plot(X, linear_reg2.predict(poly_reg.fit_transform(X)))