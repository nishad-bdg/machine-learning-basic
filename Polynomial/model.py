import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# dataset
dataset = pd.read_csv("Position_Salaries.csv")

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[: , -1].values


# Fiiting a linear regression to the dataset
liner_reg = LinearRegression()
liner_reg = liner_reg.fit(X,y)


# Fitting a polynomial regression to the dataset
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

#poly_reg = poly_reg.fit(X_poly,y)

liner_reg2 = LinearRegression()
liner_reg2 = liner_reg2.fit(X_poly,y)



y_pred = liner_reg.predict(6.5)

y_pred_poly = liner_reg2.predict(poly_reg.fit_transform(6.5))

print(y_pred)
print(y_pred_poly)

plt.scatter(X,y, color='red')
plt.plot(X,liner_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()


plt.scatter(X,y, color = 'red')
plt.plot(X, liner_reg2.predict(poly_reg.fit_transform(X)),color='blue')
plt.title('Truth or Bluff')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()




