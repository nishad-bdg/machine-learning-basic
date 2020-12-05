import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,Imputer
from sklearn.tree import DecisionTreeRegressor



# dataset 
dataset = pd.read_csv('train.csv')

independent_vars = ["Pclass","Sex","Age","Fare","Embarked"]

testing_data = pd.read_csv('test.csv')

testing_data_split= testing_data[independent_vars]
X_test = testing_data_split.iloc[:, : -1].values

#label encode for test data
labelencoder_test = LabelEncoder()
X_test[:, 1] = labelencoder_test.fit_transform(X_test[:,1])

imputer_test = Imputer(missing_values = 'NaN', strategy = 'median', axis  = 0)
X_test[:, 2:4] = imputer_test.fit_transform(X_test[:, 2:4])





X = dataset[independent_vars].iloc[:,:-1].values
y = dataset.iloc[:, 1].values

labelencoder = LabelEncoder()
X[:, 1] = labelencoder.fit_transform(X[:, 1])

imputer = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
X[:, 2:4] = imputer.fit_transform(X[:,2:4])

regressor = DecisionTreeRegressor(random_state = 0)
regressor = regressor.fit(X,y)


y_pred = regressor.predict(X_test)

plt.scatter(X[:, 0],y, color = 'red')
plt.show()





