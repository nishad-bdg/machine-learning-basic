# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# dataset
dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values

from sklearn.preprocessing import StandardScaler

X_sc = StandardScaler()
y_sc = StandardScaler()

X = X_sc.fit_transform(X)
y = X_sc.fit_transform(y)