# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

# dataset
dataset = pd.read_csv('train.csv')
X = dataset.iloc[:, [2,4,5]].values
y = dataset.iloc[:, 1].values

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

