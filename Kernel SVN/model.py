t# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#dataset
dataset = pd.read_csv('Social_Network_ads.csv')

X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, -1].values

# spliting training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.25, random_state = 0)

# Scaling Data
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

from sklearn.svm import SVC
clf = SVC(kernel = 'rbf', random_state = 0)

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix
acc_score = accuracy_score(y_test,y_pred)
cm = confusion_matrix(y_test,y_pred)

from matplotlib.colors import ListedColormap
X_set,y_set = X_test,y_test
X1,X2 = np.meshgrid(np.arange(start = X_set[:,0].min() -1 , stop= X_set[:, 0].max()+1, step = 0.01 ),
                    np.arange(start = X_set[:,1].min() -1 , stop= X_set[:, 1].max()+1, step = 0.01))
plt.contourf(X1,X2, clf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.5, cmap = ListedColormap( ('red','green') ))

plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(), X2.max())

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j,0], X_set[y_set == j,1],c = ListedColormap(('red','green'))(i),label= j )

plt.title('SVN')
plt.legend()
plt.show()


