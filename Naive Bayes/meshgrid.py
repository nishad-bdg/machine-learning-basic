# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-4,4,9)
y = np.linspace(-5,5,11)


xx,yy = np.meshgrid(x,y)

random_data = np.random.random((11,9))

plt.contourf(xx,yy,random_data, cmap= 'jet')
plt.colorbar()
plt.show()