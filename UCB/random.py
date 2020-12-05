# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')
N = 10000
d = 10
ads_selected = []
total_rewards = 0
for n in range(0,N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = dataset.values[n,ad]
    total_rewards = total_rewards + reward

plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each was selected')
plt.show()