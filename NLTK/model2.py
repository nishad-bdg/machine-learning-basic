# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# dataset 
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

review = re.sub('[^a-zA-z]',' ', dataset['Review'][0])
review = review.lower()
review = review.split()
ps = PorterStemmer()

review = [word for word in review if not word in stopwords.words('english')]
