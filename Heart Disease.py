# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 17:57:43 2020

@author: neelabh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

raw_df = pd.read_csv('heart.csv')
print(raw_df.info(), raw_df.head(), raw_df.isna().sum())

# This a discrete datset, hence classification models are to used
# Classification model

# Splitting the dataset
train_x, test_x, train_y, test_y = train_test_split(raw_df.drop(['target'], axis =1), raw_df.filter(['target'], axis =1), test_size = 0.2, random_state = 0)
regressor = LogisticRegression()
regressor.fit(train_x, train_y)
pred_y = regressor.predict(test_x)
accuracy = accuracy_score(test_y, pred_y)


print(accuracy)

