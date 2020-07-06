# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 15:25:25 2019

@author: ADMIN
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("D:\\MachineLearning\\MachineLearning A to Z\\Machine Learning A-Z New\\Part 1 - Data Preprocessing\\Section 2 -------------------- Part 1 - Data Preprocessing --------------------\\Data.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values
# y = dataset.iloc[:,-1:].values


'''Splitting dataset into training and testing set'''
from sklearn.model_selection import train_test_split
x_train , x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0) 


'''feature scaling'''
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train) ##fit the object on training set and then transform
x_test = sc_X.transform(x_test) '''





