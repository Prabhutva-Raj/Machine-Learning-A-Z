# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 21:55:06 2019

@author: ADMIN
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("D:\\MachineLearning\\MachineLearning A to Z\\Machine Learning A-Z New\\Part 1 - Data Preprocessing\\Section 2 -------------------- Part 1 - Data Preprocessing --------------------\\Data.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values
# y = dataset.iloc[:,-1:].values


'''take care of missing data'''
from sklearn.preprocessing import Imputer   #'''takes care of missing values'''
imputer = Imputer(missing_values = 'NaN', strategy = 'mean' , axis = 0)   #makes object
imputer = imputer.fit(x[:,1:3])    # fit imputer to x
x[:,1:3] = imputer.transform(x[:,1:3])


'''encoding categorial data (here country and purchased)'''
from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
x[:,0] = labelencoder_x.fit_transform(x[:,0]) 

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)





'''Splitting dataset into training and testing set'''
# from sklearn.cross_validation import train_test_split
#(above is now depriciated)
from sklearn.model_selection import train_test_split
x_train , x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0) 
#0.2 = 20% of data  
#random state=0 to have same result as the tutorial



'''feature scaling'''
#variables are not on same scale : can create issue as most of them use eucledian dist.
# so the variable which is more big it will be dominant uselessly
# Two types:: Standardization and Normalization
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train) ##fit the object on training set and then transform
x_test = sc_X.transform(x_test)  # no need to "fit", as already fitted the training set 
#Do we need to fit the dummy variables ? No, already scaled(depends on context)

    



