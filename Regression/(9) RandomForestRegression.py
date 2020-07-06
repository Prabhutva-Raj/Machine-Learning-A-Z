# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 17:10:43 2020

@author: ADMIN
"""

# random forest is based on ensemble learning, which is more stable
#ensemble learning :-when you ntake multiple algo (or same algo multiple times) and put them together to make it more powerful

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("D:\\MachineLearning\\MachineLearning A to Z\\Machine Learning A-Z New\\Part 2 - Regression\\Section 9 - Random Forest Regression\\Position_Salaries.csv")
x = dataset.iloc[:,1:2].values      #should always be matrix
y = dataset.iloc[:,2].values        #should always be vector
# y = dataset.iloc[:,-1:].values


'''Splitting dataset into training and testing set'''  #no need to split as data is already less
'''from sklearn.model_selection import train_test_split
x_train , x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0) 
'''

'''feature scaling'''  #since linearregressor library itself does feature scaling
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train) ##fit the object on training set and then transform
x_test = sc_X.transform(x_test) '''


'''fitting the Regression Model to the dataset'''
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300, random_state=0)
# n_estimators are the number of tress we want
regressor.fit(x,y)


'''predicting a new result with Regression Model'''
y_pred = regressor.predict(6.5)   # for any new nonLinear regression model


'''visualising the results of Regression (for higher resolution and smoother curve)'''
x_grid = np.arange(min(x),max(x),0.01)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x,y,color="red")
plt.plot(x_grid,regressor.predict(x_grid),color="blue")
plt.title("Truth or Bluff (Regression Model)")
plt.xlabel("Position level") ; plt.ylabel("Salary")
plt.show()

