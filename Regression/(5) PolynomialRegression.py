# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 12:02:51 2020

@author: ADMIN
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("D:\\MachineLearning\\MachineLearning A to Z\\Machine Learning A-Z New\\Part 2 - Regression\\Section 6 - Polynomial Regression\\Position_Salaries.csv")
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


'''fitting linear regression to the dataset'''
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

'''fitting polynomial regression to the dataset'''
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)
poly_reg.fit(x_poly,y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly,y)

'''visualising the results of linear regression'''
plt.scatter(x,y,color="red")
plt.plot(x,lin_reg.predict(x),color="blue")
plt.title("Truth or Bluff (Linear Regression)")
plt.xlabel("Position level") ; plt.ylabel("Salary")
plt.show()

'''visualising the results of polynomial regression'''
x_grid = np.arange(min(x),max(x),0.1)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x,y,color="red")
#plt.plot(x,lin_reg_2.predict(x_poly),color="blue")
plt.plot(x_grid,lin_reg_2.predict(poly_reg.fit_transform(x_grid)),color="blue")
plt.title("Truth or Bluff (Polynomial Regression)")
plt.xlabel("Position level") ; plt.ylabel("Salary")
plt.show()





'''predicting a new result with linear rregression'''
lin_reg.predict(6.5)

'''predicting a new result with polynomial rregression'''
lin_reg_2.predict(poly_reg.fit_transform(6.5))




