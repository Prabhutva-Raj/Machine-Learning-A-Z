# -*- coding: utf-8 -*-
"""Created on Fri Nov 15 16:11:38 2019"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("D:/MachineLearning/MachineLearning A to Z/Machine Learning A-Z New/Part 2 - Regression/Section 4 - Simple Linear Regression/Salary_Data.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values
# y = dataset.iloc[:,-1:].values


'''Splitting dataset into training and testing set'''
from sklearn.model_selection import train_test_split
x_train , x_test, y_train, y_test = train_test_split(x,y,test_size=1/3,random_state=0) 


'''feature scaling'''
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train) ##fit the object on training set and then transform
x_test = sc_X.transform(x_test) '''


'''fitting simplelinearRegression model to the training set'''
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
#simple linear regressor machine learnt on the training set

'''predicting the test set results'''
y_pred = regressor.predict(x_test)




'''visualizing the training set results'''
plt.scatter(x_train,y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color='blue' )
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

'''visualizing the test set results'''
plt.scatter(x_test,y_test, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue' )
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()




