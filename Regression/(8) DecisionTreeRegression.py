# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 12:38:31 2020

@author: ADMIN
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("D:\\MachineLearning\\MachineLearning A to Z\\Machine Learning A-Z New\\Part 2 - Regression\\Section 8 - Decision Tree Regression\\Position_Salaries.csv")
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


'''fitting the Decision tree Regression Model to the dataset'''
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0) #default criterion is mean squared value
#random_state to get same value as in video
regressor.fit(x,y)

'''predicting a new result with Decison Tree Regression Model'''
y_pred = regressor.predict(6.5)   # for any new nonLinear regression model


'''visualising the results of Regression'''
plt.scatter(x,y,color="red")
plt.plot(x,regressor.predict(x),color="blue")
plt.title("Truth or Bluff (Decision Tree Regression Model)")
plt.xlabel("Position level") ; plt.ylabel("Salary")
plt.show()
# it is non continuous model.
# splits in several interval and takes avg for value. Hence value be constatnt horizontal in that interval but it is a slope between 2 points.
#it only plotting graph(predictions) for each points and then joining the points with lines because it has no predictions to plot in between.
# solution is 'high resolution'


'''visualising the results of Regression (for higher resolution and smoother curve)'''
x_grid = np.arange(min(x),max(x),0.01)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x,y,color="red")
plt.plot(x_grid,regressor.predict(x_grid),color="blue")
plt.title("Truth or Bluff (Decision Tree Regression Model)")
plt.xlabel("Position level") ; plt.ylabel("Salary")
plt.show()

