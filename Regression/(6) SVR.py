# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 23:25:36 2020

@author: ADMIN
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("D:\\MachineLearning\\MachineLearning A to Z\\Machine Learning A-Z New\\Part 2 - Regression\\Section 6 - Polynomial Regression\\Position_Salaries.csv")
x = dataset.iloc[:,1:2].values      #should always be matrix
y = dataset.iloc[:,2].values
y = np.reshape(y,(-1,1))        #should always be vector
# y = dataset.iloc[:,-1:].values


'''Splitting dataset into training and testing set'''  #no need to split as data is already less
'''from sklearn.model_selection import train_test_split
x_train , x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0) 
'''

'''feature scaling'''  #since SVR library itself does NOT do feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x) ##fit the object on training set and then transform
y = sc_y.fit_transform(y)


'''fitting the SVR to the dataset'''
from sklearn.svm import SVR
regressor = SVR(kernel="rbf")  #we wont take kernel=linear as our problem is non linear
#we take most common rbf ...for gaussian kernel
regressor.fit(x,y)


'''predicting a new result with Regression Model'''
y_pred = regressor.predict(sc_x.transform(np.array([[6.5]])))   # for any new nonLinear regression model
# since, we applied feature scaling to data x and y
# we have to feature scaling and transform 6.5 also, hence transform
# Now, As the transform func required array as parameter, we used numpyArray
# Single sq braces will make it a vector of one element, we want array, hence double sq braces

#Aftre all this, we will get the scaled prediction. hence we have to inverse the scale transformation
y_pred = sc_y.inverse_transform(y_pred)



'''visualising the results of SVR'''
plt.scatter(x,y,color="red")
plt.plot(x,regressor.predict(x),color="blue")
plt.title("Truth or Bluff (SVR)")
plt.xlabel("Position level") ; plt.ylabel("Salary")
plt.show()


'''visualising the results of Regression (for higher resolution and smoother curve)'''
x_grid = np.arange(min(x),max(x),0.1)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x,y,color="red")
plt.plot(x_grid,regressor.predict(x_grid),color="blue")
plt.title("Truth or Bluff (SVR Model)")
plt.xlabel("Position level") ; plt.ylabel("Salary")
plt.show()

