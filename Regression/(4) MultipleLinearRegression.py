import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("D:\\MachineLearning\\MachineLearning A to Z\\Machine Learning A-Z New\\Part 2 - Regression\\Section 5 - Multiple Linear Regression\\50_Startups.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values
# y = dataset.iloc[:,-1:].values


'''encoding categorial data '''
from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
x[:,3] = labelencoder_x.fit_transform(x[:,3]) 

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()


'''avoiding dummy variable trap'''
x = x[:,1:]

'''Splitting dataset into training and testing set'''
from sklearn.model_selection import train_test_split
x_train , x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0) 



'''fitting multiple linear regression model to the training set'''
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

'''predicting test_set results'''
y_pred = regressor.predict(x_test)



##########

''' building an optimal model using backward elimination '''
import statsmodels.formula.api as sm   
# this model neglects the constant b0 unlike previous models. 
# Hence in order to make b0 count, append ones column which would act as x0 for b0
#x = np.append(arr=x, values=np.ones((50,1)).astype(int), axis=1)
x = np.append(arr=np.ones((50,1)).astype(int), values=x, axis=1)

x_opt = x[:,[0,1,2,3,4,5]]   #select all index of x at first
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt = x[:,[0,1,3,4,5]]   #removing 2 index of x_opt
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt = x[:,[0,3,4,5]]   #removing 1 index of x_opt
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt = x[:,[0,3,5]]   #removing 2 index of x_opt
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt = x[:,[0,3]]   #removing 2 index of x_opt
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()

'''x_train , x_test, y_train, y_test = train_test_split(x_opt,y,test_size=0.2,random_state=0) 
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred_after_bk_ele = regressor.predict(x_test)'''



