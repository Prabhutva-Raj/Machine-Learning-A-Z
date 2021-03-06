"""Created on Sat Apr 18 16:49:32 2020"""

# Classification Template

#---------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("D:\\MachineLearning\\MachineLearning A to Z\\Machine Learning A-Z New\\Part 3 - Classification\Section 14 - Logistic Regression\\Social_Network_Ads.csv")
x = dataset.iloc[:,[2,3]].values    # age,salary
y = dataset.iloc[:,4].values
# y = dataset.iloc[:,-1:].values


'''Splitting dataset into training and testing set'''
from sklearn.model_selection import train_test_split
x_train , x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=0) 


'''feature scaling'''
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train) ##fit the object on training set and then transform
x_test = sc_X.transform(x_test)

#-------------------------------------------------------------------------

'''fitting the classifier to training set'''
'''create own classifier'''

#------------------------------------------------------------------------

'''predicting the test set result'''
y_pred = classifier.predict(x_test)

#-----------------------------------------------------------------------

'''making the confusion matrix'''
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

#-----------------------------------------------------------------------

'''visualizing the training set results'''
from matplotlib.colors import ListedColorMap
x_set, y_set = x_train, y_train
x1, x2 = np.meshgrid()







