"""Created on Sat May  9 14:58:49 2020"""


# ------------- Part1 - Data Preprocessing--------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Churn_Modelling.csv")
x = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values

'''encoding categorial data (here country and purchased)'''
from sklearn.preprocessing import LabelEncoder
labelencoder_x_1 = LabelEncoder()
x[:,1] = labelencoder_x_1.fit_transform(x[:,1]) 
labelencoder_x_2 = LabelEncoder()
x[:,2] = labelencoder_x_2.fit_transform(x[:,2]) 
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [1])
x = onehotencoder.fit_transform(x).toarray()
x = x[:, 1:]

'''Splitting dataset into training and testing set'''
from sklearn.model_selection import train_test_split
x_train , x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=0) 

'''feature scaling'''
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train) ##fit the object on training set and then transform
x_test = sc_X.transform(x_test)


# ------------- Part2 - Making ANN ----------------------------------------------------------------

'''importing keras libraries and packages'''
import keras   # defaultly use tensorflow instead of thenos
from keras.models import Sequential      #initialise NN
from keras.layers import Dense           #use to create the layers in ANN
'''initializing ANN'''
classifier = Sequential()   # since we are going to do classification work, this neural network will be a classifier
'''adding the input layer and first hidden layer'''
classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=11)) # add method : use to add diff layers in NN (no of nodes in first hidden layer)
#Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform")`
# 6 = 11 (input nodes) + 1 (output nodes)
#uniform = initialises weights randomly and makes sure that weights are close to 0
'''adding second hidden layer'''
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu')) 
'''adding the output layer'''
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid')) 
# for more than 2 categories we do 'units = #ofcategories' and 'activation='softmax'' '
#softmax is sigmoid fuction only but modified for more than 2 category cases

'''compiling the ANN''' #i.e. applying the stochastic gradient descent of the ANN and finding the appropriate weights
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# if outcome is >2 then loss = categorical

'''fitting ANN to the training set'''
classifier.fit(x_train, y_train, batch_size=10, nb_epoch=100)
# batch_size = no of observations after which you want to update the weights
# epoch = no of times the whole training set would pas through the ANN 


# -------------- Part3 - Making prediction and evaluating the model------------------------------------------------------------------------

'''predicting the test set result'''
y_pred = classifier.predict(x_test)
y_pred = (y_pred>0.5)  # of y_pred>0.5 return true

'''making the confusion matrix'''
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

#-----------------------------------------------------------------------