"""Created on Sun Apr 19 17:48:06 2020"""

# Kernal SVM

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

'''fitting logictic regression to training set'''
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=0)   # rbf = gausian kernal
classifier.fit(x_train, y_train)

#------------------------------------------------------------------------

'''predicting the test set result'''
y_pred = classifier.predict(x_test)

#-----------------------------------------------------------------------

'''making the confusion matrix'''
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

#-----------------------------------------------------------------------

'''visualizing the training set results'''

from matplotlib.colors import ListedColormap

x_set, y_set = x_train, y_train
x1, x2 = np.meshgrid(np.arange(start=x_set[:,0].min()-1 , stop=x_set[:,0].max()+1 , step=0.01) ,
                     np.arange(start=x_set[:,1].min()-1 , stop=x_set[:,1].max()+1 , step=0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('red','green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j, 0], x_set[y_set==j, 1],
                c = ListedColormap(('red','green'))(i), label=j )
    
plt.title('KernelSVM (Training Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

#------------------------------------------------------------------------


'''visualizing the test set results'''

from matplotlib.colors import ListedColormap

x_set, y_set = x_test, y_test
x1, x2 = np.meshgrid(np.arange(start=x_set[:,0].min()-1 , stop=x_set[:,0].max()+1 , step=0.01) ,
                     np.arange(start=x_set[:,1].min()-1 , stop=x_set[:,1].max()+1 , step=0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('red','green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j, 0], x_set[y_set==j, 1],
                c = ListedColormap(('red','green'))(i), label=j )
    
plt.title('KernelSVM (Test Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
