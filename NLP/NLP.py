"""Created on Mon May  4 15:53:34 2020"""

# Natural Language Processing (NLP) 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting=3)   # by quoting=3, we are ignoring the double quotes

#------------------------------------------------------------------------------------

''' Cleaning the text '''
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []                     # corpus : collection of list
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()          # stemming : keep only the 'root word'
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = " ".join(review)
    corpus.append(review)
    
#------------------------------------------------------------------------------------

''' Creating Bag of words model '''
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)   #1500  most frequent features (words)
x = cv.fit_transform(corpus).toarray()      # creates the sparse matrix of features(words)
y = dataset.iloc[:,1].values

#------------------------------------------------------------------------------------

'''Using naive bayes classifier'''

'''Splitting dataset into training and testing set'''
from sklearn.model_selection import train_test_split
x_train , x_test, y_train, y_test = train_test_split(x,y,test_size=0.20,random_state=0) 

'''fitting logictic regression to training set'''
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()   # no parameters req in this
classifier.fit(x_train, y_train)

'''predicting the test set result'''
y_pred = classifier.predict(x_test)

'''making the confusion matrix'''
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

'''accuracy'''
accuracy = (cm[0][0]+cm[1][1])/200       # (55+91)/total_size
print(accuracy)