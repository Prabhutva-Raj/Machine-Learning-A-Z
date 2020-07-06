"""Created on Tue Apr 21 20:11:14 2020"""

# K Means Clustering
#%reset -f

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("D:\\MachineLearning\\MachineLearning A to Z\\Machine Learning A-Z New\\Part 4 - Clustering\\Section 24 - K-Means Clustering\\Mall_Customers.csv")
x = dataset.iloc[:,[3,4]].values

#----------------------------------------------------------------------------------

''' using elbow method to find optimal clusters'''  '''metrics used: wcss(within clusters sum of square)'''
from sklearn.cluster import KMeans
wcss = []  # within clusters sum of square = inertia
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)  
    # init is the initialization of centroids
    # max-iter = maximum number of iterations to find clusters when kmeans is running
    # n_init = number of times k means algorithm will be run with diff initial centroids
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('No of clusters')
plt.ylabel('wcss')
plt.show()           #by th eelbow graph we get clusters = 5


#-----------------------------------------------------------------------------------

'''Applying kmeans to mall dataset'''
kmeans = KMeans(n_clusters = 5, init='k-means++',max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(x)  # fitpredict will give the cluster to which customer belongs
# fit_predict returns the vector of the cluster

#-----------------------------------------------------------------------------------
    
'''Visualizing the clusters'''
plt.scatter(x[y_kmeans==0,0] , x[y_kmeans==0,1], s=100, c='red', label='Careful')
plt.scatter(x[y_kmeans==1,0] , x[y_kmeans==1,1], s=100, c='blue', label='Standard')
plt.scatter(x[y_kmeans==2,0] , x[y_kmeans==2,1], s=100, c='green', label='Target')
plt.scatter(x[y_kmeans==3,0] , x[y_kmeans==3,1], s=100, c='cyan', label='Careless')
plt.scatter(x[y_kmeans==4,0] , x[y_kmeans==4,1], s=100, c='magenta', label='Sensible')
# x[y_means==0,0] = plot from first column of x (i.e. 0) where data belongs to cluster1(y_means==0)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300, c='yellow', label='centroids')
plt.title('Clusters of client')
plt.xlabel('Annual Income (K$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
