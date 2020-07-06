"""Created on Wed Apr 22 13:54:15 2020"""

# Hierarchical Clustering

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("D:\\MachineLearning\\MachineLearning A to Z\\Machine Learning A-Z New\\Part 4 - Clustering\\Section 24 - K-Means Clustering\\Mall_Customers.csv")
x = dataset.iloc[:,[3,4]].values

#----------------------------------------------------------------------------------

''' using dendogram to get optimal no of clusters''' '''metrics used: wcv(within cluster variance)'''
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(x, method='ward'))
# linkage = algorithm of hierarchical clustering
# x = data on which we apply linkage
# method is that is used to find the clusters
# ward method tries to minimize the variance within each cluster (so here instead of minimizing the wcss, we are minimizing variance)
plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Eucledian distances')
plt.show()

#----------------------------------------------------------------------------------

'''fitting heirarchical clustering to the mall dataset'''
from sklearn.cluster import AgglomerativeClustering    # among the 2 types of hc
hc = AgglomerativeClustering(n_clusters = 5, affinity='euclidean', linkage='ward')  #as we used 'ward' only to get dendogram
y_hc = hc.fit_predict(x)

#----------------------------------------------------------------------------------

'''visualizing the clusters'''
plt.scatter(x[y_hc==0,0] , x[y_hc==0,1], s=100, c='red', label='Careful')
plt.scatter(x[y_hc==1,0] , x[y_hc==1,1], s=100, c='blue', label='Standard')
plt.scatter(x[y_hc==2,0] , x[y_hc==2,1], s=100, c='green', label='Target')
plt.scatter(x[y_hc==3,0] , x[y_hc==3,1], s=100, c='cyan', label='Careless')
plt.scatter(x[y_hc==4,0] , x[y_hc==4,1], s=100, c='magenta', label='Sensible')
# x[y_hc==0,0] = plot from first column of x (i.e. 0) where data belongs to cluster1(y_hc==0)
plt.title('Clusters of client')
plt.xlabel('Annual Income (K$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

