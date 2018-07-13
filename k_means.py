#importing all the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

#using elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i , init = 'k-means++' , n_init = 10 , max_iter = 300 , random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11) , wcss)
plt.title('The elbow method')
plt.xlabel('number of clusters')
plt.ylabel('wcss')
plt.show()

#fitting kmeans to the dataset
kmeans = KMeans(n_clusters = 5 , init = 'k-means++' , n_init = 10 , max_iter = 300 , random_state = 0)
y_means = kmeans.fit_predict(X)

#visualising the cluster
plt.scatter(X[y_means == 0,0], X[y_means == 0,1], s =50, c = 'red', label = 'careful')
plt.scatter(X[y_means == 1,0], X[y_means == 1,1], s =50, c = 'blue', label = 'standard')
plt.scatter(X[y_means == 2,0], X[y_means == 2,1], s =50, c = 'green', label = 'target')
plt.scatter(X[y_means == 3,0], X[y_means == 3,1], s =50, c = 'cyan', label = 'careless')
plt.scatter(X[y_means == 4,0], X[y_means == 4,1], s =50, c = 'magenta', label = 'sensible')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 200 , c = 'yellow' , label = 'centroid')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()