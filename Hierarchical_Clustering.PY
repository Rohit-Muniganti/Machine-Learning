# Hierarchical Clustering

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# Importing the dataset
dataset = pd.read_csv(r"C:\Users\rohit\OneDrive\Desktop\FSDS & GEN-AI 25TH-NOV-024.NOTES\4th-Feb-025 - Clustering,\3.HIERARCHICAL CLUSTERING\Mall_Customers.csv")
X = dataset.iloc[:, [3, 4]].values


import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))

plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()


# YOU CAN IMPLETE HEAR FIND ELBOW METOD 


# Training the Hierarchical Clustering model on the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, metric = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


#New column
dataset['Clusters']=y_hc

dataset.to_csv("filename.csv", index=False)

import os
os.getcwd()
with open('Hierarchy_Clustering.pkl', 'wb') as file:
    pickle.dump(dataset, file)
    print("Model has been pickled and saved to Hierarchy_Clustering.pkl")
 
import os
os.getcwd()

