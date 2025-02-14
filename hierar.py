import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import pickle
from io import StringIO

# Title of the web app
st.title("Hierarchy Clustering of Mall Customer")

# Upload file functionality
st.sidebar.header("Upload Your Dataset")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    dataset = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.write(dataset.head())

   
    if 'Annual income (k$)' in dataset.columns and 'Spending score (1-100)' in dataset.columns:
        X = dataset[['Annual income (k$)', 'Spending score (1-100)']].values

        # Dendrogram
        st.write("Dendrogram")
        fig, ax = plt.subplots()
        sch.dendrogram(sch.linkage(X, method = 'ward'))
        plt.title('Dendrogram')
        plt.xlabel('Customers')
        plt.ylabel('Euclidean distances')
        plt.show(fig)

        n_clusters = st.slider("Select number of clusters", min_value=2, max_value=10, value=5)


        # Hierarchical clustering 
        hc = AgglomerativeClustering(n_clusters = 5, metric = 'euclidean', linkage = 'ward')
        y_hc = hc.fit_predict(X)

        st.write("Clusters of Customers")
        fig, ax = plt.subplots()
        colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'orange', 'brown']

        for i in range(n_clusters):
            plt.scatter(X[y_hc == i, 0], X[y_hc == i, 1], s=100, c=colors[i], label=f'cluster{i+1}')
        

        # Visualize the clusters
        plt.title('Clusters of customers')
        plt.xlabel('Annual Income (k$)')
        plt.ylabel('Spending Score (1-100)')
        plt.legend()
        plt.show(fig)
    else:
        st.write("The uploaded CSV must contain 'Annual income (k$)' and 'Spending score (1-100)'")
else:
    st.write("Please upload a CSV file to proceed.")


 

    