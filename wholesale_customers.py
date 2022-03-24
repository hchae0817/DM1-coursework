# Part 2: Cluster Analysis

import numpy as np
import pandas as pd

# Return a pandas dataframe containing the data set that needs to be extracted from the data_file.
# data_file will be populated with the string 'wholesale_customers.csv'.
def read_csv_2(data_file):
    data_file = pd.read_csv(data_file)
    data_file.drop(columns = ['Channel','Region'], inplace = True)
    return data_file
    
# Return a pandas dataframe with summary statistics of the data.
# Namely, 'mean', 'std' (standard deviation), 'min', and 'max' for each attribute.
# These strings index the new dataframe columns.
# Each row should correspond to an attribute in the original data and be indexed with the attribute name.
def summary_statistics(df):
    describe_df = df.describe()
    describe_df.drop(describe_df.index[[0,4,5,6]], inplace=True)
    tr_df = describe_df.transpose()
    return round(tr_df) # rounded to closest integer

# Given a dataframe df with numeric values, return a dataframe (new copy)
# where each attribute value is subtracted by the mean and then divided by the
# standard deviation for that attribute.

def standardize(df):
    normalized_df = (df - df.mean()) / df.std()
    normalized_df
    return normalized_df

# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans.
# y should contain values in the set {0,1,...,k-1}.

from sklearn.cluster import KMeans

def kmeans(df, k):
    kmeans = KMeans(n_clusters = k, init = 'random').fit(df)
    #centroids = kmeans.cluster_centers_
    return pd.Series(kmeans.labels_)

# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans++.
# y should contain values from the set {0,1,...,k-1}.
def kmeans_plus(df, k):
    kmeans = KMeans(n_clusters = k).fit(df)
    #centroids = kmeans.cluster_centers_
    return pd.Series(kmeans.labels_)

# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using agglomerative hierarchical clustering.
# y should contain values from the set {0,1,...,k-1}.

from sklearn.cluster import AgglomerativeClustering

def agglomerative(df, k):
    cluster = AgglomerativeClustering(n_clusters = k, affinity='euclidean', linkage='ward')
    return pd.Series(cluster.fit_predict(df))

# Given a data set X and an assignment to clusters y
# return the Solhouette score of the clustering.

from sklearn.metrics import silhouette_score

def clustering_score(X,y):
    score = silhouette_score(X, y, metric='euclidean')
    return score

# Perform the cluster evaluation described in the coursework description.
# Given the dataframe df with the data to be clustered,

# return a pandas dataframe with an entry for each clustering algorithm execution.
# Each entry should contain the:
# 'Algorithm' name: either 'Kmeans' or 'Agglomerative',
# 'data' type: either 'Original' or 'Standardized',
# 'k': the number of clusters produced,
# 'Silhouette Score': for evaluating the resulting set of clusters.

# evaluation is done in kmeans(df,k) as n_init is initialised as 10 as a default

# 12 rows

def cluster_evaluation(df):
    data = [['Kmeans', 'Original', 3, clustering_score(df,kmeans(df, 3))],
            ['Kmeans', 'Standardized', 3, clustering_score(standardize(df),kmeans(standardize(df), 3))],
            
            ['Kmeans', 'Original', 5, clustering_score(df,kmeans(standardize(df), 5))],
            ['Kmeans', 'Standardized', 5, clustering_score(standardize(df),kmeans(standardize(df), 5))],
            
            ['Kmeans', 'Original', 10, clustering_score(df,kmeans(standardize(df), 10))],
            ['Kmeans', 'Standardized', 10, clustering_score(standardize(df),kmeans(standardize(df), 10))],
            
            ['Agglomerative', 'Original', 3, clustering_score(df,agglomerative(df, 3))],
            ['Agglomerative', 'Standardized', 3, clustering_score(standardize(df),agglomerative(standardize(df), 3))],
            
            ['Agglomerative', 'Original', 5, clustering_score(df,agglomerative(df, 5))],
            ['Agglomerative', 'Standardized', 5, clustering_score(standardize(df),agglomerative(standardize(df), 5))],
            
            ['Agglomerative', 'Original', 10, clustering_score(df,agglomerative(df, 10))],
            ['Agglomerative', 'Standardized', 10, clustering_score(standardize(df),agglomerative(standardize(df), 10))]]
 
    # Create the pandas DataFrame
    rdf = pd.DataFrame(data, columns = ['Algorithm', 'data', 'k', 'Silhouette Score'])
    
    return rdf

# Given the performance evaluation dataframe produced by the cluster_evaluation function,
# return the best computed Silhouette score.
def best_clustering_score(rdf):
    # Identify which run resulted in the best set of clusters using the Silhouette score as your evaluation metric.
    bestScore = rdf['Silhouette Score'].max()
    return bestScore

# Run some clustering algorithm of your choice with k=3 and generate a scatter plot for each pair of attributes.
# Data points in different clusters should appear with different colors.

# Visualize the best set of clusters computed in the previous question.
# For this, construct a scatterplot for each pair of attributes using Pyplot.
# Therefore, 15 scatter plots should be constructed in total.
# Different clusters should appear with different colors in each scatter plot.
# Note that these plots could be used to manually assess how well the clusters separate the data points.

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations


def scatter_plots(df):

    list_pairs = list(combinations(df.columns,2))
    colors = ("red", "green", "blue")

    labels = kmeans(df,3)

    label_0 = df[labels == 0]
    label_1 = df[labels == 1]
    label_2 = df[labels == 2]

    for list_pair in list_pairs:
    
        # Create data
        g1 = (label_0[list_pair[0]], label_0[list_pair[1]])
        g2 = (label_1[list_pair[0]], label_1[list_pair[1]])
        g3 = (label_2[list_pair[0]], label_2[list_pair[1]])
    
        data = (g1,g2,g3)

        for i in range(len(data)):
            plt.xlabel(list_pair[0])
            plt.ylabel(list_pair[1])
            plt.scatter(data[i][0], data[i][1], c = colors[i])
        
        # one graph with all g1,g2,g3 data
        plt.title('Scatter plot')
        plt.show()
