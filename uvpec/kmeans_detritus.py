from scipy.spatial.distance import euclidean
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
standard_scaler = preprocessing.StandardScaler()
import pandas as pd

def create_detritus_classes(n_class, data):
    """
    Function that does a KMEANS on the 'detritus' class and split it on n class, ie. detritus_0, detritus_1, ..., detritus_N
    """
    detritus = data[data['labels'] == 'detritus']
    detritus.drop(columns = 'labels', inplace=True)
    X = detritus.to_numpy()
    x_scaled = standard_scaler.fit_transform(X)
    kmeans = KMeans(n_clusters = n_class, precompute_distances = True, random_state = 42, n_jobs = 6, algorithm="elkan").fit(x_scaled)
    kmeans_labels = kmeans.labels_
    new_detritus = ['detritus_' + str(det) for det in kmeans_labels]
    detritus['labels'] = new_detritus
    #data.update(detritus) # update directly the dataset

    # compute euclidian distance
    #distances = [] # respective to the cluster
    #cluster_label = []
    #for i in range(X.shape[0]):
    ##for i in range(10):
    #    cluster = kmeans.labels_[i]
    #    tmp = euclidean(x_scaled[i], kmeans.cluster_centers_[cluster])
    #    distances.append(tmp)
    #    cluster_label.append(cluster)
    #detritus['cluster'] = cluster_label
    #detritus['distance'] = distances
    ##return(distances, cluster_label, detritus)
    #data.update(detritus)
    return(True)
