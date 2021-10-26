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
    detritus = detritus.drop(columns = 'labels')
    X = detritus.to_numpy()
    x_scaled = standard_scaler.fit_transform(X)
    # sometimes, 'nan' values can occur and kmeans will not work because of that
    index_to_remove = np.unique(np.argwhere(np.isnan(x_scaled))[:,0])
    print('Index(es) removed: ', index_to_remove)
    # remove bad rows from x_scaled
    x_scaled = np.delete(x_scaled, index_to_remove, axis = 0)
    # we also need to remove that line from the detritus AND data dataframe otherwise, we are gonna have a mismatch..
    detritus = detritus.drop(index = index_to_remove)
    data = data.drop(index = index_to_remove)
    # now, do KMEANS
    kmeans = KMeans(n_clusters = n_class, random_state = 42, algorithm="elkan").fit(x_scaled)
    kmeans_labels = kmeans.labels_
    new_detritus = ['detritus_' + str(det) for det in kmeans_labels]
    detritus['labels'] = new_detritus
    data.update(detritus) # update directly the dataset

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
