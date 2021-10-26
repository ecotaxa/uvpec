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
    # now, we can scale the data
    X = detritus.to_numpy()
    x_scaled = standard_scaler.fit_transform(X)
    # now, do KMEANS
    kmeans = KMeans(n_clusters = n_class, random_state = 42, algorithm="elkan").fit(x_scaled)
    kmeans_labels = kmeans.labels_
    new_detritus = ['detritus_' + str(det) for det in kmeans_labels]
    detritus['labels'] = new_detritus
    data.update(detritus) # update directly the dataset
    return(data)
