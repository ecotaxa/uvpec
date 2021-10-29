from scipy.spatial.distance import euclidean
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
standard_scaler = preprocessing.StandardScaler()
import pandas as pd

def create_detritus_classes(n_class, data, key, output_dir):
    """
    Function that does a KMEANS on the 'detritus' class and split it on n class, ie. detritus_0, detritus_1, ..., detritus_N
    """
    # subset detritus from the training set (features)
    detritus = data[data['labels'] == 'detritus']
    detritus = detritus.drop(columns = 'labels')

    # get the mean and variance for scaling the test set later 
    #X = detritus.to_numpy() # actually not needed for the standard scaler to work
    standard_scaler.fit(detritus)
    scale_mean = standard_scaler.mean_
    scale_var = standard_scaler.var_

    # now we can really scale (fit AND transform)
    x_scaled = standard_scaler.fit_transform(detritus)
    # now, do KMEANS
    kmeans = KMeans(n_clusters = n_class, random_state = 42, algorithm="elkan").fit(x_scaled)
    kmeans_labels = kmeans.labels_
    coord_centroids = kmeans.cluster_centers_
    new_detritus = ['detritus_' + str(det) for det in kmeans_labels]
    detritus['labels'] = new_detritus
    data.update(detritus) # update directly the dataset

    # save data for later use
    np.save(os.path.join(output_dir, 'coord_centroids_'+key+'.npy'), coord_centroids)
    np.save(os.path.join(output_dir, 'scale_mean_'+key+'.npy'), scale_mean)
    np.save(os.path.join(output_dir, 'scale_var_'+key+'.npy'), scale_var)

    return(data, scale_mean, scale_var, coord_centroids)
