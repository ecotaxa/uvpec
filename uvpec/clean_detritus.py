import numpy as np
from sklearn.ensemble import IsolationForest

def remove_outliers(data):
    
    print(data.shape)

    # isolate the detritus
    detritus = data[data['labels'] == 'detritus']
    detritus = detritus.drop(columns = 'labels')

    # check for NaN values (need to remove them for the IsolationForest and Kmeans
    index_to_remove = np.array(detritus[detritus.isnull().any(axis=1)].index)
    print('Index(es) removed: ', index_to_remove)
    # remove bad rows from detritus AND data (otherwise -> we'll have a mismatch)
    detritus = detritus.drop(index = index_to_remove)
    data = data.drop(index = index_to_remove)
    # That being done, now we can use the Isolation Forest to remove outliers
    search_outliers = IsolationForest(n_estimators = 100, n_jobs = 6, random_state = 42).fit_predict(detritus)
    print("Number of outliers:", len(search_outliers[search_outliers < 1]))
    outliers = np.concatenate(np.array(np.where(search_outliers == -1)))
    rows_to_remove = detritus.iloc[outliers].index
    data = data.drop(rows_to_remove, axis = 0)
    print(data.shape)
    return(data)
