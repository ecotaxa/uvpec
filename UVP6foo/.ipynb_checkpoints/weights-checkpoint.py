# source : https://github.com/ThelmaPana/plankton_classif_benchmark/blob/main/train_rf.py
import math
import numpy as np
import pandas as pd

def weights(dataset, use_weights, weight_sensitivity):
    """
    Function that generates class weights 
    """
    class_counts = dataset.groupby('labels').size()
    count_max = 0
    class_weights = {}
    if use_weights:
        count_max = np.max(class_counts)
        for i,idx in enumerate(class_counts.items()):
            class_weights.update({idx[0] : (count_max / idx[1])**weight_sensitivity})
    else:
        for idx in class_counts.items():
            class_weights.update({idx[0] : 1.0})
    
    return(class_weights)