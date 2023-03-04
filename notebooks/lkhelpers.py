import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import average_precision_score

def make_dataset(random_state, n_samples=10000, n_features=20, n_informative=15
                 ,n_redundant=5, n_repeated=0, n_classes=2, n_clusters_per_class=1
                 ,weights=[0.90,0.10], flip_y=0.1, class_sep=1.0
                 ,contamination=0.5):
    """
    Creates (X, y, ids_hidden)
    - y is 1 if seed, 0 caso contrario (hidden+poblacion)
    - idx has the indices of hidden obs
    """
    X, y = make_classification(n_samples=n_samples
                           ,n_features=n_features
                           ,n_informative=n_informative
                           ,n_redundant=n_redundant
                           ,n_repeated=n_repeated 
                           ,n_classes=n_classes
                           ,n_clusters_per_class=n_clusters_per_class
                           ,weights=weights
                           ,flip_y=flip_y
                           ,class_sep=class_sep
                           ,random_state=random_state)
    idx_positivos = np.argwhere(y == 1).ravel()
    idx_hidden = random_state.choice(idx_positivos
                                    ,size=int(len(idx_positivos)*contamination)
                                    ,replace=False)
    y[idx_hidden] = 0
    return X, y, idx_hidden

# function para performance
def avg_precision(y, idx_hidden, scores):
    y_true = y.copy()
    y_true[idx_hidden] = 1
    y_true = y_true[y == 0]
    return average_precision_score(y_true, scores)