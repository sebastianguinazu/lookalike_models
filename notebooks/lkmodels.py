import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import distance

def scores_centroide(X_seed, X_poblacion, robust=False):
    if robust:
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
    scaler.fit(X_seed)
    X_poblacion_scaled = scaler.transform(X_poblacion)
    centroide = scaler.center_ if robust else scaler.mean_
    distancias_poblacion = distance.cdist(X_poblacion_scaled
                                         ,centroide[np.newaxis,:], 'euclidean')
    scores = 1/(distancias_poblacion+1)
    scores_minmax = (scores - scores.min()) / (scores.max() - scores.min())
    return scores_minmax.ravel()


def scores_distancias(X_seed, X_poblacion, robust=False):
    if robust:
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
    scaler.fit(X_seed)
    X_seed_scaled = scaler.transform(X_seed)
    X_poblacion_scaled = scaler.transform(X_poblacion)
    distancias_matrix = distance.cdist(X_poblacion_scaled
                                      ,X_seed_scaled, 'euclidean')
    similarities_matrix = 1/(distancias_matrix+1)
    scores = similarities_matrix.mean(axis=1)
    scores_minmax = (scores - scores.min()) / (scores.max() - scores.min())
    return scores_minmax.ravel()

def scores_clf(X_seed, X_poblacion, random_state, clf='rf', **kwargs_clf):
    """ 
    Returns predictions of classifier trained with poblacion+seed
    Param:
        - clf: base estimator (one of rf, logistic)
    """
    # features: primero poblacion - luego seed
    X_train = np.concatenate([X_poblacion, X_seed])
    # vector target: primero poblaicion - luego seed
    y_train = np.concatenate([np.ones(X_poblacion.shape[0]), np.zeros(X_seed.shape[0])])
    # train
    if clf=='rf':
        clf = RandomForestClassifier(oob_score=True, random_state=random_state, **kwargs_clf)
    elif clf=='logistic':
        clf = LogisticRegression(random_state=random_state, **kwargs_clf)
    elif clf=='tree':
        clf = DecisionTreeClassifier(random_state=random_state, **kwargs_clf)
    elif clf=='knn':
        clf = KNeighborsClassifier(**kwargs_clf)
    clf.fit(X_train, y_train)
    # predict en poblacion
    if clf=='rf':
        scores = clf.oob_score_[:X_poblacion.shape[0]]
    else:
        scores = clf.predict_proba(X_poblacion)[:,clf.classes_ == 1].ravel()
    return scores


def scores_bagged_clf(X_seed, X_poblacion, random_state, T=50, clf='rf', **kwargs_clf):
    """
    Returns avg of oob predictions of classifier para la poblacion
    Param:
        - T number of baggint iteractions 
        - clf: base estimator (one of rg, logistic)
    """
    # K: size of boostrap sample (= size of seed)
    K = X_seed.shape[0]
    # U: size of poblation
    U = X_poblacion.shape[0]
    # se entrena con una muestra balanceada
    # vector target: primero seed - luego poblacion
    y_train = np.concatenate([np.ones(K), np.zeros(K)])
    # initialize numerador de predicciones
    pred = np.zeros(U)
    # initialize denominador de predicciones
    n = np.zeros(U)
    # bagging
    for t in range(T):
        # get sample
        idx_train = random_state.choice(U, K, replace=True)
        X_train = np.concatenate([X_seed, X_poblacion[idx_train,:]])
        # train
        if clf=='rf':
            clf = RandomForestClassifier(random_state=random_state, **kwargs_clf)
        if clf=='logistic':
            clf = LogisticRegression(random_state=random_state, **kwargs_clf)
        if clf=='tree':
            clf = DecisionTreeClassifier(random_state=random_state, **kwargs_clf)
        if clf=='knn':
            clf = KNeighborsClassifier(random_state=random_state, **kwargs_clf)
        clf.fit(X_train, y_train)
        # predict OOB
        idx_oob = np.full(U, True)
        idx_oob[idx_train] = False
        _pred = clf.predict_proba(X_poblacion[idx_oob,:])[:,clf.classes_ == 1].ravel()
        pred[idx_oob] += _pred
        n[idx_oob] += 1
    scores = pred / n
    return scores