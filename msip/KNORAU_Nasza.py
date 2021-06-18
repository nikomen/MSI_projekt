import numpy as np
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import DistanceMetric


class KNORA_U_nasz(BaseEstimator, ClassifierMixin):

    def __init__(self, pool_classifiers=None, k=7, random_state=66):

        self.pool_classifiers = pool_classifiers
        self.k = k
        self.random_state = random_state

        np.random.seed(self.random_state)

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.X_, self.y_ = X, y

        return self

    # Określamy region kompetencji
    def region_of_competence(self, actual_sample, sample_bank):
        reg_com = []
        distance = DistanceMetric.get_metric("euclidean")
        for i in sample_bank:
            score = np.array(
                distance.pairwise([actual_sample, i[0]])).max()
            reg_com.append([i, score])

        reg_com = sorted(reg_com, key=lambda t: t[1])[: self.k]

        return reg_com

    # Sprawdzamy czy klasyfikator dokonał poprawnej predykcji, jeżeli tak to dadajemy go tablicy.
    # Każdy kasyfikator będzie w tablcy tyle razy ile poprawnie dokonał predykcji.
    # Jeżeli żaden nie dokonał poprawnie predykcji to wszystkie klasyfikatory będą w tablicy jeden raz.
    def selection(self, ensemble, reg_com):
        for clf in self.pool_classifiers:
            for i in reg_com:
                pred = clf.predict(i[0][0].reshape(1, -1))
                if pred == i[1]:
                    ensemble.append(clf)
        if(len(ensemble) != 0):
         return ensemble
        else:
         for clf in self.pool_classifiers:
            ensemble.append(clf)
         return ensemble

    def predict(self, X_test):
        check_is_fitted(self)
        X_test = check_array(X_test)

        y_predict = []

        for actual_sample in X_test:
            reg_com = self.region_of_competence(actual_sample, zip(self.X_, self.y_))
            tab_good_classifier = []
            tab_good_classifier = self.selection(tab_good_classifier,reg_com)

            # majority voting
            suma = 0
            for clf in tab_good_classifier:
                value = clf.predict(actual_sample.reshape(1, -1))
                suma += value

            if suma <= (len(tab_good_classifier) / 2):
                y_predict.append(0)
            else:
                y_predict.append(1)

        return y_predict
