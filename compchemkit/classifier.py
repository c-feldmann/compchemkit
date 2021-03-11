import numpy as np
from compchemkit.kernel import TanimotoKernel


class TanimotoKNN:
    def __init__(self, n_neighbors=1):
        self._params = dict()
        self._params["n_neighbors"] = n_neighbors
        self._training_feature_mat = None
        self._training_labels = None

    def fit(self, X, y):
        if X.shape[0] != y.shape[0]:
            raise IndexError("Not same shape")
        self._training_feature_mat = X
        self._training_labels = y
        return self

    def predict(self, X):
        sim_mat = TanimotoKernel.similarity_from_sparse(X, self._training_feature_mat)
        k = self._params["n_neighbors"]
        # get k last indices (k instances with highest similarity) for each row
        nn_list = np.argsort(sim_mat, axis=1)[:, -k:]
        predicted = []
        for nns in nn_list:
            assert len(nns) == k
            nn_label = self._training_labels[nns]
            label, label_occ = np.unique(nn_label, return_counts=True)
            predicted.append(label[np.argmax(label_occ)])
        return np.array(predicted)

    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict(X)

    def get_params(self, deep=True):
        return self._params

    def set_params(self, **params):
        self._params = params
        return self
