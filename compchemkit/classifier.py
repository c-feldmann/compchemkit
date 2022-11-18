from __future__ import annotations
from typing import Any, Dict

import numpy as np
import numpy.typing as npt
from scipy import sparse

from compchemkit.kernel import similarity_from_sparse


class TanimotoKNN:
    _params: Dict[str, Any]
    _training_labels: npt.NDArray[np.int_]
    _training_feature_mat: sparse.csr_matrix

    def __init__(self, n_neighbors: int = 1):
        self._params = dict()
        self._params["n_neighbors"] = n_neighbors

    def fit(self, feature_matrix: sparse.csr_matrix, y: npt.NDArray[np.int_]) -> TanimotoKNN:
        if feature_matrix.shape[0] != y.shape[0]:
            raise IndexError("Not same shape")
        self._training_feature_mat = feature_matrix
        self._training_labels = y
        return self

    def predict(self, feature_matrix: sparse.csr_matrix) -> npt.NDArray[np.int_]:
        sim_mat = similarity_from_sparse(feature_matrix, self._training_feature_mat)
        k = self._params["n_neighbors"]
        # get k last indices (k instances with the highest similarity) for each row
        nn_list = np.argsort(sim_mat, axis=1)[:, -k:]
        predicted = []
        for nns in nn_list:
            assert len(nns) == k
            nn_label = self._training_labels[nns]
            label, label_occ = np.unique(nn_label, return_counts=True)
            predicted.append(label[np.argmax(label_occ)])
        return np.array(predicted)

    def fit_predict(self, feature_matrix: sparse.csr_matrix, y: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        self.fit(feature_matrix, y)
        return self.predict(feature_matrix)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return self._params

    def set_params(self, **params: Dict[str, Any]) -> TanimotoKNN:
        self._params = params
        return self
