import numpy as np
import scipy.sparse as sparse


class TanimotoKernel:
    def __init__(self, sparse_features=True):
        self.sparse_features = sparse_features

    @staticmethod
    def similarity_from_sparse(matrix_a: sparse.csr_matrix, matrix_b: sparse.csr_matrix):
        intersection = matrix_a.dot(matrix_b.transpose()).toarray()
        norm_1 = np.array(matrix_a.multiply(matrix_a).sum(axis=1))
        norm_2 = np.array(matrix_b.multiply(matrix_b).sum(axis=1))
        union = norm_1 + norm_2.T - intersection
        return intersection / union

    @staticmethod
    def similarity_from_dense(matrix_a: np.ndarray, matrix_b: np.ndarray):
        intersection = matrix_a.dot(matrix_b.transpose())
        norm_1 = np.multiply(matrix_a, matrix_a).sum(axis=1)
        norm_2 = np.multiply(matrix_b, matrix_b).sum(axis=1)
        union = np.add.outer(norm_1, norm_2.T) - intersection

        return intersection / union

    def __call__(self, matrix_a, matrix_b):
        if self.sparse_features:
            return self.similarity_from_sparse(matrix_a, matrix_b)
        else:
            raise self.similarity_from_dense(matrix_a, matrix_b)


def tanimoto_from_sparse(matrix_a: sparse.csr_matrix, matrix_b: sparse.csr_matrix):
    DeprecationWarning("Please use TanimotoKernel.sparse_similarity")
    return TanimotoKernel.similarity_from_sparse(matrix_a, matrix_b)


if __name__ == "__main__":
    fp1 = sparse.csr_matrix(np.array([[0, 0, 0, 1],
                                      [0, 0, 1, 1],
                                      [0, 1, 0, 0]]
                                     )
                            )
    fp2 = sparse.csr_matrix(np.array([[0, 0, 0, 1],
                                      [0, 0, 1, 1],
                                      [0, 1, 1, 0],
                                      [1, 0, 0, 0],
                                      [1, 0, 0, 0],
                                      ]
                                     )
                            )
    sim = tanimoto_from_sparse(fp1, fp2)
    print(type(sim))
    print(isinstance(sim, np.ndarray))
    print(sim.shape)
    print(sim)
