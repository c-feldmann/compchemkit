import numpy as np
import numpy.typing as npt
import scipy.sparse as sparse


def similarity_from_dense(
    matrix_a: npt.NDArray[np.int_], matrix_b: npt.NDArray[np.int_]
) -> npt.NDArray[np.float_]:
    intersection = matrix_a.dot(matrix_b.transpose())
    norm_1 = np.multiply(matrix_a, matrix_a).sum(axis=1)
    norm_2 = np.multiply(matrix_b, matrix_b).sum(axis=1)
    union = np.add.outer(norm_1, norm_2.T) - intersection
    return intersection / union


def similarity_from_sparse(
    matrix_a: sparse.csr_matrix, matrix_b: sparse.csr_matrix
) -> npt.NDArray[np.float_]:
    intersection = matrix_a.dot(matrix_b.transpose()).toarray()
    norm_1 = np.array(matrix_a.multiply(matrix_a).sum(axis=1))
    norm_2 = np.array(matrix_b.multiply(matrix_b).sum(axis=1))
    union = norm_1 + norm_2.T - intersection
    return intersection / union


if __name__ == "__main__":
    fp1 = sparse.csr_matrix(np.array([[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 0, 0]]))
    fp2 = sparse.csr_matrix(
        np.array(
            [
                [0, 0, 0, 1],
                [0, 0, 1, 1],
                [0, 1, 1, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
            ]
        )
    )
    sim = similarity_from_sparse(fp1, fp2)
    print(type(sim))
    print(isinstance(sim, np.ndarray))
    print(sim.shape)
    print(sim)
