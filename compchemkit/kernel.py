"""Kernel functions for compound similarities."""

import numpy as np
import numpy.typing as npt
from scipy import sparse


def similarity_from_dense(
    matrix_a: npt.NDArray[np.int_], matrix_b: npt.NDArray[np.int_]
) -> npt.NDArray[np.float_]:
    """Calculate the Tanimoto similarity for two dense matrices.

    In returned similarity matrix each row corresponds to a row from matrix_a, whereas each colum corresponds to a
    row from matrix_b.

    Parameters
    ----------
    matrix_a: npt.NDArray[np.int_]
        First feature matrix.
    matrix_b: npt.NDArray[np.int_]
        Second feature matrix.

    Returns
    -------
    npt.NDArray[np.float_]
        Similarity matrix.
    """
    intersection = matrix_a.dot(matrix_b.transpose())
    norm_1 = np.multiply(matrix_a, matrix_a).sum(axis=1)
    norm_2 = np.multiply(matrix_b, matrix_b).sum(axis=1)
    norm_1 = np.atleast_2d(norm_1)
    norm_2 = np.atleast_2d(norm_2)
    union = norm_1.T + norm_2 - intersection
    similarity: npt.NDArray[np.float_] = intersection / union
    return similarity


def tanimoto_from_sparse(
    matrix_a: sparse.csr_matrix, matrix_b: sparse.csr_matrix
) -> npt.NDArray[np.float_]:
    """Calculate the Tanimoto similarity for two sparse matrices.

    In returned similarity matrix each row corresponds to a row from matrix_a, whereas each colum corresponds to a
    row from matrix_b.

    Parameters
    ----------
    matrix_a: sparse.csr_matrix
        First feature matrix.
    matrix_b: sparse.csr_matrix
        Second feature matrix.

    Returns
    -------
    npt.NDArray[np.float_]
        Similarity matrix.
    """
    intersection = matrix_a.dot(matrix_b.transpose()).toarray()
    norm_1 = np.array(matrix_a.multiply(matrix_a).sum(axis=1))
    norm_2 = np.array(matrix_b.multiply(matrix_b).sum(axis=1))
    union = norm_1 + norm_2.T - intersection
    similarity: npt.NDArray[np.float_] = intersection / union
    return similarity
