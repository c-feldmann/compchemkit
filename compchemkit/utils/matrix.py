from typing import Iterable
from scipy.sparse import csr_matrix


def generate_matrix_from_item_list(
    item_list: Iterable[dict[int, int]], n_cols: int
) -> csr_matrix:
    """Transform an iterable of dicts to a sparse matrix.

    Each dict encodes a row: Key: position, Value: value

    Parameters
    ----------
    item_list: Iterable[dict[int, int]],
        Each dict in iterable encodes a row: Key: position, Value: value
    n_cols:
        Number of columns for returned matrix.

    Returns
    -------
    csr_matrix:
        sparse matrix constructed from input.
    """
    data: list[int] = []
    rows: list[int] = []
    cols: list[int] = []
    n_col = 0
    for i, mol_fp in enumerate(item_list):
        feature_list, count_list = zip(*mol_fp.items())
        data.extend(count_list)
        rows.extend(feature_list)
        cols.extend([i] * len(feature_list))
        n_col += 1
    return csr_matrix((data, (cols, rows)), shape=(n_col, n_cols))
