from typing import *

import numpy as np
from scipy import sparse
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GroupShuffleSplit
import abc

FeatureMatrix = Union[np.ndarray, sparse.csr.csr_matrix]


def undersample(x: FeatureMatrix,
                y: np.ndarray
                ) -> Tuple[FeatureMatrix, np.ndarray]:
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"First Dimension of X ({x.shape[0]}) and y ({y.shape[0]}) are not equal!")

    label, count = np.unique(y, return_counts=True)
    if len(label) != 2:
        ValueError("This function is designed for only two labels!")

    minority_count = min(count)

    undersample_features_per_class = []
    # Split feature matrix by class and remove all extra data
    for label_sym in label:
        features_of_class = x[np.where(y == label_sym)]
        assert list(set(y[np.where(y == label_sym)]))[0] == label_sym

        undersample_features = features_of_class[: minority_count]
        assert undersample_features.shape == (minority_count, x.shape[1])
        undersample_features_per_class.append(undersample_features)

    # Merge split and undersampled features back to one matrix
    if isinstance(x, np.ndarray):
        undersample_features_per_class = np.vstack(undersample_features_per_class)
    elif isinstance(x, sparse.csr.csr_matrix):
        undersample_features_per_class = sparse.vstack(undersample_features_per_class)
    else:
        raise TypeError(f"Unexpected Type: {type(x)}")

    return undersample_features_per_class, np.hstack([[ls] * minority_count for ls in label])


def balance_groups(x, y, group, seed=7791):
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"First Dimension of X ({x.shape[0]}) and y ({y.shape[0]}) are not equal!")
    if x.shape[0] != group.shape[0]:
        raise ValueError(f"First Dimension of X ({x.shape[0]}) and group ({group.shape[0]}) are not equal!")

    np.random.seed(seed)
    unique_group_label = np.unique(group)

    label, count = np.unique(y, return_counts=True)
    if len(label) != 2:
        ValueError("This function is designed for only two labels!")

    retained_index = list()
    for group_number in unique_group_label:
        group_idx = np.where(group == group_number)[0]

        group_label = y[group_idx]
        class0 = np.where(group_label == label[0])[0]
        class1 = np.where(group_label == label[1])[0]
        class0_selection = np.random.choice(class0, 1)[0]
        class1_selection = np.random.choice(class1, 1)[0]

        class0_selection_idx = group_idx[class0_selection]
        class1_selection_idx = group_idx[class1_selection]
        assert group[class0_selection_idx] == group[class1_selection_idx]
        assert y[class0_selection_idx] != y[class1_selection_idx]
        retained_index.append(class0_selection_idx)
        retained_index.append(class1_selection_idx)
    retained_index = np.array(retained_index)
    selected_y_values = y[retained_index]
    _, group_counts = np.unique(group[retained_index], return_counts=True)

    label, count = np.unique(selected_y_values, return_counts=True)
    assert len(label) == 2
    assert count[0] == count[1], print(count)
    assert all([c == 2 for c in group_counts])
    return retained_index


class GroupSplit(abc.ABC):
    def __init__(self, kwargs):
        self._split_kwargs = kwargs
        self._base_split = None
        self.n_splits = None

    def __len__(self):
        return self.n_splits

    def split(self, x, y, g):
        data_splitter = self._base_split(**self._split_kwargs)

        for train_idx, test_idx in data_splitter.split(x, y, g):
            x_train, y_train, g_train = x[train_idx], y[train_idx], g[train_idx]
            bal_train_idx = balance_groups(x_train, y_train, g_train)

            x_test, y_test, g_test = x[test_idx], y[test_idx], g[test_idx]
            bal_test_idx = balance_groups(x_test, y_test, g_test)

            assert len(set(g_train[bal_train_idx]) & set(g_test[bal_test_idx])) == 0
            assert len(set(train_idx[bal_train_idx]) & set(test_idx[bal_test_idx])) == 0
            yield train_idx[bal_train_idx], test_idx[bal_test_idx]


class BalancedGroupShuffleSplit(GroupSplit):
    def __init__(self, n_splits, test_size, random_state=None):
        super().__init__({"n_splits": n_splits,
                          "test_size": test_size,
                          "random_state": random_state,
                          })
        self.n_splits = n_splits
        self._base_split = GroupShuffleSplit


class BalancedGroupKFold(GroupSplit):
    def __init__(self, n_splits):
        self.n_splits = n_splits
        super().__init__({"n_splits": n_splits})
        self._base_split = GroupKFold

