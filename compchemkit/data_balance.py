"""Functions used for balancing data."""

from typing import Any, Literal, TypeVar

import numpy as np
import numpy.typing as npt
from numpy.random import default_rng

from compchemkit.data_storage import DataSet

__all__ = ["undersample_dataset", "oversample_dataset"]

_T = TypeVar("_T")


def determine_sample_size(
    group_arr: npt.ArrayLike,
    ratios: dict[_T, float] | None,
    method: Literal["oversample", "undersample"],
) -> dict[_T, int]:
    """Determine the number of items to sample.

    Parameters
    ----------
    group_arr: npt.ArrayLike
        Array containing the groups which will be balanced.
    ratios: dict[_T, float] | None.
        Ratios of groups required after sampling.
    method: Literal["oversample", "undersample"]
        Defines if the group sample size is calculated for over- or undersampling.

    Returns
    -------
    dict[_T, int]:
        Number of samples (value) per group (key).
    """
    unique_groups, count = np.unique(group_arr, return_counts=True)
    group_count = dict(zip(unique_groups, count))

    if not ratios:
        ratios = {group_l: 1 for group_l in unique_groups}

    group_count_scaled = {
        group: group_count[group] / ratios[group] for group in unique_groups
    }
    if method == "oversample":
        group_factor = max(gcs for gcs in group_count_scaled.values())
    elif method == "undersample":
        group_factor = min(gcs for gcs in group_count_scaled.values())
    else:
        raise ValueError(f"Unknown method: {method}")
    group_sample_size = {
        group: int(np.floor(ratios[group] * group_factor)) for group in unique_groups
    }
    return group_sample_size


def undersample_dataset(
    dataset: DataSet,
    column: str = "label",
    ratios: dict[Any, float] | None = None,
    seed: int | None = None,
    only_index: bool = False,
) -> DataSet | npt.NDArray[np.int64]:
    """Undersample the dataset by the column given by label.

    Parameters
    ----------
    dataset: DataSet
        Dataset to balance by undersampling.
    column: str
        Values to balance the dataset for.
    ratios: dict[Any, float] | None, optional
        Specifies specific ratios for `column` labels for balance.
    seed: int | None, optional
        Random seed to use for random sampling.
    only_index: bool, default: True
        If true, only the indices of the sampled rows are returned.
    Returns
    -------
    DataSet | npt.NDArray[np.int64]
        The subsampled Dataset, or if `only_index=True` the respective row indices.
    """
    group_sample_size = determine_sample_size(
        dataset.attribute_dict[column], ratios, "undersample"
    )

    group_indices = {}
    for group in group_sample_size:
        group_indices[group] = np.where(dataset.attribute_dict[column] == group)[0]

    random_gen = default_rng(seed)
    sampled_indices: set[int] = set()
    for group, n_sample in group_sample_size.items():
        index_pool = group_indices[group]
        sampled_indices.update(random_gen.choice(index_pool, n_sample, replace=False))
    sampled_indices_array = np.array(sorted(sampled_indices))
    if only_index:
        return sampled_indices_array
    r_dataset = dataset[sampled_indices_array]
    if isinstance(r_dataset, dict):
        raise TypeError
    return r_dataset


def oversample_dataset(
    dataset: DataSet,
    column: str = "label",
    ratios: dict[Any, float] | None = None,
    seed: int | None = None,
    only_index: bool = False,
) -> DataSet | npt.NDArray[np.int_]:
    """Oversample the dataset by the column given by label.

    Parameters
    ----------
    dataset: DataSet
        Dataset to balance by oversamplesampling.
    column: str
        Values to balance the dataset for.
    ratios: dict[Any, float] | None, optional
        Specifies specific ratios for `column` labels for balance.
    seed: int | None, optional
        Random seed to use for random sampling.
    only_index: bool, default: True
        If true, only the indices of the sampled rows are returned.
    Returns
    -------
    DataSet | npt.NDArray[np.int64]
        The oversamplesampled Dataset, or if `only_index=True` the respective row indices.
    """
    group_sample_size = determine_sample_size(
        dataset.attribute_dict[column], ratios, "oversample"
    )

    group_indices = {}
    for group in group_sample_size:
        group_indices[group] = np.where(dataset.attribute_dict[column] == group)[0]

    random_gen = default_rng(seed)
    sampled_indices: list[int] = []
    for group, n_sample in group_sample_size.items():
        index_pool = group_indices[group]
        sampled_indices.extend(index_pool)
        n_sample -= len(index_pool)
        if n_sample == 0:
            continue
        sampled_indices.extend(random_gen.choice(index_pool, n_sample, replace=True))
    sampled_index_array = np.array(sorted(sampled_indices))
    if only_index:
        return sampled_index_array
    r_dataset = dataset[sampled_index_array]
    if isinstance(r_dataset, dict):
        raise TypeError
    return r_dataset
