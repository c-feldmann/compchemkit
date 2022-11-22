import numpy as np
import numpy.typing as npt
from .data_storage import DataSet
from typing import Any, Dict, List, Optional, Set, Union
from numpy.random import default_rng


def undersample_dataset(
    dataset: DataSet,
    column: str = "label",
    ratios: Optional[Dict[Any, float]] = None,
    seed: Optional[int] = None,
    only_index: bool = False,
) -> Union[DataSet, npt.NDArray[np.int_]]:
    unique_groups, count = np.unique(dataset.attribute_dict[column], return_counts=True)
    group_count = dict(zip(unique_groups, count))

    group_indices = dict()
    for group in unique_groups:
        group_indices[group] = np.where(dataset.attribute_dict[column] == group)[0]

    if not ratios:
        ratios = {group_l: 1 for group_l in unique_groups}

    group_count_scaled = {group: group_count[group] / ratios[group] for group in unique_groups}
    limiting_group_count = min([gcs for gcs in group_count_scaled.values()])
    group_sample_size = {
        group: int(np.floor(ratios[group] * limiting_group_count)) for group in unique_groups
    }

    random_gen = default_rng(seed)
    sampled_indices: Set[int] = set()
    for group in unique_groups:
        n_sample = group_sample_size[group]
        index_pool = group_indices[group]
        sampled_indices.update(random_gen.choice(index_pool, n_sample, replace=False))
    sampled_indice_array = np.array(sorted(sampled_indices))
    if only_index:
        return sampled_indice_array
    else:
        r_dataset = dataset[sampled_indice_array]
        if isinstance(r_dataset, Dict):
            raise TypeError
        return r_dataset


def oversample_dataset(
    dataset: DataSet,
    column: str = "label",
    ratios: Optional[Dict[Any, float]] = None,
    seed: Optional[int] = None,
    only_index: bool = False,
) -> Union[DataSet, npt.NDArray[np.int_]]:

    unique_groups, count = np.unique(dataset.attribute_dict[column], return_counts=True)
    group_count = dict(zip(unique_groups, count))

    group_indices = dict()
    for group in unique_groups:
        group_indices[group] = np.where(dataset.attribute_dict[column] == group)[0]

    if not ratios:
        ratios = {group_l: 1 for group_l in unique_groups}

    group_count_scaled = {group: group_count[group] / ratios[group] for group in unique_groups}
    required_group_count = max([gcs for gcs in group_count_scaled.values()])
    group_sample_size = {
        group: int(np.floor(ratios[group] * required_group_count)) for group in unique_groups
    }

    random_gen = default_rng(seed)
    sampled_indices: List[int] = []
    for group in unique_groups:
        n_sample = group_sample_size[group]
        index_pool = group_indices[group]

        sampled_indices.extend(index_pool)
        n_sample -= len(index_pool)
        if n_sample == 0:
            continue
        sampled_indices.extend(random_gen.choice(index_pool, n_sample, replace=True))
    sampled_index_array = np.array(sorted(sampled_indices))
    if only_index:
        return sampled_index_array
    else:
        r_dataset = dataset[sampled_index_array]
        if isinstance(r_dataset, Dict):
            raise TypeError
        return r_dataset
