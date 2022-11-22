from __future__ import annotations
from typing import Any, Dict, List, Optional, Union, Set

import numpy as np
import numpy.typing as npt

from compchemkit.ccktypes import FeatureMatrix


class DataSet:
    """Object to contain paired data such das features and label. Supports adding other attributes such as groups."""

    def __init__(
        self,
        feature_matrix: FeatureMatrix,
        label: Optional[Union[npt.NDArray[np.int_], npt.NDArray[np.float_]]] = None,
    ):
        self.feature_matrix = feature_matrix
        self._additional_attributes: Set[str] = set()
        if label is not None:
            self.add_attribute("label", label)

    def add_attribute(self, attribute_name: str, attribute_values: npt.NDArray[Any]) -> None:
        if attribute_values.shape[0] != self.feature_matrix.shape[0]:
            raise IndexError("Size does not match!")
        self._additional_attributes.add(attribute_name)
        setattr(self, attribute_name, attribute_values)

    @property
    def columns(self) -> List[str]:
        return sorted(self._additional_attributes | {"feature_matrix"})

    @property
    def attribute_dict(self) -> Dict[str, Any]:
        return {col: getattr(self, col) for col in self.columns}

    def __getitem__(
        self, idx: Union[slice, npt.NDArray[np.int_]]
    ) -> Union[Dict[str, Any], DataSet]:
        if isinstance(idx, int):
            return {col: self.__dict__[col] for col in self.columns}

        data_slice = DataSet(self.feature_matrix[idx])
        for additional_attribute in self._additional_attributes:
            data_slice.add_attribute(additional_attribute, getattr(self, additional_attribute)[idx])
        return data_slice
