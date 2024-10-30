"""Classes used for storing sparse and dense data."""

from __future__ import annotations

from typing import Any, overload

import numpy as np
import numpy.typing as npt

from compchemkit.utils.custom_types import FeatureMatrix


class DataSet:
    """Object to contain paired data such das features and label. Supports adding other attributes such as groups."""

    def __init__(
        self,
        feature_matrix: FeatureMatrix,
        label: npt.NDArray[np.int64] | npt.NDArray[np.float64] | None = None,
    ):
        """Initialize the DataSet object.

        Parameters
        ----------
        feature_matrix: FeatureMatrix
            Sparse or dense array containing the features (columns) for each datapoint (rows).
        label: npt.NDArray[np.int64] | npt.NDArray[np.float64] | None, optional
            Label or target value for each datapoint.
        """
        self.feature_matrix = feature_matrix
        self._additional_attributes: set[str] = set()
        if label is not None:
            self.add_attribute("label", label)

    def add_attribute(
        self, attribute_name: str, attribute_values: npt.NDArray[Any]
    ) -> None:
        """Add additional attributes for each datapoint.

        Parameters
        ----------
        attribute_name: str
            Name of the attribute.
        attribute_values: npt.NDArray[Any]
            Attribute values for each datapoint.
        """
        if attribute_values.shape[0] != self.feature_matrix.shape[0]:
            raise IndexError("Size does not match!")
        self._additional_attributes.add(attribute_name)
        setattr(self, attribute_name, attribute_values)

    @property
    def columns(self) -> list[str]:
        """Returns the columns of the Dataset."""
        return sorted(self._additional_attributes | {"feature_matrix"})

    @property
    def attribute_dict(self) -> dict[str, Any]:
        """Returns a dict of columns and the corresponding datapoint values."""
        return {col: getattr(self, col) for col in self.columns}

    @overload
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return a subset of the Dataset.

        Parameters
        ----------
        idx: int
            Index of the datapoint to return.

        Returns
        -------
        dict[str, Any]
            Dict with columns as key and the datapoint attributes as corresponding values.
        """

    @overload
    def __getitem__(self, idx: slice | npt.NDArray[np.int64]) -> DataSet:
        """Return a subset of the Dataset.

        Parameters
        ----------
        idx: slice | npt.NDArray[np.int64]
            Indices of the datapoints to return.

        Returns
        -------
        DataSet
            Subset of the Dataset with datapoints corresponding to the given indices.
        """

    def __getitem__(
        self, idx: int | slice | npt.NDArray[np.int64]
    ) -> dict[str, Any] | DataSet:
        """Return a subset of the Dataset.

        Parameters
        ----------
        idx: slice | npt.NDArray[np.int64]
            Index of datapoints to

        Returns
        -------
        dict[str, Any] | DataSet
            Dict with columns as key and the datapoint attributes as corresponding values.
            Or subset of the Dataset with datapoints corresponding to the given indices.
        """
        if isinstance(idx, int):
            return {col: self.__dict__[col] for col in self.columns}

        data_slice = DataSet(self.feature_matrix[idx])
        for additional_attribute in self._additional_attributes:
            data_slice.add_attribute(
                additional_attribute, getattr(self, additional_attribute)[idx]
            )
        return data_slice
