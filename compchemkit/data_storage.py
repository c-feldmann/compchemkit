import numpy as np
from typing import *
from scipy import sparse

FeatureMatrix = Union[np.ndarray, sparse.csr.csr_matrix]


class DataSet:
    """ Object to contain paired data such das features and label. Supports adding other attributes such as groups.
    """
    def __init__(self, label: np.ndarray, feature_matrix: FeatureMatrix):

        if not isinstance(label, np.ndarray):
            label = np.array(label).reshape(-1)

        if label.shape[0] != feature_matrix.shape[0]:
            raise IndexError

        self.label = label
        self.feature_matrix = feature_matrix
        self._additional_attributes = set()

    def add_attribute(self, attribute_name, attribute_values: np.ndarray):
        if not isinstance(attribute_values, np.ndarray):
            attribute_values = np.array(attribute_values).reshape(-1)

        if attribute_values.shape[0] != self.label.shape[0]:
            raise IndexError("Size does not match!")

        self._additional_attributes.add(attribute_name)
        self.__dict__[attribute_name] = attribute_values

    @property
    def columns(self) -> dict:
        r_dict = {k: v for k, v in self.__dict__.items() if k in self._additional_attributes}
        r_dict["label"] = self.label
        r_dict["feature_matrix"] = self.feature_matrix
        return r_dict

    def __getitem__(self, idx) -> Union[dict, 'DataSet']:
        if isinstance(idx, int):
            return {col: values[idx] for col, values in self.columns.items()}

        data_slice = DataSet(self.label[idx], self.feature_matrix[idx])
        for additional_attribute in self._additional_attributes:
            data_slice.add_attribute(additional_attribute, self.__dict__[additional_attribute][idx])

        return data_slice
