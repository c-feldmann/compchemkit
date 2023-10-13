from typing import Union

import numpy.typing as npt
import numpy as np
from scipy import sparse

RNGATuple = tuple[float, float, float, float]

NPNumberArray = Union[npt.NDArray[np.int_], npt.NDArray[np.float_]]
FeatureMatrix = Union[NPNumberArray, sparse.csr.csr_matrix]
