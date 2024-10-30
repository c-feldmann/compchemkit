"""Define Types throughout the package."""

import numpy as np
import numpy.typing as npt
from scipy import sparse

RNGATuple = tuple[float, float, float, float]

NPNumberArray = npt.NDArray[np.int_] | npt.NDArray[np.float64]
FeatureMatrix = NPNumberArray | sparse.csr.csr_matrix
