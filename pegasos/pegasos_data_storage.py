# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Data storage for PegasosQSVC and PegasosQKA"""

from typing import Optional
import numpy as np

from qiskit.utils import algorithm_globals


class PegasosDataStorage:
    """
    This class implements the dynamic data set structure used in PegasosQSVC for non-stationary
    data.
    """

    def __init__(self) -> None:
        """
        Args:
            X: Train features. For a callable kernel (an instance of ``QuantumKernel``) the shape
               should be ``(n_samples, n_features)``, for a precomputed kernel the shape should be
               ``(n_samples, n_samples)``.
            y: shape (n_samples), train labels . Must not contain more than two unique labels.
        """
        self.length = 0
        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self._data_dim: Optional[int] = None

    def shuffle(self):
        shuffled_indices = algorithm_globals.random.choice(
            np.arange(self.length), size=self.length, replace=False
        )
        self.X = self.X[shuffled_indices]
        self.y = self.y[shuffled_indices]

    def store_new(self, x: np.ndarray, y: int):
        """
        Saves a new pair of features `x` and label `y` to the storage.
        """
        if self._data_dim is None:
            self._data_dim = len(x)
            self.X = np.array(x)
            self.y = np.array([y])
        else:
            if len(x) != self._data_dim:
                raise ValueError("The feature dimension cannot change over time.")
            self.X = np.vstack([self.X, x])
            self.y = np.append(self.y, y)
        self.length += 1
