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
"""Statevector Pseudo Kernel"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from .statevector_kernel import StatevectorKernel


def make_2d(array: np.ndarray, n_copies: int):
    """
    Takes a 1D numpy array and copies it n times along a second axis.
    """
    return np.repeat(array[np.newaxis, :], n_copies, axis=0)

class StatevectorPseudoKernel(StatevectorKernel):
    r"""
    An implementation of the pseudo kernel optimized for statevector simulations.
    """

    def __init__(
        self,
        *,
        feature_map: QuantumCircuit | None = None,
        num_training_parameters: int = 0,
    ) -> None:
        """
        Args:
            feature_map: Parameterized circuit to be used as the feature map. If ``None`` is given,
                :class:`~qiskit.circuit.library.ZZFeatureMap` is used with two qubits. If there's
                a mismatch in the number of qubits of the feature map and the number of features
                in the dataset, then the kernel will try to adjust the feature map to reflect the
                number of features.
        """
        super().__init__(feature_map=feature_map)
        self.num_parameters = num_training_parameters

    def evaluate(self, x_vec: np.ndarray, y_vec: np.ndarray = None) -> np.ndarray:
        # allow users to only provide features, parameters are set to 0
        if x_vec.shape[1] + self.num_parameters == self._num_features:
            return self.evaluate_batch(x_vec, y_vec)
        else:
            return super().evaluate(x_vec, y_vec)

    def evaluate_batch(
        self,
        x_vec: np.ndarray,
        y_vec: np.ndarray,
        x_parameters: np.ndarray = None,
        y_parameters: np.ndarray = None,
    ) -> np.ndarray:
        r"""
        Construct kernel matrix for given data and feature map
        If y_vec is None, self inner product is calculated.
        If using `statevector_simulator`, only build circuits for :math:`\Psi(x)|0\rangle`,
        then perform inner product classically.
        Args:
            x_vec: 1D or 2D array of datapoints, NxD, where N is the number of datapoints,
                                                            D is the feature dimension
            y_vec: 1D or 2D array of datapoints, MxD, where M is the number of datapoints,
                                                            D is the feature dimension
            x_parameters: 1D or 2D array of parameters, NxP, where N is the number of datapoints,
                                                        P is the number of trainable parameters
            y_parameters: 1D or 2D array of parameters, MxP
        Returns:
            2D matrix, NxM
        Raises:
            QiskitMachineLearningError:
                A quantum instance or backend has not been provided
            ValueError:
                x_vec and/or y_vec are not one or two dimensional arrays
                x_vec and y_vec have have incompatible dimensions
                x_vec and/or y_vec have incompatible dimension with feature map and
                    and feature map can not be modified to match.
        """
        if x_parameters is None:
            x_parameters = np.zeros((x_vec.shape[0], self.num_parameters))

        if y_parameters is None:
            y_parameters = np.zeros((y_vec.shape[0], self.num_parameters))

        if len(x_vec.shape) == 1:
            x_vec = x_vec.reshape(1, -1)

        if len(y_vec.shape) == 1:
            y_vec = y_vec.reshape(1, -1)

        if len(x_parameters.shape) == 1:
            x_parameters = make_2d(x_parameters, x_vec.shape[0])

        if len(y_parameters.shape) == 1:
            y_parameters = make_2d(y_parameters, y_vec.shape[0])

        if x_vec.shape[0] != x_parameters.shape[0]:
            if x_parameters.shape[0] == 1:
                x_parameters = make_2d(x_parameters, x_vec.shape[0])
            else:
                raise ValueError(
                    f"Number of x data points ({x_vec.shape[0]}) does not coincide"
                    f"with number of parameter vectors {x_parameters.shape[0]}."
                )
        if y_vec.shape[0] != y_parameters.shape[0]:
            if y_parameters.shape[0] == 1:
                x_parameters = make_2d(y_parameters, y_vec.shape[0])
            else:
                raise ValueError(
                    f"Number of y data points ({y_vec.shape[0]}) does not coincide"
                    f"with number of parameter vectors {y_parameters.shape[0]}."
                )

        if x_parameters.shape[1] != self.num_parameters:
            raise ValueError(
                f"Number of parameters provided ({x_parameters.shape[0]}) does not"
                f"coincide with the number provided in the feature map ({self.num_parameters})."
            )

        if y_parameters.shape[1] != self.num_parameters:
            raise ValueError(
                f"Number of parameters provided ({y_parameters.shape[0]}) does not coincide"
                f"with the number provided in the feature map ({self.num_parameters})."
            )

        return self.evaluate(np.hstack((x_vec, x_parameters)), np.hstack((y_vec, y_parameters)))



