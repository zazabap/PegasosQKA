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
"""Statevector Quantum Kernel"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.kernels.base_kernel import BaseKernel


class StatevectorKernel(BaseKernel):
    r"""
    An implementation of the quantum kernel optimized for statevector simulations.
    """

    def __init__(
        self,
        *,
        feature_map: QuantumCircuit | None = None,
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
        self._statevector_cache = {}

    def evaluate(self, x_vec: np.ndarray, y_vec: np.ndarray | None = None) -> np.ndarray:

        x_vec, y_vec = self._validate_input(x_vec, y_vec)

        if y_vec is None:
            y_vec = x_vec

        kernel_shape = (x_vec.shape[0], y_vec.shape[0])

        x_svs = [self._get_sv(x) for x in x_vec]
        y_svs = [self._get_sv(y) for y in y_vec]

        kernel_matrix = np.zeros(kernel_shape)
        for i, x in enumerate(x_svs):
            for j, y in enumerate(y_svs):
                kernel_matrix[i, j] = np.abs(np.conj(x) @ y)**2

        return kernel_matrix

    def _get_sv(self, param_values):
        param_values = tuple(param_values)
        sv = self._statevector_cache.get(param_values, None)

        if sv is None:
            qc = self._feature_map.bind_parameters(param_values)
            sv = Statevector(qc).data
            self._statevector_cache[param_values] = sv

        return sv


