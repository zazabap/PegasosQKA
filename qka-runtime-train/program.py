from mimetypes import init
from time import time
from typing import Optional
from unittest import result

import numpy as np

"""
Fidelity
"""
from qiskit.circuit import QuantumCircuit, ParameterVector, Parameter
from qiskit.providers.aer.aerprovider import AerProvider


import abc

from qiskit.circuit.gate import Gate

try:
    from libs_qrem import LeastNormFilter

    _QREM = True
except ImportError:
    _QREM = False


import qiskit
from qiskit.providers.backend import Backend
from qiskit.compiler import transpile

from qiskit.circuit.library import XGate
from qiskit.transpiler import PassManager, InstructionDurations
from qiskit.transpiler.passes import ALAPSchedule, DynamicalDecoupling

from typing import List, Union, Optional, Dict


class Fidelity:
    """Class that outputs the overlap between two quantum states."""

    def __init__(
        self,
        left_circ: QuantumCircuit,
        right_circ: QuantumCircuit,
        backend: Optional[Backend] = None,
        initial_layout: Optional[List[int]] = None,
        shots: int = 1024,
        pauli_twirling: bool = False,
        num_twirls: int = 10,
        dynamical_decoupling: bool = False,
        dd_sequence: Optional[List[Gate]] = None,
        measurement_error_mitigation=None,
        calibration_shots: int = 1024,
        insert_barrier: bool = True,
    ) -> None:
        """
        Args:
            backend: The backend used to run the circuits. Defautls to qasm_simulator with 1024 shots.
            initial_layout: Construct a Layout from a bijective dictionary, mapping virtual qubits to physical qubits.
            shots: The default number of evaluations with the backend.
            pauli_twirling: If True, Pauli twirling will be applied.
            dynamical_decoupling: When "dd_sequence" is not set to None, dynamical decoupling is applied with the gate sequence "dd_sequence".
            measurement_error_mitigation: Given the inputs "MThree", "TEM" or "SEM", the corresponding measurement error mitigation methods are applied.

        """
        self._left_circ = left_circ
        self._right_circ = right_circ
        if backend is None:
            self._backend = qiskit.Aer.get_backend("qasm_simulator")
        else:
            self._backend = backend

        # if isinstance(backend.provider(), AerProvider):
        #     self._is_aer_backend = True
        # else:
        #     self._is_aer_backend = False

        self._is_aer_backend = False

        self._initial_layout = initial_layout
        self._shots = shots
        self._pauli_twirling = pauli_twirling
        self._num_twirls = num_twirls
        self._dynamical_decoupling = dynamical_decoupling
        if dd_sequence is None:
            self._dd_sequence = [XGate()] * 2
        else:
            self._dd_sequence = dd_sequence
        self._measurement_error_mitigation = measurement_error_mitigation
        self._cal_shots = calibration_shots
        self._insert_barrier = insert_barrier

        a = ParameterVector("a", left_circ.num_parameters)
        b = ParameterVector("b", right_circ.num_parameters)
        self._left_parameters = a
        self._right_parameters = b

        self._overlap = left_circ.assign_parameters(a)
        if insert_barrier:
            self._overlap.barrier()
        self._overlap.compose(right_circ.assign_parameters(b).inverse(), inplace=True)
        self._overlap.measure_all()

    def compute(
        self,
        left_parameters: Union[np.ndarray, List[np.ndarray]],
        right_parameters: Union[np.ndarray, List[np.ndarray]],
    ) -> Union[float, List[float]]:
        """Compute the overlap of two quantum states encoded in a parametrized quantum circuit ('left' or 'right') by the 'values_left' and 'values_right'

        Args:
            left: Parametrized circuit that characterizes a quantum state.
            right: Parametrized circuit that characterizes a quantum state.
            values_left: Parameter vector that when binded to the parametrizable circuit left defines a qantum state.
            values_right: Parameter vector that when binded to the parametrizable circuit right defines a qantum state.

        Returns:
            The overlap of two quantum states defined by two parametrized circuits.
        """

        # Ensure values are given as 2D-arrays
        left_parameters = np.atleast_2d(left_parameters)
        right_parameters = np.atleast_2d(right_parameters)

        # Structure of value dictionaries differ for aer and ibmq backends
        if self._is_aer_backend:
            value_dicts = self._get_aer_dictionaries(left_parameters, right_parameters)
        else:
            value_dicts = self._get_ibmq_dictionaries(left_parameters, right_parameters)

        circuits = [self._overlap]

        return measure_overlap(
            self._backend,
            self._initial_layout,
            self._shots,
            circuits,
            value_dicts,
            self._dynamical_decoupling,
            self._dd_sequence,
            self._measurement_error_mitigation,
        )

    def _get_aer_dictionaries(
        self,
        left_parameters: np.array,
        right_parameters: np.array,
    ) -> List[Dict[Parameter, List[float]]]:
        """
        Returns parameter dictionary of the form
        value_dicts = [
            {
                param1: [value1_1, value1_2, ..],
                param2: [value2_2, value2_2, ..],
                ...
            }
        ]
        """
        value_dict = {
            parameter: left_parameters[:, i].tolist()
            for i, parameter in enumerate(self._left_parameters)
        }
        value_dict.update(
            {
                parameter: right_parameters[:, i].tolist()
                for i, parameter in enumerate(self._right_parameters)
            }
        )
        return [value_dict]

    def _get_ibmq_dictionaries(
        self,
        left_parameters: np.array,
        right_parameters: np.array,
    ) -> List[Dict[Parameter, float]]:
        """
        Returns parameter dictionary of the form
        value_dict = [
            { param1: value1_1, param2: value2_1, ... },
            { param1: value1_2, param2: value2_2, ... },
            ...
        ]
        """
        value_dicts = [
            dict(
                {
                    parameter: left_parameters[j, i]
                    for i, parameter in enumerate(self._left_parameters)
                }.items()
                | {
                    parameter: right_parameters[j, i]
                    for i, parameter in enumerate(self._right_parameters)
                }.items()
            )
            for j in range(len(left_parameters))
        ]
        return value_dicts


def run_transpile(
    backend: Backend,
    circuits: List[QuantumCircuit],
    dynamical_decoupling: bool,
    dd_sequence: List[Gate],
    initial_layout: Optional[list] = None,
) -> QuantumCircuit:
    """Returns transpiled quantum circuit for a specific dynamical decoupling sequence."""

    transpiled_circ = transpile(
        circuits, backend=backend, initial_layout=initial_layout
    )

    if dynamical_decoupling:
        durations = InstructionDurations.from_backend(backend)
        pass_manager = PassManager(
            [ALAPSchedule(durations), DynamicalDecoupling(durations, dd_sequence)]
        )

        return pass_manager.run(transpiled_circ)

    return transpiled_circ


def measure_overlap(
    backend: Backend,
    initial_layout: List[int],
    shots: int,
    circuits: List[QuantumCircuit],
    values_dicts: List[Dict[Parameter, List[float]]],
    dynamical_decoupling: bool,
    dd_sequence: List[Gate],
    measurement_error_mitigation=None,
) -> float:
    """Outputs the measurement probability of the all-zero state."""

    transpiled_circ = run_transpile(
        backend=backend,
        circuits=circuits,
        dynamical_decoupling=dynamical_decoupling,
        dd_sequence=dd_sequence,
        initial_layout=initial_layout,
    )

    job_result = backend.run(
        circuits=transpiled_circ, parameter_binds=values_dicts, shots=shots
    ).result()

    if measurement_error_mitigation is not None:
        result = measurement_error_mitigation.apply_correction(
            backend, transpiled_circ, job_result
        )

        zero_prob = np.array(
            [counts.get("0" * circuits[0].num_qubits, 0) for counts in result]
        )

    else:
        result_counts = job_result.get_counts()
        if isinstance(result_counts, list):
            zero_prob = np.array(
                [counts.int_outcomes().get(0, 0) / shots for counts in result_counts]
            )
        else:
            zero_prob = np.array([result_counts.int_outcomes().get(0, 0)]) / shots

    zero_prob = zero_prob.reshape((len(circuits), -1))
    zero_prob = np.mean(zero_prob, axis=0)

    return zero_prob, zero_prob / np.sqrt(shots)


class FidelityFactory:
    def __init__(
        self,
        backend: Backend,
        shots: int,
        measurement_error_mitigation=None,
        initial_layout: Optional[List[int]] = None,
        pauli_twirling: bool = False,
        num_twirls: int = 10,
        dynamical_decoupling: bool = False,
        dd_sequence: Optional[list] = None,
        calibration_shots: int = 1024,
        insert_barrier: bool = True,
    ) -> None:
        self.backend = backend
        self.shots = shots
        self.measurement_error_mitigation = measurement_error_mitigation
        self.initial_layout = initial_layout
        self.pauli_twriling = pauli_twirling
        self.num_twirls = num_twirls
        self.dynamical_decouplng = dynamical_decoupling
        self.dd_sequence = dd_sequence
        self.calibration_shots = calibration_shots
        self.insert_barrier = insert_barrier

    def __call__(
        self, left_circuit: QuantumCircuit, right_circuit: QuantumCircuit
    ) -> Fidelity:
        return Fidelity(
            left_circ=left_circuit,
            right_circ=right_circuit,
            backend=self.backend,
            initial_layout=self.initial_layout,
            shots=self.shots,
            pauli_twirling=self.pauli_twriling,
            num_twirls=self.num_twirls,
            dynamical_decoupling=self.dynamical_decouplng,
            dd_sequence=self.dd_sequence,
            measurement_error_mitigation=self.measurement_error_mitigation,
            calibration_shots=self.calibration_shots,
            insert_barrier=self.insert_barrier,
        )


"""
Kernels
"""


def make_2D(array: np.ndarray, n_copies: int):
    """
    Takes a 1D numpy array and copies n times it along a second axis.
    """
    return np.repeat(array[np.newaxis, :], n_copies, axis=0)


from abc import abstractmethod

from typing import Tuple


class BaseKernel:
    """
    Abstract class providing the interface for the quantum kernel classes.
    """

    def __init__(self, enforce_psd: bool = True) -> None:
        """
        Args:
            enforce_psd: Project to closest positive semidefinite matrix if x = y.
                Only enforced when not using the state vector simulator. Default True.
            sampler: A qiskit runtime primitives sampler instance
        """
        self._num_features: int = 0
        self._enforce_psd = enforce_psd

    @abstractmethod
    def evaluate(self, x_vec: np.ndarray, y_vec: np.ndarray = None) -> np.ndarray:
        r"""
        Construct kernel matrix for given data

        If y_vec is None, self inner product is calculated.

        Args:
            x_vec: 1D or 2D array of datapoints, NxD, where N is the number of datapoints,
                                                            D is the feature dimension
            y_vec: 1D or 2D array of datapoints, MxD, where M is the number of datapoints,
                                                            D is the feature dimension

        Returns:
            2D matrix, NxM

        Raises:
            QiskitMachineLearningError:
                - A quantum instance or backend has not been provided
            ValueError:
                - x_vec and/or y_vec are not one or two dimensional arrays
                - x_vec and y_vec have have incompatible dimensions
                - x_vec and/or y_vec have incompatible dimension with feature map and
                    and feature map can not be modified to match.
        """
        raise NotImplementedError()

    def _check_and_reshape(
        self, x_vec: np.ndarray, y_vec: np.ndarray = None
    ) -> Tuple[np.ndarray]:
        r"""
        Performs checks on the dimensions of the input data x_vec and y_vec.
        Reshapes the arrays so that `x_vec.shape = (N,D)` and `y_vec.shape = (M,D)`.
        """
        if not isinstance(x_vec, np.ndarray):
            x_vec = np.asarray(x_vec)

        if y_vec is not None and not isinstance(y_vec, np.ndarray):
            y_vec = np.asarray(y_vec)

        if x_vec.ndim > 2:
            raise ValueError("x_vec must be a 1D or 2D array")

        if x_vec.ndim == 1:
            x_vec = x_vec.reshape(1, -1)

        if y_vec is not None and y_vec.ndim > 2:
            raise ValueError("y_vec must be a 1D or 2D array")

        if y_vec is not None and y_vec.ndim == 1:
            y_vec = y_vec.reshape(1, -1)

        if y_vec is not None and y_vec.shape[1] != x_vec.shape[1]:
            raise ValueError(
                "x_vec and y_vec have incompatible dimensions.\n"
                f"x_vec has {x_vec.shape[1]} dimensions, but y_vec has {y_vec.shape[1]}."
            )

        if x_vec.shape[1] != self._num_features:
            raise ValueError(
                "x_vec and class feature map have incompatible dimensions.\n"
                f"x_vec has {x_vec.shape[1]} dimensions, "
                f"but feature map has {self._num_features}."
            )

        if y_vec is None:
            y_vec = x_vec

        return x_vec, y_vec

    def _make_psd(self, kernel_matrix: np.ndarray) -> np.ndarray:
        r"""
        Find the closest positive semi-definite approximation to symmetric kernel matrix.
        The (symmetric) matrix should always be positive semi-definite by construction,
        but this can be violated in case of noise, such as sampling noise, thus the
        adjustment is only done if NOT using the statevector simulation.

        Args:
            kernel_matrix: symmetric 2D array of the kernel entries
        """
        d, u = np.linalg.eig(kernel_matrix)
        return u @ np.diag(np.maximum(0, d)) @ u.transpose()


from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap
from typing import Optional

from typing import Tuple


class QuantumKernel(BaseKernel):
    """
    Overlap Kernel
    """

    def __init__(
        self,
        fidelity_factory: FidelityFactory,
        feature_map: Optional[QuantumCircuit] = None,
        enforce_psd: bool = True,
    ) -> None:
        super().__init__(enforce_psd)

        if feature_map is None:
            feature_map = ZZFeatureMap(2).decompose()

        self._num_features = feature_map.num_parameters
        self._fidelity = fidelity_factory(feature_map, feature_map)

    def evaluate(self, x_vec: np.ndarray, y_vec: np.ndarray = None) -> np.ndarray:
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

        Returns:
            2D matrix, NxM

        Raises:
            QiskitMachineLearningError:
                - A quantum instance or backend has not been provided
            ValueError:
                - x_vec and/or y_vec are not one or two dimensional arrays
                - x_vec and y_vec have have incompatible dimensions
                - x_vec and/or y_vec have incompatible dimension with feature map and
                    and feature map can not be modified to match.
        """
        x_vec, y_vec = self._check_and_reshape(x_vec, y_vec)
        is_symmetric = np.all(x_vec == y_vec)
        shape = len(x_vec), len(y_vec)

        if is_symmetric:
            left_parameters, right_parameters = self._get_symmetric_parametrization(
                x_vec
            )
            kernel_matrix = self._get_symmetric_kernel_matrix(
                left_parameters, right_parameters, shape
            )

        else:
            left_parameters, right_parameters = self._get_parametrization(x_vec, y_vec)
            kernel_matrix = self._get_kernel_matrix(
                left_parameters, right_parameters, shape
            )

        if is_symmetric and self._enforce_psd:
            kernel_matrix = self._make_psd(kernel_matrix)
        return kernel_matrix

    def _get_parametrization(
        self, x_vec: np.ndarray, y_vec: np.ndarray
    ) -> Tuple[np.ndarray]:
        """
        Combines x_vec and y_vec to get all the combinations needed to evaluate the kernel entries.
        """
        x_count = x_vec.shape[0]
        y_count = y_vec.shape[0]

        left_parameters = np.zeros((x_count * y_count, x_vec.shape[1]))
        right_parameters = np.zeros((x_count * y_count, y_vec.shape[1]))
        index = 0
        for x_i in x_vec:
            for y_j in y_vec:
                left_parameters[index, :] = x_i
                right_parameters[index, :] = y_j
                index += 1

        return left_parameters, right_parameters

    def _get_symmetric_parametrization(self, x_vec: np.ndarray) -> Tuple[np.ndarray]:
        """
        Combines two copies of x_vec to get all the combinations needed to evaluate the kernel entries.
        """
        x_count = x_vec.shape[0]

        left_parameters = np.zeros((x_count * (x_count + 1) // 2, x_vec.shape[1]))
        right_parameters = np.zeros((x_count * (x_count + 1) // 2, x_vec.shape[1]))

        index = 0
        for i, x_i in enumerate(x_vec):
            for x_j in x_vec[i:]:
                left_parameters[index, :] = x_i
                right_parameters[index, :] = x_j
                index += 1

        return left_parameters, right_parameters

    def _get_kernel_matrix(
        self, left_parameters: np.ndarray, right_parameters: np.ndarray, shape: Tuple
    ) -> np.ndarray:
        """
        Given a parametrization, this computes the symmetric kernel matrix.
        """
        kernel_entries = self._fidelity.compute(left_parameters, right_parameters)[0]
        kernel_matrix = np.zeros(shape)

        index = 0
        for i in range(shape[0]):
            for j in range(shape[1]):
                kernel_matrix[i, j] = kernel_entries[index]
                index += 1
        return kernel_matrix

    def _get_symmetric_kernel_matrix(
        self, left_parameters: np.ndarray, right_parameters: np.ndarray, shape: Tuple
    ) -> np.ndarray:
        """
        Given a set of parametrization, this computes the kernel matrix.
        """
        kernel_entries = self._fidelity.compute(left_parameters, right_parameters)[0]
        kernel_matrix = np.zeros(shape)

        index = 0
        for i in range(shape[0]):
            for j in range(i, shape[1]):
                kernel_matrix[i, j] = kernel_entries[index]
                index += 1

        kernel_matrix = (
            kernel_matrix + kernel_matrix.T - np.diag(kernel_matrix.diagonal())
        )
        return kernel_matrix


from typing import Tuple, Optional
from qiskit.circuit.library import ZZFeatureMap
from qiskit import QuantumCircuit


class PseudoKernel(QuantumKernel):
    """
    PseudoKernel
    """

    def __init__(
        self,
        fidelity_factory: FidelityFactory,
        feature_map: Optional[QuantumCircuit] = None,
        enforce_psd: bool = True,
        num_training_parameters: int = 0,
    ) -> None:
        super().__init__(fidelity_factory, feature_map, enforce_psd)

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
                - A quantum instance or backend has not been provided
            ValueError:
                - x_vec and/or y_vec are not one or two dimensional arrays
                - x_vec and y_vec have have incompatible dimensions
                - x_vec and/or y_vec have incompatible dimension with feature map and
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
            x_parameters = make_2D(x_parameters, x_vec.shape[0])

        if len(y_parameters.shape) == 1:
            y_parameters = make_2D(y_parameters, y_vec.shape[0])

        if x_vec.shape[0] != x_parameters.shape[0]:
            if x_parameters.shape[0] == 1:
                x_parameters = make_2D(x_parameters, x_vec.shape[0])
            else:
                raise ValueError(
                    f"Number of x data points ({x_vec.shape[0]}) does not coincide with number of parameter vectors {x_parameters.shape[0]}."
                )
        if y_vec.shape[0] != y_parameters.shape[0]:
            if y_parameters.shape[0] == 1:
                x_parameters = make_2D(y_parameters, y_vec.shape[0])
            else:
                raise ValueError(
                    f"Number of y data points ({y_vec.shape[0]}) does not coincide with number of parameter vectors {y_parameters.shape[0]}."
                )

        if x_parameters.shape[1] != self.num_parameters:
            raise ValueError(
                f"Number of parameters provided ({x_parameters.shape[0]}) does not coincide with the number provided in the feature map ({self.num_parameters})."
            )

        if y_parameters.shape[1] != self.num_parameters:
            raise ValueError(
                f"Number of parameters provided ({y_parameters.shape[0]}) does not coincide with the number provided in the feature map ({self.num_parameters})."
            )

        return self.evaluate(
            np.hstack((x_vec, x_parameters)), np.hstack((y_vec, y_parameters))
        )


"""
Pegasos QSVC
"""
from qiskit.circuit.library import RealAmplitudes, ZFeatureMap, ZZFeatureMap
from qiskit.utils import algorithm_globals
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import IterableDataset


class PegasosDataset(IterableDataset):
    def __init__(self) -> None:
        super().__init__()
        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None

    def from_numpy(self, X: np.ndarray, y: np.ndarray):
        if len(X.shape) != 2:
            raise ValueError("X has to be a 2D array")
        if len(y.shape) != 1 and not (len(y.shape) == 2 and y.shape[0] == 1):
            raise ValueError("y has to be a 1D array")
        y = y.flatten()
        if y.shape[0] != X.shape[0]:
            raise ValueError(
                "Number of feature vectors does not match number of labels"
            )
        for label in np.unique(y):
            if label not in [-1, 1]:
                raise ValueError("The labels have to be in {-1,1}")
        self.X = X
        self.y = y

        return self

    def __next__(self):
        index = np.random.randint(0, len(self.X))
        return self.X[index], self.y[index]

    def __iter__(self):
        return self


"""Pegasos Quantum Support Vector Classifier."""

import logging
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

from qiskit.utils import algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import QSVC
from qiskit_machine_learning.exceptions import QiskitMachineLearningError
from sklearn.svm import SVC
from torch.utils.data import DataLoader, IterableDataset

logger = logging.getLogger(__name__)


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

    def to_dict(self):
        return {"X": self.X.tolist(), "y": self.y.tolist()}

    def from_dict(self, dict):
        self.X = np.array(dict["X"])
        self.y = np.array(dict["y"])
        self._data_dim = self.X.shape[1]
        self.length = self.X.shape[0]
        return self


class PegasosDataLoading(Iterable):
    """
    This class handels the data loading for PegasosQSVC. Wrapper for the pytorch data loader.
    """

    def __init__(self, dataset: IterableDataset) -> None:
        self._data_loader = DataLoader(dataset, batch_size=1)

    def __iter__(self):
        return self

    def __next__(self):
        x, y = next(iter(self._data_loader))
        x = np.array(x)
        if x.shape[0] != 1 and len(x.shape) > 1:
            raise ValueError(
                "The feature vector `x` returned by iterating the data loader needs to have shape (1,n_dim) or (n_dim,)"
            )
        y = y[0]
        if y not in [-1, 1]:
            raise ValueError("The labels have to be in {-1,1}.")
        return x.flatten(), y

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

"""Pegasos Quantum Support Vector Classifier."""

import logging
from datetime import datetime
from typing import Callable, Dict, Iterable, List, Optional, Tuple


from qiskit.utils import algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import QSVC
from qiskit_machine_learning.exceptions import QiskitMachineLearningError
from sklearn.svm import SVC
from torch.utils.data import DataLoader, IterableDataset

logger = logging.getLogger(__name__)

# callback arguments: step number, decision value, x_step, y_step, PegasosQSVC
CALLBACK = Callable[[int, float, bool, np.ndarray, int], None]


class PegasosDataLoading(Iterable):
    """
    This class handels the data loading for PegasosQSVC. Wrapper for the pytorch data loader.
    """

    def __init__(self, dataset: IterableDataset) -> None:
        self._data_loader = DataLoader(dataset, batch_size=1)

    def __iter__(self):
        return self

    def __next__(self):
        x, y = next(iter(self._data_loader))
        x = np.array(x)
        if x.shape[0] != 1 and len(x.shape) > 1:
            raise ValueError(
                "The feature vector `x` returned by iterating the data loader needs to have shape (1,n_dim) or (n_dim,)"
            )
        y = int(y[0])
        if y not in [-1, 1]:
            raise ValueError("The labels have to be in {-1,1}.")
        return x.flatten(), y


class PegasosQSVC(SVC):
    """
    This class implements Pegasos Quantum Support Vector Classifier algorithm developed in [1]
    and includes overridden methods ``fit`` and ``predict`` from the ``SVC`` super-class. This
    implementation is adapted to work with quantum kernels.

    **Example**

    .. code-block:: python

        quantum_kernel = QuantumKernel()

        pegasos_qsvc = PegasosQSVC(quantum_kernel=quantum_kernel)
        pegasos_qsvc.fit(sample_train, label_train)
        pegasos_qsvc.predict(sample_test)

    **References**
        [1]: Shalev-Shwartz et al., Pegasos: Primal Estimated sub-GrAdient SOlver for SVM.
            `Pegasos for SVM <https://home.ttic.edu/~nati/Publications/PegasosMPB.pdf>`_

    """

    FITTED = 0
    UNFITTED = 1

    def __init__(
        self,
        quantum_kernel: Optional[QuantumKernel] = None,
        C: float = 1.0,
        precomputed: bool = False,
        seed: Optional[int] = None,
        steps_used: Optional[int] = None,
        callback: Optional[CALLBACK] = None,
    ) -> None:
        """
        Args:
            quantum_kernel: QuantumKernel to be used for classification. Has to be ``None`` when
                a precomputed kernel is used.
            C: Positive regularization parameter. The strength of the regularization is inversely
                proportional to C. Smaller ``C`` induce smaller weights which generally helps
                preventing overfitting. However, due to the nature of this algorithm, some of the
                computation steps become trivial for larger ``C``. Thus, larger ``C`` improve
                the performance of the algorithm drastically. If the data is linearly separable
                in feature space, ``C`` should be chosen to be large. If the separation is not
                perfect, ``C`` should be chosen smaller to prevent overfitting.

            precomputed: a boolean flag indicating whether a precomputed kernel is used. Set it to
                ``True`` in case of precomputed kernel.
            seed: a seed for the random number generator
            steps_used: if set to a positive integer value, only the ``steps_used`` newest steps are
                used to calculate the weights.

        Raises:
            ValueError:
                - if ``quantum_kernel`` is passed and ``precomputed`` is set to ``True``. To use
                a precomputed kernel, ``quantum_kernel`` has to be of the ``None`` type.
            TypeError:
                - if ``quantum_instance`` neither instance of ``QuantumKernel`` nor ``None``.
        """
        super().__init__(C=C)

        if precomputed:
            raise NotImplementedError(
                "PegasosQSVC is not yet implemented for a precomputed kernel."
            )
        else:
            if quantum_kernel is None:
                quantum_kernel = QuantumKernel()
            elif not isinstance(quantum_kernel, QuantumKernel):
                print("oh oh, wrong quantum kernel class.. trying anyway")
                # raise TypeError("'quantum_kernel' has to be of type None or QuantumKernel")

        self._quantum_kernel = quantum_kernel
        self._precomputed = precomputed
        if seed is not None:
            algorithm_globals.random_seed = seed

        self._steps_used: Optional[int] = None

        self._callback = callback

        if steps_used is not None:
            if not isinstance(steps_used, int) or steps_used < 1:
                raise ValueError("'steps_used' has to be a positive integer or None.")
            self._steps_used = steps_used

        # these are the parameters being fit and are needed for prediction
        self._alphas: Optional[List[int]] = None
        self._data_loader: Optional[PegasosDataLoading] = None
        self._training_data: Optional[PegasosDataStorage] = None
        self._n_samples: Optional[int] = None
        self._label_map: Optional[Dict[int, int]] = None
        self._label_pos: Optional[int] = None
        self._label_neg: Optional[int] = None

        # added to all kernel values to include an implicit bias to the hyperplane
        self._kernel_offset = 1

        # for compatibility with the base SVC class. Set as unfitted.
        self.fit_status_ = PegasosQSVC.UNFITTED

    # pylint: disable=invalid-name
    def fit(
        self,
        dataset: IterableDataset,
        sample_weight: Optional[np.ndarray] = None,
        reset: bool = True,
        num_steps: int = 1000,
        warm_starting_batch: int = 10,
    ) -> "PegasosQSVC":
        """Fit the model according to the given training data.
        Args:
            dataset: A pytorch IterableDataset providing the training data. Iterating over this data set
                     should return a tuple of a single data point `(x, y)' for features `x` and the label `y`.
            sample_weight: this parameter is not supported, passing a value raises an error.
            reset: if set to ``True``, the training data is overwritten and a fresh model is trained.
                   if set to ``False``, the new data is added to existing training data and the model
                   is fitted to adjust for new data.
            num_steps: number of steps in the Pegasos algorithm. There is no early stopping
                criterion. The algorithm iterates over all steps.
            warm_starting_batch: Number of data points used for warm starting using the dual qsvc. Set to None
                                 for cold starting.

        Returns:
            ``self``, Fitted estimator.

        Raises:
            ValueError:
                - X and/or y have the wrong shape.
                - X and y have incompatible dimensions.
                - y includes more than two unique labels.
                - Pre-computed kernel matrix has the wrong shape and/or dimension.

            NotImplementedError:
                - when a sample_weight which is not None is passed.
        """
        if sample_weight is not None:
            raise NotImplementedError(
                "Parameter 'sample_weight' is not supported. All samples have to be weighed equally"
            )

        self._num_steps = num_steps
        # reset the fit state
        self.fit_status_ = PegasosQSVC.UNFITTED

        if (self._training_data is None) or (reset):
            # reset training data
            self._training_data = PegasosDataStorage()
            self._alphas = []
        # update data loader
        self._data_loader = PegasosDataLoading(dataset)

        t_0 = datetime.now()

        if warm_starting_batch is not None:
            self._warm_start(warm_starting_batch)

        # training loop
        step = 0
        for x_step, y_step in self._data_loader:
            self._training_data.store_new(x_step, y_step)
            self._update_step(x_step, y_step)
            step += 1
            if step > num_steps:
                break
            if step % 100 == 0:
                print(f"Step {step}/{num_steps}.")

        self.fit_status_ = PegasosQSVC.FITTED

        logger.debug("fit completed after %s", str(datetime.now() - t_0)[:-7])

        return self

    def step(self, x_step: np.ndarray, y_step: int):
        """
        Implements an update step for the fit method.

        Args:
            x_step: features of the data point sampled in this step
            y_step: label of the data point sampled in this step

        """
        self._training_data.store_new(x_step, y_step)
        self._update_step(x_step, y_step)

    # pylint: disable=invalid-name
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Perform classification on samples in X.
        Args:
            X: Features. For a callable kernel (an instance of ``QuantumKernel``) the shape
               should be ``(m_samples, n_features)``, for a precomputed kernel the shape should be
               ``(m_samples, n_samples)``. Where ``m`` denotes the set to be predicted and ``n`` the
               size of the training set. In that case, the kernel values in X have to be calculated
               with respect to the elements of the set to be predicted and the training set.
        Returns:
            An array of the shape (n_samples), the predicted class labels for samples in X.
        Raises:
            QiskitMachineLearningError:
                - predict is called before the model has been fit.
            ValueError:
                - Pre-computed kernel matrix has the wrong shape and/or dimension.
        """

        t_0 = datetime.now()
        values = self.decision_function(X)
        y = np.array(
            [self._label_pos if val > 0 else self._label_neg for val in values]
        )
        logger.debug("prediction completed after %s", str(datetime.now() - t_0)[:-7])

        return y

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the decision function for the samples in X.
        Args:
            X: Features. For a callable kernel (an instance of ``QuantumKernel``) the shape
               should be ``(m_samples, n_features)``, for a precomputed kernel the shape should be
               ``(m_samples, n_samples)``. Where ``m`` denotes the set to be predicted and ``n`` the
               size of the training set. In that case, the kernel values in X have to be calculated
               with respect to the elements of the set to be predicted and the training set.
        Returns:
            An array of the shape (n_samples), the decision function of the sample.
        Raises:
            QiskitMachineLearningError:
                - the method is called before the model has been fit.
            ValueError:
                - Pre-computed kernel matrix has the wrong shape and/or dimension.
        """
        if self.fit_status_ == PegasosQSVC.UNFITTED:
            raise QiskitMachineLearningError("The PegasosQSVC has to be fit first")
        X = np.atleast_2d(X)
        if self._precomputed and self._n_samples != X.shape[1]:
            raise ValueError(
                "For a precomputed kernel, X should be in shape (m_samples, n_samples)"
            )

        values = self._compute_weighted_kernel_sum(X)

        return values / np.min([self._training_data.length, self._steps_used])

    def predict_history(self, X: np.ndarray):
        """
        Experimental
        """
        if self.fit_status_ == PegasosQSVC.UNFITTED:
            raise QiskitMachineLearningError("The PegasosQSVC has to be fit first")
        if np.ndim(X) != 2:
            raise ValueError("X has to be a 2D array")

        history = np.ones((self._training_data.length, X.shape[0]))
        t_0 = datetime.now()
        steps_used = self._steps_used
        if steps_used is not None:
            self._steps_used = None
        support, values = self._compute_weighted_kernel_sum(X, return_history=True)

        last_i = 0
        for i, s in enumerate(support):
            if steps_used is not None:
                while (s - support[last_i]) > steps_used:
                    last_i += 1

            y = np.sum(values[:, last_i : i + 1], axis=-1)
            if i + 1 < len(support):
                next_s = support[i + 1]
            else:
                next_s = None
            history[s:next_s, y < 0] = -1

        self._steps_used = steps_used
        logger.debug("prediction completed after %s", str(datetime.now() - t_0)[:-7])

        return history

    def _update_step(self, x_step: np.ndarray, y_step: int):
        """
        Implements an update step for the fit method.

        Args:
            x_step: features of the data point sampled in this step
            y_step: label of the data point sampled in this step

        """
        value = self._compute_weighted_kernel_sum(x_step)
        updated = False
        step = np.min([self._training_data.length + 1, self._steps_used])
        # step = self._training_data.length + 1
        if y_step * self.C / step * value < 1:
            # only way for a component of alpha to become non zero
            self._alphas.append(1)
            updated = True
        else:
            self._alphas.append(0)

        if self._callback is not None:
            self._callback(step, value, updated, x_step, y_step)

    def _compute_weighted_kernel_sum(
        self, x: np.ndarray, return_history: bool = False
    ) -> float:
        """Helper function to compute the weighted sum over support vectors used for both training
        and prediction with the Pegasos algorithm.

        Args:
            X: Feature vector

        Returns:
            Weighted sum of kernel evaluations employed in the Pegasos algorithm

        If return_history is set to True:
        Returns:
            support: Iterations of the support vectors
            terms: Terms in the weighted sum
        """
        if len(self._alphas) == 0:
            return 0

        alphas = np.array(self._alphas)
        support = np.arange(len(alphas))[alphas == 1]
        if self._steps_used is not None:
            support = support[support > len(alphas) - self._steps_used - 1]
        # support vectors
        x_supp = self._training_data.X[support]

        # evaluate kernel function only for the fixed datum and the support vectors
        kernel = self._quantum_kernel.evaluate(x, x_supp) + self._kernel_offset

        y = self._training_data.y[support]

        if not return_history:
            # this value corresponds to a sum of kernel values weighted by their labels and alphas
            value = np.sum(y * kernel, axis=-1)
            return value

        return support, y * kernel

    def _warm_start(self, batch_size: int):
        """
        This function solves the QSVM problem on a subset of size `batch_size`,
        to get initial support vectors and alphas.
        """

        # load a small data set
        X = []
        y = []
        for x_i, y_i in self._data_loader:
            X.append(x_i)
            y.append(y_i)
            if len(X) >= batch_size and len(set(y)) > 1:
                break
        X, y = np.array(X), np.array(y)

        # solve the dual qsvm problem
        qsvc = QSVC(quantum_kernel=self._quantum_kernel, C=self.C)
        qsvc.fit(X, y)
        # convert dual coefs to alphas of reasonable size
        alphas = np.abs(qsvc.dual_coef_).flatten()
        alphas = len(alphas) / np.sum(alphas) * alphas
        alphas = np.round(alphas).astype(int)

        # save the support vectors
        for i, a in enumerate(alphas):
            index = qsvc.support_[i]
            for _ in range(a):
                self._alphas.append(1)
                self._training_data.store_new(X[index], y[index])

        # shuffle support vectors
        self._training_data.shuffle()

    @property
    def quantum_kernel(self) -> QuantumKernel:
        """Returns quantum kernel"""
        return self._quantum_kernel

    @quantum_kernel.setter
    def quantum_kernel(self, quantum_kernel: QuantumKernel):
        """
        Sets quantum kernel. If previously a precomputed kernel was set, it is reset to ``False``.
        """

        self._quantum_kernel = quantum_kernel
        # quantum kernel is set, so we assume the kernel is not precomputed
        self._precomputed = False

        # reset training status
        self._reset_state()

    @property
    def num_steps(self) -> int:
        """Returns number of steps in the Pegasos algorithm."""
        return self._num_steps

    @num_steps.setter
    def num_steps(self, num_steps: int):
        """Sets the number of steps to be used in the Pegasos algorithm."""
        self._num_steps = num_steps

        # reset training status
        self._reset_state()

    @property
    def precomputed(self) -> bool:
        """Returns a boolean flag indicating whether a precomputed kernel is used."""
        return self._precomputed

    @precomputed.setter
    def precomputed(self, precomputed: bool):
        """Sets the pre-computed kernel flag. If ``True`` is passed then the previous kernel is
        cleared. If ``False`` is passed then a new instance of ``QuantumKernel`` is created."""
        self._precomputed = precomputed
        if precomputed:
            # remove the kernel, a precomputed will
            self._quantum_kernel = None
        else:
            # re-create a new default quantum kernel
            self._quantum_kernel = QuantumKernel()

        # reset training status
        self._reset_state()

    def _reset_state(self):
        """Resets internal data structures used in training."""
        self.fit_status_ = PegasosQSVC.UNFITTED
        self._alphas = None
        self._x_train = None
        self._n_samples = None
        self._y_train = None
        self._label_map = None
        self._label_pos = None
        self._label_neg = None


"""
QKA Part
"""
from typing import Optional, Union, Iterator

from qiskit.utils import algorithm_globals
from qiskit.algorithms.optimizers import SPSA


class PegasosQKA(PegasosQSVC):
    def __init__(
        self,
        quantum_kernel: Optional[PseudoKernel] = None,
        C: float = 1,
        precomputed: bool = False,
        seed: Optional[int] = None,
        steps_used: Optional[int] = None,
        initial_guess: Optional[np.ndarray] = None,
        spsa_calibration: bool = True,
        calibrate_every: Optional[int] = None,
        calibration_steps: int = 0,
        use_theta_history: bool = True,
        learning_rate: float = 0.01,
        callback: Optional[CALLBACK] = None,
    ) -> None:
        super().__init__(
            quantum_kernel=quantum_kernel,
            C=C,
            precomputed=precomputed,
            seed=seed,
            steps_used=steps_used,
            callback=callback,
        )

        if initial_guess is None:
            self._thetas = [np.zeros(quantum_kernel.num_parameters)]
        else:
            if len(initial_guess) == quantum_kernel.num_parameters:
                self._thetas = [initial_guess]
            else:
                raise ValueError(
                    f"Number of parameters in initial guess ({len(initial_guess)}) does not mach number of parameters in PseudoKernel ({quantum_kernel.num_parameters})."
                )

        self._left_theta = self._thetas[-1].copy()
        self._calibration_steps = calibration_steps
        self._blasphemy = not use_theta_history
        self.learning_rate: Union[float, Iterator[float]] = learning_rate
        self.perturbations: Union[float, Iterator[float]] = learning_rate
        self._calibrated = False
        self._use_calibration = spsa_calibration
        self._calibrate_every = calibrate_every

    def _update_step(self, x_step: np.ndarray, y_step: int) -> bool:
        """
        Implements an update step for the fit method.

        Args:
            x_step: features of the data point sampled in this step
            y_step: label of the data point sampled in this step
        """
        value = self._compute_weighted_kernel_sum(x_step)
        step = self._training_data.length
        new_theta = self._thetas[-1]
        updated = False
        if (y_step * self.C / (step + 1)) * value < 1:
            updated = True
            if step >= self._calibration_steps:
                if self._use_calibration:
                    if not self._calibrated:
                        self.learning_rate, self.perturbations = self._calibrate()
                        self._calibrated = True
                    # learning rate is given as iterator
                    c_step = next(self.perturbations)
                    h_step = next(self.learning_rate)
                else:
                    # learning rate is given as float
                    c_step = self.perturbations
                    h_step = self.learning_rate

                # choose update direction
                n = bernoulli_perturbation(self._quantum_kernel.num_parameters)

                # approximate gradient in that direction
                factor = y_step * self.C / (step + 1)
                theta_plus = new_theta + c_step * n
                theta_minus = new_theta - c_step * n
                self._left_theta = theta_plus
                obj_plus = -factor * self._compute_weighted_kernel_sum(x_step)
                self._left_theta = theta_minus
                obj_minus = -factor * self._compute_weighted_kernel_sum(x_step)

                gradient = (obj_plus - obj_minus) / (2 * c_step) if c_step > 0 else 0

                new_theta = (new_theta - h_step * gradient * n).flatten()
            # only way for a component of alpha to become non zero
            self._alphas.append(1)
        else:
            self._alphas.append(0)

        # update kernel parameters
        self._thetas.append(new_theta)
        self._left_theta = new_theta

        # recalibrate if wished
        if self._calibrate_every is not None:
            if (step + 1) % self._calibrate_every == 0:
                self._calibrated = False

        if self._callback is not None:
            self._callback(step, value, updated, x_step, y_step)

    def _compute_weighted_kernel_sum(
        self, x: np.ndarray, return_history: bool = False
    ) -> float:
        """Helper function to compute the weighted sum over support vectors used for both training
        and prediction with the Pegasos algorithm.

        Args:
            X: Feature vector

        Returns:
            Weighted sum of kernel evaluations employed in the Pegasos algorithm

        If return_history is set to True:
        Returns:
            support: Iterations of the support vectors
            terms: Terms in the weighted sum
        """
        if len(self._alphas) == 0:
            return 0

        alphas = np.array(self._alphas)
        thetas = np.array(self._thetas)
        support = np.arange(len(alphas))[alphas == 1]
        if self._steps_used is not None:
            support = support[support > len(alphas) - self._steps_used - 1]
        # support vectors
        x_supp = self._training_data.X[support]
        theta_supp = thetas[support]

        # evaluate kernel function only for the fixed datum and the support vectors
        if self._blasphemy:
            kernel = (
                self._quantum_kernel.evaluate_batch(
                    x, x_supp, self._left_theta, self._left_theta
                )
                + self._kernel_offset
            )
        else:
            kernel = (
                self._quantum_kernel.evaluate_batch(
                    x, x_supp, self._left_theta, theta_supp
                )
                + self._kernel_offset
            )

        y = self._training_data.y[support]

        if not return_history:
            # this value corresponds to a sum of kernel values weighted by their labels and alphas
            value = np.sum(y * kernel, axis=-1)
            return value

        return support, y * kernel

    def _warm_start(self, batch_size: int):
        super()._warm_start(batch_size)
        self._thetas = list(make_2D(self._left_theta, self._training_data.length))

    def _calibrate(self, num_samples=20):
        """
        Wrapper around SPSA.calibrate() method to get learning rate and perturbations.
        """
        if self._training_data.length < num_samples:
            num_samples = self._training_data.length
        train_x = self._training_data.X[:num_samples]
        train_y = self._training_data.y[:num_samples]

        if train_x is None:
            raise ValueError(
                "To use SPSA calibration either warm starting has to be enabled or calibration steps must be a positive number."
            )

        def loss(theta):
            self._left_theta = theta
            values = self._compute_weighted_kernel_sum(train_x)
            hinge_loss = np.max(
                [
                    np.zeros(num_samples),
                    1.0 - self.C / (self._training_data.length + 1) * train_y * values,
                ]
            )
            return np.mean(hinge_loss)

        left_theta = self._left_theta
        learning_rate, perturbations = SPSA.calibrate(loss, left_theta)
        self._left_theta = left_theta
        return learning_rate(), perturbations()

    def to_dict(self):
        pegasos_dict = {
            "C": self.C,
            "data": self._training_data.to_dict(),
            "alphas": self._alphas,
            "kernel_offset": self._kernel_offset,
            "fit_status": self.fit_status_,
            "steps_used": self._steps_used,
            "thetas": self._thetas,
            "learning_rate": self.learning_rate,
            "perturbations": self.perturbations,
            "calibration_steps": self._calibration_steps,
            "no_theta_hist": self._blasphemy,
            "use_calibration": self._use_calibration,
            "calibrate_every": self._calibrate_every,
            "calibrated": self._calibrated,
        }
        return pegasos_dict

    def from_dict(self, pegasos_dict: dict) -> None:
        self._training_data = PegasosDataStorage().from_dict(pegasos_dict["data"])
        self._alphas = pegasos_dict["alphas"]
        self.C = pegasos_dict["C"]
        self._kernel_offset = pegasos_dict["kernel_offset"]
        self.fit_status_ = pegasos_dict["fit_status"]
        self._steps_used = pegasos_dict["steps_used"]
        self._thetas = pegasos_dict["thetas"]
        self.learning_rate = pegasos_dict["learning_rate"]
        self.perturbations = pegasos_dict["perturbations"]
        self._left_theta = self._thetas[-1].copy()
        self._calibration_steps = pegasos_dict["calibration_steps"]
        self._blasphemy = pegasos_dict["no_theta_hist"]
        self._calibrated = pegasos_dict["calibrated"]
        self._use_calibration = pegasos_dict["use_calibration"]
        self._calibrate_every = pegasos_dict["calibrate_every"]
        return self


def bernoulli_perturbation(dim, perturbation_dims=None):
    """Get a Bernoulli random perturbation."""
    if perturbation_dims is None:
        return 1 - 2 * algorithm_globals.random.binomial(1, 0.5, size=dim)

    pert = 1 - 2 * algorithm_globals.random.binomial(1, 0.5, size=perturbation_dims)
    indices = algorithm_globals.random.choice(
        list(range(dim)), size=perturbation_dims, replace=False
    )
    result = np.zeros(dim)
    result[indices] = pert

    return result


"""
Feature Map
"""


from typing import Union, List, Callable


# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.algorithms.optimizers import SPSA
from qiskit_machine_learning.algorithms import QSVC


from typing import Callable, Optional, Union, List, Dict, Any

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


class CovariantFeatureMap(QuantumCircuit):
    """The Covariant Feature Map circuit.

    On 3 qubits and a linear entanglement,  the circuit is represented by:

    .. parsed-literal::


         ┌──────────────┐       ░ ┌─────────────────┐┌─────────────────┐
    q_0: ┤ Ry(θ_par[0]) ├─■─────░─┤ Rz(-2*x_par[1]) ├┤ Rx(-2*x_par[0]) ├
         ├──────────────┤ │     ░ ├─────────────────┤├─────────────────┤
    q_1: ┤ Ry(θ_par[1]) ├─■──■──░─┤ Rz(-2*x_par[3]) ├┤ Rx(-2*x_par[2]) ├
         ├──────────────┤    │  ░ ├─────────────────┤├─────────────────┤
    q_2: ┤ Ry(θ_par[2]) ├────■──░─┤ Rz(-2*x_par[5]) ├┤ Rx(-2*x_par[4]) ├
         └──────────────┘       ░ └─────────────────┘└─────────────────┘

    where θ_par is a vector of trainable feature map parameters and x_par is a
    vector of data-bound feature map parameters.
    """

    def __init__(
        self,
        feature_dimension: int,
        entanglement: Union[str, List[List[int]], Callable[[int], List[int]]] = None,
        single_training_parameter: bool = False,
        name: str = "CovariantFeatureMap",
    ) -> None:
        """Create a new Covariant Feature Map circuit.

        Args:
            feature_dimension: The number of features
            insert_barriers: If True, barriers are inserted around the entanglement layer

        """
        if (feature_dimension % 2) != 0:
            raise ValueError(
                """
            Covariant feature map requires an even number of input features.
                """
            )
        self.feature_dimension = feature_dimension
        self.entanglement = entanglement
        self.single_training_parameter = single_training_parameter
        self.user_parameters = None
        self.input_parameters = None

        num_qubits = feature_dimension / 2
        super().__init__(
            num_qubits,
            name=name,
        )

        self._generate_feature_map()

    @property
    def settings(self) -> Dict[str, Any]:
        user_parameters_list = [param for param in self.user_parameters]
        input_parameters_list = [param for param in self.input_parameters]
        return {
            "feature_dimension": self.feature_dimension,
            "entanglement": self.entanglement,
            "single_training_parameter": self.single_training_parameter,
            "user_parameters": user_parameters_list,
            "input_parameters": input_parameters_list,
        }

    def _generate_feature_map(self):
        # If no entanglement scheme specified, use linear entanglement
        if self.entanglement is None:
            self.entanglement = [[i, i + 1] for i in range(self.num_qubits - 1)]

        # Vector of data parameters
        input_params = ParameterVector("x_par", self.feature_dimension)

        # Use a single parameter to rotate each qubit if sharing is desired
        if self.single_training_parameter:
            user_params = ParameterVector("\u03B8_par", 1)
            # Create an initial rotation layer using a single Parameter
            for i in range(self.num_qubits):
                self.ry(user_params[0], self.qubits[i])

        # Train each qubit's initial rotation individually
        else:
            user_params = ParameterVector("\u03B8_par", self.num_qubits)
            # Create an initial rotation layer of trainable parameters
            for i in range(self.num_qubits):
                self.ry(user_params[i], self.qubits[i])

        self.user_parameters = user_params
        self.input_parameters = input_params

        # Create the entanglement layer
        for source, target in self.entanglement:
            self.cz(self.qubits[source], self.qubits[target])

        self.barrier()

        # Create a circuit representation of the data group
        for i in range(self.num_qubits):
            self.rz(-2 * input_params[2 * i + 1], self.qubits[i])
            self.rx(-2 * input_params[2 * i], self.qubits[i])


class DriftingCovariantFeatureMap(CovariantFeatureMap):
    """
    This feature map is assuming that the last feature is
    the true value of lambda. Be sure that you are using a dataset loaded
    from the method `load_dataset_given_lambda`, otherwise this will break.
    """

    def __init__(
        self,
        feature_dimension: int,
        entanglement: Union[str, List[List[int]], Callable[[int], List[int]]] = None,
        single_training_parameter: bool = False,
        name: str = "CovariantFeatureMap",
    ) -> None:
        super().__init__(
            feature_dimension - 1, entanglement, single_training_parameter, name
        )

    def _generate_feature_map(self):
        # Get the normal feature map from the parent class
        super()._generate_feature_map()

        # Add a new input parameter to the feature map, which will
        # be bound to the lambda feature we added to the dataset
        self.input_parameters.resize(len(self.input_parameters) + 1)
        ry_offset = np.pi / 2 - self.input_parameters[-1]

        # Replace the theta parameter(s) with itself plus an offset
        rebinds = {}
        for up in self.user_parameters:
            rebinds[up] = up + ry_offset
        self.assign_parameters(rebinds, inplace=True)


"""
Dataset
"""


class DriftingLambda(IterableDataset):
    """
    Drifting lambda dataset for Jen's paper
    """

    def __init__(
        self,
        data: np.ndarray,
        size: Optional[int] = None,
        step_size=0.1,
        initial_lambda=0.0,
        test_size=0,
    ) -> None:
        """
        If size is provided, step_size is overridden by 2*pi/size.
        """
        super().__init__()
        self.X = data[:, :-1]
        self.y = data[:, -1].reshape(-1)
        self._count = 0
        self.size = size
        self._initial_lambda = initial_lambda
        self._step_size = step_size
        if size is not None:
            self._step_size = 2 * np.pi / size

        if test_size > 0:
            self.X_test = self.X[-test_size:, :]
            self.y_test = self.y[-test_size:]
            self.X = self.X[:test_size, :]
            self.y = self.y[:test_size]

    def _get_data_point(self):
        index = np.random.choice(len(self.X))
        x = np.append(
            self.X[index, :],
            np.sin(self._initial_lambda + self._count * self._step_size),
        )
        return x, self.y[index]

    def from_dict(self, ds_dict):
        """
        Create the data set from a dictionary
        """
        self._count = ds_dict["count"]
        self._initial_lambda = ds_dict["initial_lambda"]

        return self

    def to_dict(self):
        """
        Save dataset to dictionary
        """
        return {"count": self._count, "initial_lambda": self._initial_lambda}

    def get_test_set(self):
        """
        Return the saved test set.
        """
        X = np.hstack(
            [
                self.X_test,
                np.sin(
                    self._initial_lambda
                    + np.arange(self._count, self._count + len(self.X_test))
                    * self._step_size
                ).reshape((1, -1)),
            ]
        )
        return X, self.y_test

    def __next__(self):
        if self.size is not None and self._count >= self.size:
            raise StopIteration
        self._count += 1
        return self._get_data_point()

    def __iter__(self):
        return self


"""
Actual code!
"""


def program(
    backend,
    user_messenger,
    num_steps,
    data,
    pegasos_dict=None,
    data_dict=None,
    qubits=7,
    dynamical_decoupling=False,
    tau=100,
    shots=1024,
):
    """
    Actual program
    """
    if qubits == 7:
        entangler_map = [[0, 2], [3, 4], [2, 5], [1, 4], [2, 3], [4, 6]]
        initial_layout = [0, 4, 1, 3, 5, 2, 6]
    elif qubits == 27:
        entangler_map = [
            [19, 20],
            [8, 9],
            [3, 2],
            [0, 1],
            [4, 7],
            [10, 12],
            [15, 18],
            [21, 23],
            [24, 25],
            [22, 19],
            [14, 13],
            [11, 8],
            [5, 3],
            [2, 1],
            [1, 4],
            [7, 6],
            [7, 10],
            [12, 13],
            [16, 14],
            [12, 15],
            [18, 17],
            [18, 21],
            [23, 24],
            [25, 26],
            [25, 22],
            [14, 11],
            [19, 16],
            [8, 5],
        ]
        initial_layout = list(range(27))
    else:
        return {"Error": "Qubits must be set to 7 or 27"}

    num_features = data.shape[1] - 1 + 1
    fm = DriftingCovariantFeatureMap(
        feature_dimension=num_features,
        entanglement=entangler_map,
        single_training_parameter=True,
    )

    stepsize = 2 * np.pi / 1000
    ds = DriftingLambda(data, step_size=stepsize)
    if data_dict is not None:
        ds = ds.from_dict(data_dict)

    values = []
    y_true = []

    def callback(step, value, updated, x_step, y_step):
        values.append(value)
        y_true.append(y_step)

    factory = FidelityFactory(
        backend,
        shots,
        initial_layout=initial_layout,
        dynamical_decoupling=dynamical_decoupling,
    )
    kernel = PseudoKernel(factory, fm, num_training_parameters=1)
    qka = PegasosQKA(
        kernel,
        C=1000,
        spsa_calibration=False,
        learning_rate=0.1,
        steps_used=tau,
        use_theta_history=True,
        callback=callback,
    )
    if pegasos_dict is not None:
        qka.from_dict(pegasos_dict)
        qka.fit(ds, num_steps=num_steps, reset=False, warm_starting_batch=None)
    else:
        qka.fit(ds, num_steps=num_steps)

    result_dict = {
        "values": np.array(values).reshape(-1),
        "y_true": np.array(y_true).reshape(-1),
        "qka": qka.to_dict(),
        "dataset": ds.to_dict(),
    }

    return result_dict


def main(backend, user_messenger, **kwargs):
    """This is the main entry point of a runtime program.

    The name of this method must not change. It also must have ``backend``
    and ``user_messenger`` as the first two positional arguments.

    Args:
        backend: Backend for the circuits to run on.
        user_messenger: Used to communicate with the program user.
        kwargs: User inputs.
    """
    num_steps = kwargs.get("num_steps", 500)
    data = kwargs.get("data", None)
    qka_dict = kwargs.get("qka", None)
    dataset_dict = kwargs.get("dataset", None)
    qubits = kwargs.get("qubits", 7)
    dynamical_decoupling = kwargs.get("dynamical_decoupling", False)
    tau = kwargs.get("tau", 150)
    shots = kwargs.get("shots", 1024)

    # do the work
    start = time()
    result_dict = program(
        backend,
        user_messenger,
        num_steps,
        data,
        qka_dict,
        dataset_dict,
        qubits,
        dynamical_decoupling,
        tau,
        shots,
    )
    end = time()

    result_dict["time"] = end - start
    user_messenger.publish(result_dict, final=True)
