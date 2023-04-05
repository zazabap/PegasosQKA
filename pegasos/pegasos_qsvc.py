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
from typing import Callable, Optional, Dict, List, Union, Tuple

import numpy as np

from torch.utils.data import IterableDataset
from sklearn.base import ClassifierMixin
from qiskit.utils import algorithm_globals

from qiskit_machine_learning.algorithms.classifiers import QSVC
from qiskit_machine_learning.exceptions import QiskitMachineLearningError


from .pegasos_data_loader import PegasosDataLoading
from .pegasos_data_storage import PegasosDataStorage


logger = logging.getLogger(__name__)

# callback arguments: step number, decision value, if-condition, x_step, y_step
CALLBACK = Callable[[int, float, bool, np.ndarray, int], None]


class PegasosQSVC(ClassifierMixin):
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

    def __init__(
        self,
        quantum_kernel,
        C: float = 1000.0,
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
        if C > 0:
            self.C = C
        else:
            raise ValueError(f"C has to be a positive number, found {C}.")

        self._quantum_kernel = quantum_kernel

        if seed is not None:
            algorithm_globals.random_seed = seed

        self._callback = callback

        if steps_used is not None:
            if not isinstance(steps_used, int) or steps_used < 1:
                raise ValueError("'steps_used' has to be a positive integer or None.")
            self._steps_used = steps_used
        else:
            self._steps_used = None

        # these are the parameters being fit and are needed for prediction
        self._alphas: Optional[List[int]] = None
        self._data_loader: Optional[PegasosDataLoading] = None
        self._training_data: Optional[PegasosDataStorage] = None
        self._n_samples: Optional[int] = None
        self._label_map: Optional[Dict[int, int]] = None
        self._label_pos: Optional[int] = 1
        self._label_neg: Optional[int] = -1

        # added to all kernel values to include an implicit bias to the hyperplane
        self._kernel_offset = 1

    # pylint: disable=invalid-name
    def fit(
        self,
        dataset: IterableDataset,
        reset: bool = True,
        num_steps: int = 1000,
        warm_starting_batch: int = 10,
    ) -> "PegasosQSVC":
        """Fit the model according to the given training data.
        Args:
            dataset: A pytorch IterableDataset providing the training data. Iterating over this data set
                     should return a tuple of a single data point `(x, y)' for features `x` and the label `y`.
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
        X = np.atleast_2d(X)
        values = self._compute_weighted_kernel_sum(X)

        if self._steps_used is not None:
            return values / np.min([self._training_data.length, self._steps_used])
        else:
            return values / self._training_data.length

    def predict_history(self, X: np.ndarray):
        """
        Experimental, returns the support vector indices as well as the predictions
        at every iteration. Better to use a callback to get the same information.
        Keeping this so that old experiments still run.
        """
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
        if y_step * self.C / step * value < 1:
            # only way for a component of alpha to become non zero
            self._alphas.append(1)
            updated = True
        else:
            self._alphas.append(0)

        if self._callback is not None:
            self._callback(
                self._training_data.length, value / step, updated, x_step, y_step
            )

    def _compute_weighted_kernel_sum(
        self, x: np.ndarray, return_history: bool = False
    ) -> Union[float, Tuple[np.ndarray, np.ndarray]]:
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
        alphas = alphas * len(alphas) / np.sum(alphas)
        alphas = np.round(alphas).astype(int)

        # save the support vectors
        for i, a in enumerate(alphas):
            index = qsvc.support_[i]
            for _ in range(a):
                self._alphas.append(1)
                self._training_data.store_new(X[index], y[index])

        # shuffle support vectors
        self._training_data.shuffle()
