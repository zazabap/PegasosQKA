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

"""Pegasos Quantum Kernel Alignment algorithm."""
from typing import Optional, Tuple, Union, Iterator

import numpy as np
from qiskit.utils import algorithm_globals
from qiskit.algorithms.optimizers import SPSA


from .pegasos_qsvc import CALLBACK

from .pegasos_qsvc import PegasosQSVC


def make_2d(array: np.ndarray, n_copies: int):
    """
    Takes a 1D numpy array and copies it n times along a second axis.
    """
    return np.repeat(array[np.newaxis, :], n_copies, axis=0)


class PegasosQKA(PegasosQSVC):
    def __init__(
        self,
        quantum_kernel=None,
        C: float = 1000.0,
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
            seed=seed,
            steps_used=steps_used,
            callback=callback,
        )

        if initial_guess is None:
            self._thetas = [np.zeros(quantum_kernel.num_parameters)]
        else:
            if len(initial_guess) == quantum_kernel.num_parameters:
                self._thetas = [np.atleast_1d(initial_guess)]
            else:
                raise ValueError(
                    f"Number of parameters in initial guess ({len(initial_guess)}) does not mach"
                    f"number of parameters in PseudoKernel ({quantum_kernel.num_parameters})."
                )

        self._left_theta = self._thetas[-1].copy()
        self._calibration_steps = calibration_steps
        self._blasphemy = not use_theta_history
        self.learning_rate: Union[float, Iterator[float]] = learning_rate
        self.perturbations: Union[float, Iterator[float]] = learning_rate
        self._calibrated = False
        self._use_calibration = spsa_calibration
        self._calibrate_every = calibrate_every

    def _update_step(self, x_step: np.ndarray, y_step: int) -> None:
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
            # update theta
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
            # update alpha
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
            self._callback(step, value, updated, x_step, y_step, self)

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
        self._thetas = list(make_2d(self._left_theta, self._training_data.length))

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
