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


from torch.utils.data import IterableDataset
import numpy as np
from typing import Optional
from qiskit.utils import algorithm_globals
import pandas as pd


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
        index = algorithm_globals.random.integers(0, len(self.X))
        return self.X[index], self.y[index]

    def __iter__(self):
        return self


class DynamicalDataset(IterableDataset):
    def __init__(
        self,
        filename="data/parametrized/dynamical_data.csv",
        stay_in_batch: Optional[int] = None,
    ) -> None:
        super().__init__()
        data = pd.read_csv(filename)
        self._batchsize = stay_in_batch
        if stay_in_batch is not None:
            self.data = self._sample_data(data)
        else:
            self.data = data

        self.data.sort_values(by="Theta", inplace=True, ignore_index=True)

        self._index = 0

    def _sample_data(self, data):
        thetas = list(set(data["Theta"]))
        new_df = pd.DataFrame(columns=data.columns)
        for theta in thetas:
            sample = algorithm_globals.random.choice(
                data[data["Theta"] == theta], self._batchsize, replace=True
            )
            new_df = new_df.append(
                pd.DataFrame(columns=data.columns, data=sample), ignore_index=True
            )
        return new_df

    def __next__(self):
        try:
            datum = np.array(self.data.loc[self._index])
        except KeyError:
            raise StopIteration
        self._index += 1
        return datum[1:-1], datum[-1]

    def __iter__(self):
        return self


class DriftingLambda(IterableDataset):
    """
    Drifting lambda dataset for Jen's paper
    """

    def __init__(
        self,
        size: Optional[int] = None,
        step_size=0.1,
        initial_lambda=0.0,
        test_size=0,
    ) -> None:
        """
        If size is provided, step_size is overridden by 2*pi/size.
        """
        super().__init__()
        data = pd.read_csv("data/dataset_graph10.csv")
        self.X = np.array(data.iloc[:, :-1])
        self.y = np.array(data.iloc[:, -1]).reshape(-1)
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

    def _get_data_point(self, test=False):
        index = np.random.choice(len(self.X))
        x = np.append(
            self.X[index, :],
            self._initial_lambda + np.sin(self._count * self._step_size),
        )
        return x, self.y[index]

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
                ).reshape((-1, 1)),
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
