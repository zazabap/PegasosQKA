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
"""Data loader for PegasosQSVC and PegasosQKA"""

from typing import Iterable

import numpy as np
from torch.utils.data import IterableDataset, DataLoader


class PegasosDataLoading(Iterable):
    """
    This class handles the data loading for PegasosQSVC. Wrapper for the pytorch data loader.
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
