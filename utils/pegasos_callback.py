from typing import Optional
import numpy as np


class PegasosCallback:
    def __init__(self, print_every: Optional[int] = None) -> None:
        """
        Args:
            print_every: Set to positive integer to print accuracy every `print_every` iteration.
        """
        self.values = []
        self.y_true = []
        self._print_every = print_every

    def __call__(self, step, value, updated, x_step, y_step):
        """
        Called by Pegasos to append current value and step.
        """
        self.values.append(value)
        self.y_true.append(y_step)
        if self._print_every is not None and step % self._print_every == 0:
            print(f"Step {step}: Accuracy = {self.accuracy()*100:.1f}%")

    def accuracy(self):
        """
        Print the accuracy up to the current point
        """
        return np.sum(
            np.sign(self.values).reshape(-1) == np.array(self.y_true).reshape(-1)
        ) / len(self.values)
