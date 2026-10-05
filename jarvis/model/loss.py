import numpy as np
from abc import ABC, abstractmethod
from numpy import ndarray, float64


class Loss(ABC):
    @abstractmethod
    def forward(self, y_pred: ndarray, y_actual: ndarray) -> float64:
        return NotImplemented

    @abstractmethod
    def backward(self, y_pred: ndarray, y_actual: ndarray) -> ndarray:
        return NotImplemented


class MSE(Loss):
    def forward(self, y_pred: ndarray, y_actual: ndarray) -> float64:
        return np.mean((y_pred - y_actual) ** 2)

    def backward(self, y_pred: ndarray, y_actual: ndarray) -> ndarray:
        return 2 * (y_pred - y_actual) / y_pred.size