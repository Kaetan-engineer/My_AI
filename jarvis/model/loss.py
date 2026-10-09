import numpy as np
from abc import ABC, abstractmethod
from numpy import ndarray, float64
from model.activation import Sigmoid, Softmax


class Loss(ABC):
    @abstractmethod
    def forward(self, y_pred: ndarray, y_actual: ndarray) -> float64:
        return NotImplemented

    @abstractmethod
    def backward(self, y_pred: ndarray, y_actual: ndarray) -> ndarray:
        return NotImplemented


class MeanSquareError(Loss):
    def forward(self, y_pred: ndarray, y_actual: ndarray) -> float64:
        return np.mean((y_pred - y_actual) ** 2)

    def backward(self, y_pred: ndarray, y_actual: ndarray) -> ndarray:
        return 2 * (y_pred - y_actual) / y_pred.size


class BinaryCrossEntropy(Loss):
    def forward(self, y_pred: ndarray, y_actual: ndarray) -> float64:
        print(f"Y pred (loss) dims: {y_pred.shape}")
        print(f"Y actual (loss) dims: {y_actual.shape}")
        print("\n")
        loss = np.mean(
            np.maximum(y_pred, 0)
            - y_pred * y_actual
            + np.log1p(np.exp(-np.abs(y_pred)))
        )

        return loss

    def backward(self, y_pred: ndarray, y_actual: ndarray) -> ndarray:
        print(f"Y pred (loss) dims: {y_pred.shape}")
        print(f"Y actual (loss) dims: {y_actual.shape}")
        probabilities = Sigmoid()(y_pred)
        loss = (probabilities - y_actual) / y_actual.size
        print(f"Loss gradient (loss) dims: {loss.shape}")
        print("\n")

        return loss


class CategoricalCrossEntropy(Loss):
    def forward(self, y_pred: ndarray, y_actual: ndarray) -> float64:
        logits = y_pred - np.max(y_pred, axis=1, keepdims=True)
        log_probs = logits - np.log(
            np.sum(np.exp(logits), axis=1, keepdims=True)
        )

        return -np.sum(y_actual * log_probs) / y_actual.shape[0]

    def backward(self, y_pred: ndarray, y_actual: ndarray) -> ndarray:
        logits = y_pred - np.max(y_pred, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probabilities = exp_logits / np.sum(
            exp_logits, axis=1, keepdims=True
        )

        return (probabilities - y_actual) / y_actual.shape[0]