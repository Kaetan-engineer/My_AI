from abc import abstractmethod, ABC
import numpy as np
from model.layers import DenseLayer


class Optimizer(ABC):
    def __call__(self, *args, **kwargs) -> None:
        self.update(*args, **kwargs)

    @abstractmethod
    def update(self, learning_rate: float, layer: DenseLayer) -> None:
        ...


class SGD(Optimizer):
    def update(self, learning_rate: float, layer: DenseLayer) -> None:
        layer.weights -= learning_rate * layer.dweights
        layer.biases -= learning_rate * layer.dbiases


class Momentum(Optimizer):
    def __init__(self) -> None:
        self.factor = 0.9

    def update(self, learning_rate: float, layer: DenseLayer) -> None:
        layer.weight_velocity = (self.factor * layer.weight_velocity) + (1 - self.factor) * layer.dweights
        layer.bias_velocity = (self.factor * layer.bias_velocity) + (1 - self.factor) * layer.dbiases

        layer.weights -= learning_rate * layer.weight_velocity
        layer.biases -= learning_rate * layer.bias_velocity


class Adam(Optimizer):
    def __init__(self) -> None:
        self.decay1 = 0.9
        self.decay2 = 0.999
        self.e = 10e-8

    def update(self, learning_rate: float, layer: DenseLayer) -> None:
        layer.weight_mean = (self.decay1 * layer.weight_mean) + (1 - self.decay1) * layer.dweights
        layer.bias_mean = (self.decay1 * layer.bias_mean) + (1 - self.decay1) * layer.dbiases

        layer.weight_velocity = (self.decay2 * layer.weight_velocity) + (1 - self.decay2) * layer.dweights ** 2
        layer.bias_velocity = (self.decay2 * layer.bias_velocity) + (1 - self.decay2) * layer.dbiases ** 2

        weight_mean_correction = layer.weight_mean / (1 - self.decay1)
        bias_mean_correction = layer.bias_mean / (1 - self.decay1)

        weight_variance_correction = layer.weight_velocity / (1 - self.decay2)
        bias_variance_correction = layer.bias_velocity / (1 - self.decay2)

        layer.weights -= (weight_mean_correction / (np.sqrt(weight_variance_correction) + self.e)) * learning_rate
        layer.biases -= (bias_mean_correction / (np.sqrt(bias_variance_correction) + self.e)) * learning_rate

