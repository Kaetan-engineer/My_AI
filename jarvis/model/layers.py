import numpy as np
from numpy import ndarray
from typing import Literal
from jarvis.model.activation import Activation


class DenseLayer:
    def __init__(self, input_count: int, neuron_count: int, activation: Activation, initialization: str | None=None) -> None:
        if initialization is not None:
            if initialization.lower() == "he":
                self.weights: ndarray = np.random.rand(input_count, neuron_count) * np.sqrt(2.0 / input_count)
            elif initialization.lower() == "xavier":
                self.weights = np.random.rand(input_count, neuron_count) * np.sqrt(1.0 / input_count)
        else:
            self.weights: ndarray = np.array(np.random.rand(input_count, neuron_count))
        self.biases: ndarray = np.zeros(neuron_count)

        self.activation: Activation = activation
        self.pre_activation = None

        self.dweights = np.zeros_like(self.weights)
        self.dbiases = np.zeros_like(self.biases)
        self.dinputs = None

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, inputs: ndarray) -> ndarray:
        self.inputs = inputs if inputs.ndim == 2 else np.atleast_2d(inputs)

        self.pre_activation = self.inputs @ self.weights + self.biases

        return self.activation.forward(self.pre_activation)

    def backward(self, output_gradient: ndarray) -> ndarray:
        z_gradient = self.activation.backward(output_gradient)

        self.dweights = self.inputs.T @ z_gradient
        self.dbiases = np.sum(z_gradient, axis=0)
        self.dinputs = z_gradient @ self.weights.T

        return self.dinputs

    def update(self, learning_rate: float) -> None:
        self.weights -= learning_rate * self.dweights
        self.biases -= learning_rate * self.dbiases

    def zero_grad(self) -> None:
        self.dweights = np.zeros_like(self.weights)
        self.dbiases = np.zeros_like(self.biases)

