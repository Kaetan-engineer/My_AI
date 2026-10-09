import numpy as np
from numpy import ndarray
from typing import Literal
from jarvis.model.activation import Activation


class DenseLayer:
    def __init__(self, input_count: int, neuron_count: int, activation: Activation, initialization: str | None=None) -> None:
        raw_weights = np.random.randn(input_count, neuron_count)

        if initialization is not None:
            if initialization.lower() == "he":
                self.weights: ndarray = raw_weights * np.sqrt(2.0 / input_count)
            elif initialization.lower() == "xavier":
                self.weights = raw_weights * np.sqrt(1.0 / input_count)
        else:
            self.weights: ndarray = np.array(raw_weights * 0.01)
        self.biases: ndarray = np.zeros(neuron_count)

        self.activation: Activation = activation
        self.pre_activation = None

        self.weight_mean = np.zeros_like(self.weights)
        self.bias_mean = np.zeros_like(self.biases)

        self.weight_velocity = np.zeros_like(self.weights)
        self.bias_velocity = np.zeros_like(self.biases)

        self.dweights = np.zeros_like(self.weights)
        self.dbiases = np.zeros_like(self.biases)
        self.dinputs = None

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, inputs: ndarray) -> ndarray:
        self.inputs = inputs if inputs.ndim == 2 else np.atleast_2d(inputs)

        self.pre_activation = self.inputs @ self.weights + self.biases

        print(f"Pre activation dims: {self.pre_activation.shape}")
        print(f"Weights (forward) dims: {self.weights.shape}")
        print("\n")

        return self.activation.forward(self.pre_activation)

    def backward(self, output_gradient: ndarray) -> ndarray:
        z_gradient = self.activation.backward(output_gradient)

        print(f"Z gradient dims: {z_gradient.shape}")
        print(f"Weights (backward) dims: {self.weights.shape}")
        print("\n")

        self.dweights = self.inputs.T @ z_gradient
        self.dbiases = np.sum(z_gradient, axis=0)
        self.dinputs = z_gradient @ self.weights.T

        return self.dinputs

    def update(self, learning_rate: float, optimizer) -> None:
        optimizer(learning_rate, self)

    def zero_grad(self) -> None:
        self.dweights = np.zeros_like(self.weights)
        self.dbiases = np.zeros_like(self.biases)

