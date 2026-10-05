import numpy as np
from numpy import ndarray, float64
from jarvis.model.activation import Activation
from jarvis.model.loss import Loss
from jarvis.model.layers import DenseLayer
from typing import Iterator


class NeuralNetwork:
    def __init__(self, layer_infrastructure: list[tuple[int, int, Activation, str | None]]) -> None:
        self.layers = list(
            DenseLayer(inputs, neurons, activation, initialization)
            for inputs, neurons, activation, initialization in layer_infrastructure
        )

    def weights(self) -> Iterator[ndarray]:
        for layer in self.layers:
            yield layer.weights

    def biases(self) -> Iterator[ndarray]:
        for layer in self.layers:
            yield layer.biases

    def dweights(self) -> Iterator[ndarray]:
        for layer in self.layers:
            yield layer.dweights

    def dbiases(self) -> Iterator[ndarray]:
        for layer in self.layers:
            yield layer.dbiases

    def forward(self, inputs: ndarray) -> ndarray:
        output = inputs

        for layer in self.layers:
            output = layer(output)

        return output

    def backward(self, output_gradient: ndarray) -> None:
        for layer in reversed(self.layers):
            output_gradient = layer.backward(output_gradient)

    @staticmethod
    def loss(y_pred: ndarray, y_actual: ndarray, loss_func: Loss) -> float64:
        return loss_func.forward(y_pred, y_actual)

    @staticmethod
    def loss_gradient(y_pred: ndarray, y_actual: ndarray, loss_func: Loss) -> ndarray:
        return loss_func.backward(y_pred, y_actual)

    def update(self, learning_rate: float) -> None:
        for layer in self.layers:
            layer.update(learning_rate)

    def zero_grad(self) -> None:
        for layer in self.layers:
            layer.zero_grad()