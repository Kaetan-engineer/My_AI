import numpy as np
from numpy import ndarray, float64
from jarvis.model.activation import Activation
from jarvis.model.loss import Loss
from jarvis.model.layers import DenseLayer
from model.optimizers import Optimizer
from typing import Iterator


class NeuralNetwork:
    def __init__(self, layer_infrastructure: list[tuple[int, int, Activation, str | None]]) -> None:
        self.layers: list[DenseLayer] = []

        for index, (inputs, neurons, activation, initialization) in enumerate(layer_infrastructure):
            self.layers.append(DenseLayer(inputs, neurons, activation, initialization))

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

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

        for i, layer in enumerate(self.layers, start=1):
            output = layer(output)

        return output

    @property
    def last_logit(self) -> ndarray:
        return self.layers[-1].pre_activation

    def backward(self, output_gradient: ndarray) -> None:
        for layer in reversed(self.layers):
            output_gradient = layer.backward(output_gradient)

    def loss(self, y_pred: ndarray, y_actual: ndarray, loss_func: Loss) -> float64:
        if loss_func.__class__.__name__ in ["BinaryCrossEntropy", "CategoricalCrossEntropy"]:
            return loss_func.forward(self.last_logit, y_actual)

        return loss_func.forward(y_pred, y_actual)

    def loss_gradient(self, y_pred: ndarray, y_actual: ndarray, loss_func: Loss) -> ndarray:
        if loss_func.__class__.__name__ in ["BinaryCrossEntropy", "CategoricalCrossEntropy"]:
            return loss_func.backward(self.last_logit, y_actual)

        return loss_func.backward(y_pred, y_actual)

    def update(self, learning_rate: float, optimizer: Optimizer) -> None:
        for layer in self.layers:
            layer.update(learning_rate, optimizer)

    def zero_grad(self) -> None:
        for layer in self.layers:
            layer.zero_grad()