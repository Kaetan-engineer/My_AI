import numpy as np
from abc import ABC, abstractmethod
from numpy import ndarray, float64


class Activation(ABC):
    def __init__(self) -> None:
        self.output = None

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, inputs: ndarray) -> ndarray:
        return NotImplemented

    @abstractmethod
    def backward(self, output_gradient: ndarray) -> ndarray:
        return NotImplemented


class Linear(Activation):
    def forward(self, inputs: ndarray) -> ndarray:
        return inputs

    def backward(self, output_gradient: ndarray) -> ndarray:
        return output_gradient


class ReLU(Activation):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, inputs: ndarray) -> ndarray:
        self.output = np.maximum(0, inputs)
        return self.output

    def backward(self, output_gradient: ndarray) -> ndarray:
        return output_gradient * (self.output > 0)


class LeakyReLU(Activation):
    def __init__(self) -> None:
        super().__init__()
        self.a = 0.01
    def forward(self, inputs: ndarray) -> ndarray:
        self.output = np.maximum(self.a * inputs, inputs)

        return self.output

    def backward(self, output_gradient: ndarray) -> ndarray:
        dx = np.ones_like(self.output)
        dx[self.output >= 0] = self.a

        return output_gradient * dx


class SILU(Activation):
    def __init__(self) -> None:
        super().__init__()
        self.sigmoid = None

    def forward(self, inputs: ndarray) -> ndarray:
        self.sigmoid = Sigmoid().forward(inputs)
        self.output = inputs * self.sigmoid

        return self.output

    def backward(self, output_gradient: ndarray) -> ndarray:
        return self.sigmoid + (self.output * (1 - self.sigmoid))


class Sigmoid(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: ndarray) -> ndarray:
        print(f"Input (activation) dims: {inputs.shape}")
        self.output = 1 / (1 + np.exp(-inputs))
        print(f"Output (activation) dims: {self.output.shape}")
        print("\n")
        return self.output

    def backward(self, output_gradient: ndarray) -> ndarray:
        print(f"Output gradient dims: {output_gradient.shape}")
        loss_gradient = output_gradient * self.output * (1.0 - self.output)
        print(f"Loss gradient (activation) dims: {loss_gradient.shape}")
        print("\n")
        return loss_gradient


class Softmax(Activation):

    def forward(self, inputs: ndarray) -> ndarray:
        exp_x = np.exp(inputs - np.max(inputs, axis=-1, keepdims=True))
        self.output = exp_x / np.sum(exp_x, axis=-1, keepdims=True)

        return self.output

    def backward(self, output_gradient: ndarray) -> ndarray:
        s = self.output
        jacobian = np.einsum('bi,bj->bij', s, -s)
        diag_indices = np.arange(s.shape[-1])
        jacobian[:, diag_indices, diag_indices] += s

        return np.einsum('bi,bij->bj', output_gradient, jacobian)