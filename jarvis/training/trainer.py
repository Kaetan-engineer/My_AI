import numpy as np
import json
from pathlib import Path
from numpy import ndarray, float64
from jarvis.model.neural_network import NeuralNetwork
from jarvis.model.loss import Loss
from sklearn.utils import shuffle


class Trainer:
    def __init__(self, model: NeuralNetwork, loss_func: Loss, learning_rate: float, batch_size: int) -> None:
        self.model = model
        self.loss_func = loss_func
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def train(self, x: ndarray, y: ndarray, epochs: int, /, visuals: bool=False) -> None:
        X_train, y_train = x.copy(), y.copy()
        num_samples = len(x)
        for epoch in range(epochs):
            total_loss: float = 0.0
            predictions = []
            x_shuffled, y_shuffled = shuffle(X_train, y_train)

            for start in range(0, num_samples, self.batch_size):
                end = start + self.batch_size

                x_batch, y_batch = x_shuffled[start:end], y_shuffled[start:end]

                prediction = self.model.forward(x_batch)
                predictions.append(prediction)
                loss = self.model.loss(prediction, y_batch, self.loss_func)

                total_loss += loss * len(x_batch)

                loss_gradient = self.model.loss_gradient(prediction, y_batch, self.loss_func)

                self.model.zero_grad()
                self.model.backward(loss_gradient)
                self.model.update(self.learning_rate)

            avg_loss = total_loss / num_samples

            if visuals:
                full_pred = self.model.forward(X_train)
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.20f}")
                print(f"Predictions:\n{full_pred}\n")

    def save(self, filename: str="parameters.json") -> None:
        if not Path(filename).suffix.lower() == ".json":
            raise ValueError("File path must have a JSON suffix.")

        layers: list[dict[str, int | dict[str, list]]] = []

        for i, layer in enumerate(self.model.layers, start=1):
            layer_param = {
                "layer_number": i,
                "parameters": {
                    "weights": layer.weights.tolist(),
                    "biases": layer.biases.tolist(),
                    "dweights": layer.dweights.tolist(),
                    "dbiases": layer.dbiases.tolist()
                }
            }

            layers.append(layer_param)

        with open(filename, "w") as file:
            json.dump(layers, file, indent=4)

    def load(self, filename: str= "parameters.json") -> None:
        if not Path(filename).is_file():
            raise FileExistsError(f"Path file {filename} does not exist.")

        with open(filename, "r") as file:
            params = json.load(file)

        if not isinstance(params, list):
            raise ValueError("Retrieves content does not match Pydantic model.")

        for i, (layer, saved_layers) in enumerate(zip(self.model.layers, params, strict=True), start=1):
            if not i == saved_layers["layer_number"]:
                raise ValueError("Layer number does not match iteration number.")
            saved_params = saved_layers["parameters"]

            layer.weights = np.array(saved_params["weights"])
            layer.biases = np.array(saved_params["biases"])
            layer.dweights = np.array(saved_params["dweights"])
            layer.dbiases = np.array(saved_params["dbiases"])
