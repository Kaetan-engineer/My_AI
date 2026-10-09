import numpy as np
import json
from pathlib import Path
from numpy import ndarray, float64
from jarvis.model.neural_network import NeuralNetwork, Optimizer
from jarvis.model.loss import Loss
from visuals.visualizer import Visualizer
from sklearn.utils import shuffle


class Trainer:
    def __init__(self, model: NeuralNetwork, loss_func: Loss, optimizer: Optimizer, learning_rate: float, batch_size: int) -> None:
        self.model: NeuralNetwork = model
        self.loss_func: Loss = loss_func
        self.optimizer: Optimizer = optimizer
        self.visualizer: Visualizer = Visualizer()
        self.learning_rate: float = learning_rate
        self.batch_size: int = batch_size

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def train(self,
              x: ndarray,
              y: ndarray,
              epochs: int,
              /,
              visuals: bool = False,
              save_between_epochs: int | None = None,
              visual_record_interval: int=10,
              min_loss_value: float | None=1e-10) -> None:
        X_train, y_train = x.copy(), y.copy()
        num_samples = len(x)
        for epoch in range(epochs):
            total_loss: float = 0.0
            predictions = []
            x_shuffled, y_shuffled = shuffle(X_train, y_train, random_state=42)

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
                self.model.update(self.learning_rate, self.optimizer)

            avg_loss = total_loss / num_samples

            if save_between_epochs is not None and (epoch + 1) % save_between_epochs == 0:
                self.save()

            if (1 + epoch) % visual_record_interval == 0:
                self.visualizer.record_losses(epoch, avg_loss)

            if visuals:
                full_pred = self.model.forward(X_train)
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.20f}")
                print(f"Predictions:\n{full_pred}\n")

            if min_loss_value is not None and avg_loss<= min_loss_value:
                print(f"Loss value has become too low ({avg_loss}), further training has become useless.")
                break

        if visuals:
            self.visualizer.plot()

    def save(self, filename: str = "parameters.json") -> None:
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

    def load(self, filename: str = "parameters.json") -> None:
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

    @staticmethod
    def _binary_accuracy(logits: ndarray, y_actual: ndarray) -> float:
        probabilities = 1 / (1 + np.exp(-logits))
        predictions = probabilities >= 0.5

        return np.mean(predictions == y_actual)

    @staticmethod
    def _categorical_accuracy(logits: ndarray, y_actual: ndarray) -> float:
        predictions = np.argmax(logits, axis=1)

        if y_actual.ndim == 2:
            actual = np.argmax(y_actual, axis=1)
        else:
            actual = y_actual

        return np.mean(predictions == actual)

    def test(self, x: ndarray, y: ndarray, /, loss: str) -> None:
        prediction = self(x)
        logit = self.model.last_logit

        print("Prediction:")
        print(prediction)
        print("\n")
        print("Actual")
        print(y)
        print("\n")

        match loss.lower():
            case "binary":
                accuracy = self._binary_accuracy(logit, y)
                print(f"Accuracy: {accuracy:.3%}")
            case "categorical":
                accuracy = self._categorical_accuracy(logit, y)
                print(f"Accuracy: {accuracy:.3%}")
            case _:
                print("Wrong / Invalid loss inputed")
