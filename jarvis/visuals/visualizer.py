import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    def __init__(self) -> None:
        self.records: dict[str, list[int | float]] = {
            "Epochs": [],
            "Losses": []
        }

    def record_losses(self, epoch: int, loss: float) -> None:
        self.records["Epochs"].append(epoch)
        self.records["Losses"].append(loss)

    def plot(self) -> None:
        plt.plot(
            self.records["Epochs"],
            self.records["Losses"]
        )

        plt.title("Neural Network Visual Training")
        plt.xlabel('Epochs')
        plt.ylabel("Losses")
        plt.grid()
        plt.show()