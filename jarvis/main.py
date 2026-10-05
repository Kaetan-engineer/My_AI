import numpy as np
from training.trainer import Trainer, NeuralNetwork
from model.activation import ReLU, Sigmoid, Linear, LeakyReLU
from model.loss import MSE

X = np.array([
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0]
])

y = np.array([
    [0.5],
    [0.75],
    [5.0 / 6.0]
])


model = NeuralNetwork([
    (2, 4, ReLU(), "Xavier"),
    (4, 1, Sigmoid(), "Xavier")
])

trainer = Trainer(
    model=model,
    loss_func=MSE(),
    learning_rate=1e-3,
    batch_size=1
)

trainer.load()

trainer.train(X, y, 100000, visuals=True)

trainer.save()

