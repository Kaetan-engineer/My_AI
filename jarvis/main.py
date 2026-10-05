import numpy as np
from training.trainer import Trainer, NeuralNetwork
from model.activation import ReLU, Sigmoid, Linear, LeakyReLU, Softmax
from model.optimizers import Adam, Momentum, SGD
from model.loss import MSE

X = np.array([
    [1.0, 2.0],
    [3.0, 5.0],
    [9.0, 6.0]
])

y = np.array([
    [False],
    [True],
    [False]
])

model = NeuralNetwork([
    (2, 4, ReLU(), "He"),
    (4, 1, Linear(), "Xavier")
])

trainer = Trainer(
    model=model,
    loss_func=MSE(),
    optimizer=Momentum(),
    learning_rate=1e-3,
    batch_size=1
)

trainer.load()

trainer.train(
    X,
    y,
    10000,
    visuals=True,
    save_between_epochs=(True, 1000)
)

trainer.save()
