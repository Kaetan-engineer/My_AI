import numpy as np
from training.trainer import Trainer, NeuralNetwork
from model.activation import ReLU, Sigmoid, Linear, LeakyReLU, Softmax
from model.optimizers import Adam, Momentum, SGD
from model.loss import MeanSquareError, BinaryCrossEntropy, CategoricalCrossEntropy
from data import fetch_iris_data, fetch_breast_cancer_data

X_train, X_test, y_train, y_test = fetch_breast_cancer_data(get_test=True)

model = NeuralNetwork([
    (30, 32, ReLU(), "He"),
    (32, 16, ReLU(), "He"),
    (16, 1, Sigmoid(), "Xavier")
])


trainer = Trainer(
    model=model,
    loss_func=BinaryCrossEntropy(),
    optimizer=Adam(),
    learning_rate=1e-3,
    batch_size=15
)

#trainer.load()

trainer.train(X_train, y_train, 10000, visuals=True, save_between_epochs=None, visual_record_interval=100, min_loss_value=None)

trainer.save()

#trainer.test(X_test, y_test, "Categorical")

