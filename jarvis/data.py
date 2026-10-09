from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from numpy import ndarray
from typing import Any


def fetch_iris_data(get_test: bool=True) -> tuple[ndarray | Any, ...]:
    X, y = load_iris(return_X_y=True)
    encoder = OneHotEncoder(sparse_output=False)
    scaler = StandardScaler()
    y = y.reshape(-1, 1)
    y_encoded = encoder.fit_transform(y)

    if get_test:
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2)
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.fit(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test
    else:
        return X, y


def fetch_breast_cancer_data(get_test: bool=True):
    X, y = load_breast_cancer(return_X_y=True)
    encoder = OneHotEncoder(sparse_output=False)
    scaler = StandardScaler()
    y = y.reshape(-1, 1)
    y_encoded = encoder.fit_transform(y)

    if get_test:
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2)
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.fit(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test
    else:
        return X, y
