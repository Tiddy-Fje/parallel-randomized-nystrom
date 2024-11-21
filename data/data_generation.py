import numpy as np
import pandas as pd
import os

def pol_decay(n: int, r: int, p: int = 1):
    
    A = [1.0 for _ in range(r)] + [(2.0 + o) ** (-p) for o in range(n - r)]
    return np.diag(A)

def exp_decay(n: int, r: int, q: float = 0.25):
    
    A = [1.0 for _ in range(r)] + [(10.0) ** (-(o + 1) * q) for o in range(n - r)]
    return np.diag(A)

def rbf(data: np.ndarray, c: int, savepath: str = None):
    """
    Generate a radial basis function (RBF) matrix using input data.
    This matrix is generated using the formula exp(-||xi - xj||^2 / c^2).
    """
    data_norm = np.sum(data ** 2, axis=-1)
    A = np.exp(-(1 / c ** 2) * (data_norm[:, None] + data_norm[None, :] - 2 * np.dot(data, data.T)))

    if savepath is not None:
        A.tofile(savepath, sep=',', format='%10.f')
    return A

def read_mnist(filename: str, size: int = 784, savepath: str = None):
    
    dataR = pd.read_csv(filename, sep=',', header=None)
    n = len(dataR)
    data = np.zeros((n, size))
    labels = np.zeros((n, 1))

    for i in range(n):
        l = dataR.iloc[i, 0]
        labels[i] = int(l[0])
        l = l[2:]
        indices_values = [tuple(map(float, pair.split(':'))) for pair in l.split()]
        indices, values = zip(*indices_values)
        indices = [int(i) for i in indices]
        data[i, indices] = values

    if savepath is not None:
        data.tofile('./denseData.csv', sep=',', format='%10.f')
        labels.tofile('./labels.csv', sep=',', format='%10.f')

    return data, labels

def read_yearPredictionMSD(filename: str, size: int = 784, savepath: str = None):
    
    dataR = pd.read_csv(filename, sep=',', header=None)
    n = len(dataR)
    data = np.zeros((n, size))
    labels = np.zeros((n, 1))

    for i in range(n):
        l = dataR.iloc[i, 0]
        labels[i] = int(l[0])
        l = l[6:]
        indices_values = [tuple(map(float, pair.split(':'))) for pair in l.split()]
        indices, values = zip(*indices_values)
        indices = [int(i) for i in indices]
        data[i, indices] = values

    if savepath is not None:
        data.tofile('./denseData.csv', sep=',', format='%10.f')
        labels.tofile('./labels.csv', sep=',', format='%10.f')

    return data, labels

def generate_matrix(n: int, matrix_type: str, **kwargs):
    
    if matrix_type == 'pol_decay':
        return pol_decay(n, kwargs.get('r', 10), kwargs.get('p', 1))
    elif matrix_type == 'exp_decay':
        return exp_decay(n, kwargs.get('r', 10), kwargs.get('q', 0.25))
    elif matrix_type == 'rbf':
        data = kwargs.get('data')
        if data is None:
            raise ValueError("Data must be provided for RBF matrix generation.")
        c = kwargs.get('c', 100)
        return rbf(data, c)
    elif matrix_type == 'mnist':
        filename = kwargs.get('filename', 'path_to_mnist.csv')
        data, _ = read_mnist(filename)
        c = kwargs.get('c', 100)
        return rbf(data, c)
    elif matrix_type == 'year_prediction':
        filename = kwargs.get('filename', 'path_to_year_prediction.csv')
        data, _ = read_yearPredictionMSD(filename)
        c = kwargs.get('c', 100)
        return rbf(data, c)
    else:
        raise ValueError(f"Unknown matrix type: {matrix_type}")
