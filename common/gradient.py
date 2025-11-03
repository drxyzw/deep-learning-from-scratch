import numpy as np
from common.utils import cross_entropy_error, sigmoid, softmax

def numerical_gradient(f, x):
    h = 1.e-4
    grad = np.zeros_like(x)

    for idx, _ in np.ndenumerate(x):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxph = f(x).copy()
        x[idx] = tmp_val - h
        fxmh = f(x).copy()
        grad[idx] = (fxph - fxmh) / (2.*h)
        x[idx] = tmp_val
    return grad

