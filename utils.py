from dataset.mnist import load_mnist
import numpy as np
from PIL import Image
import copy

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def sofrmax(x):
    x_max = np.max(x) # prevent overflow
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x)

def img_show(img):
    img = img.reshape(28, 28)
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

def get_data(normalize, one_hot_label = False):
    # takes a few mins at initial call
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=normalize, one_hot_label=one_hot_label)
    return x_train, t_train, x_test, t_test

def sum_squared_error(x, y):
    return np.sum((x-y)**2)

def cross_entropy(t, y, one_hot_label = True):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.ndim

    if one_hot_label:
        return -np.sum(t * np.log(y + 1.e-7)) / batch_size # normalized by batch_size
    else:
        return -np.sum(np.log(y[:, t] + 1.e-7)) / batch_size # normalized by batch_size

def numerical_gradient(f, x):
    r = []
    h = 1.e-4

    for i in range(len(x)):
        h_grid = np.zeros_like(x)
        h_grid[i] = h
        x_m = x - h_grid
        x_p = x + h_grid
        dfdx = (f(x_p) - f(x_m)) / 2.  / h
        r.append(dfdx)
    return np.array(r)

def gradient_decent(f, init_x, lr = 0.01, step_num = 100, output_history=False):
    x = init_x
    xs = [copy.copy(x)]
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
        xs.append(copy.copy(x))

    if output_history:
        return x, xs
    else:
        return x


