from dataset.mnist import load_mnist
import numpy as np
from PIL import Image
import copy

OUTPUT_DEBUG = False

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def softmax(x):
    if x.ndim == 1:
        x_max = np.max(x) # prevent overflow
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x)
    else:
        x_max = np.max(x, axis=1).reshape(-1, 1) # prevent overflow
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=1)[:,None]

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

def cross_entropy_error(y, t, one_hot_label = True):
    # cross entropy
    # E = - \sum_n \sum_i t_i * ln(y_i) / [# of n]
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]

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

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    parameters
    ------
    input_data: (# of data, # of channel, height, width)
    filter_h: height of filter
    filter_w: width of filter
    stride: size of stride
    pad: size of padding

    return:
    cols: 2d-array
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h) // stride +1
    out_w = (W + 2*pad - filter_w) // stride +1
    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """
    Docstring for col2im
    
    :param col: input data in col format
    :param input_shape: example (10, 1, 28, 28)
    :param filter_h: 
    :param filter_w: 
    :param stride: 
    :param padd: 
    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h) // stride +1
    out_w = (W + 2*pad - filter_w) // stride +1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)
    # max(y_max) = filter_h - 1 + stride * ((H + 2*pad - filter_h) // stride +1)
    # <= H + 2*pad + stride - 1 (equality hold under most assumptions)
    # similarly, max(x_max) <= W + 2*pad + stride - 1
    img = np.zeros(N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1)
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] = col[:, :, y, x, :, :]
    # reove pad
    return img[:, :, pad:(H+pad), pad:(W+pad)]

if __name__ == "__main__":
    x1 = np.random.rand(1, 3, 7, 7)
    col1 = im2col(x1, 5, 5, stride=1, pad=0)
    print(col1.shape)
    x2 = np.random.rand(10, 3, 7, 7)
    col2 = im2col(x2, 5, 5, stride=1, pad=0)
    print(col2.shape)
    