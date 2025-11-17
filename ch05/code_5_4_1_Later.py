import numpy as np
from common.utils import softmax, cross_entropy_error, OUTPUT_DEBUG

class Identity:
    def __init__(self):
        pass
    
    def forward(self, x):
        return x
    
    def backward(self, dout):
        return dout

class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None
    
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out
    
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy

class AddLayer:
    def __init__(self):
        pass
    def forward(self, x, y):
        out = x + y
        return out
    def backward(self, dout):
        dx = dout * 1.
        dy = dout * 1.
        return dx, dy

class ReLU:
    def __init__(self):
        self.mask = None
    def forward(self, x):
        self.mask = (x <= 0.)
        out = x.copy()
        out[self.mask] = 0.
        return out
    def backward(self, dout):
        dx = dout.copy()
        dx[self.mask] = 0.
        return dx

class Sigmoid:
    def __init__(self):
        self.x = None
        self.y = None
    def forward(self, x):
        out = 1./ (1. + np.exp(-x))
        self.y = out
        return out
    def backward(self, dout):
        dy = dout * self.y * (1. - self.y)
        return dy

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        if OUTPUT_DEBUG:
            np.savetxt("W1.csv", self.W, delimiter=",")
            np.savetxt("b1.csv", self.b, delimiter=",")
        self.x = None
        self.dW = None
        self.db = None
    def forward(self, x):
        self.x = x
        if OUTPUT_DEBUG:
            np.savetxt("x.csv", self.x, delimiter=",")
        out = np.dot(x, self.W) + self.b
        return out
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        # self.dW = np.dot(self.x.T, dout)
        # self.db = dout
        if dx.ndim == 1 or dx.shape[0] == 1:
            self.dW = np.dot(self.x.T, dout)
            self.db = dout
        else:
            dW = np.dot(self.x.T, dout)
            db = dout
        #     # because cross entropy is
        #     # E = - \sum_n \sum_i t_i * ln(y_i) / [# of n]
        #     # dE/dw = \sum_n [dE_n/dw] / [# of n]
            self.dW = dW / dx.shape[0]
            self.db = np.sum(db, axis=0) / dx.shape[0]
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        out = cross_entropy_error(self.y, self.t)
        return out
    def backward(self, dout = 1.):
        dx = dout * (self.y - self.t)
        return dx
        # if dx.ndim == 1:
        #     return dx
        # else:
        #     # because cross entropy is
        #     # E = - \sum_n \sum_i t_i * ln(y_i) / [# of n]
        #     # dE/dw = \sum_n [dE_n/dw] / [# of n]
        #     dx_sum = np.sum(dx, axis=0) / dx.shape[0]
        #     return dx_sum

class BatchNormalization:
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None

        # for test
        self.running_mean = running_mean
        self.running_var = running_var

        # intermediate values for backward
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None
    
    def forward(self, x, train_flag=True):
        self.input_shape = x.shape

        out = self.__forward(x, train_flag)
        return out.reshape(*self.input_shape)

    def __forward(self, x, train_flag):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if train_flag:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 1.e-7)
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            # momentum = contribution from the past
            self.running_mean = self.momentum * self.running_mean + (1. - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1. - self.momentum) * var
        else:
            xc = x - self.running_mean
            xn = x / np.sqrt(self.running_var + 1.e-7)

        out  = self.gamma * xn + self.beta
        return out
    
    def backward(self, dout):
        dx = self.__backward(dout)
        dx = dx.reshape(*self.input_shape)
        return dx
    
    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / self.std**2, axis=0)
        dvar = 1./2. * dstd / self.std

        dxc = dxc + (2. / self.batch_size) * self.xc * dvar
        dmu = -np.sum(dxc, axis=0)
        dx = dxc + dmu / self.batch_size

        self.dgamma =dgamma
        self.dbeta = dbeta
        return dx 

if __name__ == "__main__":
    apple = 100
    apple_num = 2
    tax = 0.1
    
    # layer
    mul_apple_layer = MulLayer()
    mul_tax_layer = MulLayer()

    # forwrd
    apple_price = mul_apple_layer.forward(apple, apple_num)
    price = mul_tax_layer.forward(apple_price, 1. + tax)
    print(price)

    # backward
    dprice = 1.
    dapple_price, dtax = mul_tax_layer.backward(dprice)
    dapple, dapple_num = mul_apple_layer.backward(dapple_price)
    print(dapple_price, dtax)
    print(dapple, dapple_num)

    