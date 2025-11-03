import numpy as np
from common.utils import softmax, cross_entropy_error

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
        # np.savetxt("W1.csv", self.W, delimiter=",")
        self.b = b
        # np.savetxt("b1.csv", self.b, delimiter=",")
        self.x = None
        self.dW = None
        self.db = None
    def forward(self, x):
        self.x = x
        # np.savetxt("x.csv", self.x, delimiter=",")
        out = np.dot(x, self.W) + self.b
        return out
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = dout
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

    