import numpy as np
from common.utils import softmax, cross_entropy_error, numerical_gradient

class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)
    
    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss

if __name__ == "__main__":
    net = SimpleNet()
    x = np.array([0.6, 0.8])
    t = np.array([0, 0, 1])
    print(net.W)
    def f(W):
        net.W = W.reshape(net.W.shape)
        return net.loss(x, t)
    net_W = net.W.flatten()
    dW_flatten = numerical_gradient(f, net_W)
    dW = dW_flatten.reshape(net.W.shape)
    print(dW)
