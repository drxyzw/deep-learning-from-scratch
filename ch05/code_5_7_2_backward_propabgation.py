from collections import OrderedDict
from ch05.code_5_4_1_Later import *

from common.gradient import *
import time

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01, seed = 137):
        np.random.seed(seed)
        hidden_size = output_size
        # initialize weight
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        # self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        # self.params['b2'] = np.zeros(output_size)

        # generating layers
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        # self.layers['Relu1'] = ReLU()
        # self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t): # x = input, t = target data
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y_score = np.argmax(y, axis=1)
        t_score = np.argmax(t, axis=1)
        ar = np.sum(y_score == t_score) / len(y_score)
        return ar

    def numerical_gradient_net(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        # grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        # grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads
    
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1.
        dout = self.lastLayer.backward(dout) 
        rev_layers = list(self.layers.values())
        rev_layers.reverse()
        for layer in rev_layers:
            dout = layer.backward(dout)
        # populate results
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        # grads['W2'] = self.layers['Affine2'].dW
        # grads['b2'] = self.layers['Affine2'].db
        return grads

if __name__ == "__main__":
    net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
    print(net.params['W1'].shape)
    print(net.params['b1'].shape)
    print(net.params['W2'].shape)
    print(net.params['b2'].shape)

    x = np.random.rand(110, 784)
    t = np.random.rand(110, 10)
    start_time = time.time()
    grads = net.numerical_gradient_net(x, t)
    end_time = time.time()
    # elapsed time: 236.48489713668823 sec for numerical_gradient()
    print(f"elapsed time: {end_time-start_time}")
    print(grads['W1'].shape)
    print(grads['b1'].shape)
    print(grads['W2'].shape)
    print(grads['b2'].shape)
