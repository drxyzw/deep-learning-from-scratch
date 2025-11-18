from collections import OrderedDict
from ch05.code_5_4_1_Later import *

from common.gradient import *
import time
from copy import copy

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01, seed = 137):
        np.random.seed(seed)
        if np.isscalar(weight_init_std):
           w = weight_init_std
           weight_init_std = {}
           weight_init_std["W1"] = w
           weight_init_std["W2"] = w

        # hidden_size = output_size
        # initialize weight
        self.params = {}
        self.params["W1"] = weight_init_std["W1"] * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std["W2"] * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # generating layers
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = ReLU()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
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
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
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
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        return grads

class FiveLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01, seed = 137, activation="sigmoid"):
        np.random.seed(seed)
        activation_capital = activation.upper()
        if np.isscalar(weight_init_std):
           w = weight_init_std
           weight_init_std = {}
           weight_init_std["W1"] = w
           weight_init_std["W2"] = w
           weight_init_std["W3"] = w
           weight_init_std["W4"] = w
           weight_init_std["W5"] = w

        # hidden_size = output_size
        # initialize weight
        self.params = {}
        self.params["W1"] = weight_init_std["W1"] * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std["W2"] * np.random.randn(hidden_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std["W3"] * np.random.randn(hidden_size, hidden_size)
        self.params['b3'] = np.zeros(hidden_size)
        self.params['W4'] = weight_init_std["W4"] * np.random.randn(hidden_size, hidden_size)
        self.params['b4'] = np.zeros(hidden_size)
        self.params['W5'] = weight_init_std["W5"] * np.random.randn(hidden_size, output_size)
        self.params['b5'] = np.zeros(output_size)

        # generating layers
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Sigmoid1'] = Sigmoid() if activation_capital=="SIGMOID" else ReLU()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Sigmoid2'] = Sigmoid() if activation_capital=="SIGMOID" else ReLU()
        self.layers['Affine3'] = Affine(self.params['W3'], self.params['b3'])
        self.layers['Sigmoid3'] = Sigmoid() if activation_capital=="SIGMOID" else ReLU()
        self.layers['Affine4'] = Affine(self.params['W4'], self.params['b4'])
        self.layers['Sigmoid4'] = Sigmoid() if activation_capital=="SIGMOID" else ReLU()
        self.layers['Affine5'] = Affine(self.params['W5'], self.params['b5'])
        self.layers['Sigmoid5'] = Sigmoid() if activation_capital=="SIGMOID" else ReLU()
        self.lastLayer = SoftmaxWithLoss()

        self.z = {}

    def predict(self, x):
        i = 0
        c = 0
        for layer in self.layers.values():
            x = layer.forward(x)
            if i % 2 == 1:
                self.z[c] = copy(x)
                c += 1
            i += 1
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
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        grads['W3'] = numerical_gradient(loss_W, self.params['W3'])
        grads['b3'] = numerical_gradient(loss_W, self.params['b3'])
        grads['W4'] = numerical_gradient(loss_W, self.params['W4'])
        grads['b4'] = numerical_gradient(loss_W, self.params['b4'])
        grads['W5'] = numerical_gradient(loss_W, self.params['W5'])
        grads['b5'] = numerical_gradient(loss_W, self.params['b5'])
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
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        grads['W3'] = self.layers['Affine3'].dW
        grads['b3'] = self.layers['Affine3'].db
        grads['W4'] = self.layers['Affine4'].dW
        grads['b4'] = self.layers['Affine4'].db
        grads['W5'] = self.layers['Affine5'].dW
        grads['b5'] = self.layers['Affine5'].db
        return grads

class MultiLayerNet:
    def __init__(self, n_layers, input_size, hidden_size, output_size,
                 weight_init_std = 0.01, seed = 137, activation="sigmoid",
                 use_batch_norm=False, weight_decay=None, use_dropout=False, dropout_ratio=None):
        rand_gen=np.random.RandomState(seed)
        self.n_layers = n_layers
        self.use_batch_norm = use_batch_norm
        self.weight_decay = weight_decay
        self.use_dropout = use_dropout
        activation_capital = activation.upper()
        if np.isscalar(weight_init_std):
           w = weight_init_std
           weight_init_std = {}
           for i in range(n_layers):
               weight_init_std[f"W{i+1}"] = w

        # hidden_size = output_size
        # initialize weight
        self.params = {}
        self.layers = OrderedDict()
        for i in range(n_layers):
            if i == 0:
                intermediate_inlayer_size = input_size
                intermediate_outlayer_size = hidden_size
            elif i == n_layers-1:
                intermediate_inlayer_size = hidden_size
                intermediate_outlayer_size = output_size
            else:
                intermediate_inlayer_size = hidden_size
                intermediate_outlayer_size = hidden_size
            self.params[f"W{i+1}"] = weight_init_std[f"W{i+1}"] * rand_gen.randn(intermediate_inlayer_size, intermediate_outlayer_size)
            self.params[f'b{i+1}'] = np.zeros(intermediate_outlayer_size)

            self.layers[f'Affine{i+1}'] = Affine(self.params[f'W{i+1}'], self.params[f'b{i+1}'])
            if self.use_batch_norm:
                self.params[f'gamma{i+1}'] = np.ones(intermediate_outlayer_size)
                self.params[f'beta{i+1}'] = np.ones(intermediate_outlayer_size)
                self.layers[f'BatchNorm{i+1}'] = BatchNormalization(
                    self.params[f'gamma{i+1}'],
                    self.params[f'beta{i+1}']
                )
            if self.use_dropout:
                self.layers[f'Dropout{i+1}'] = Dropout(dropout_ratio)
            self.layers[f'Sigmoid{i+1}'] = Sigmoid() if activation_capital=="SIGMOID" else ReLU()
        self.lastLayer = SoftmaxWithLoss()

        self.z = {}

    def predict(self, x, train_flag):
        i = 0
        c = 0
        for layer in self.layers.values():
            if layer.__class__.__name__.upper() in ("BATCHNORMALIZATION", "DROPOUT"):
                x = layer.forward(x, train_flag)
            else:
                x = layer.forward(x)
            if layer.__class__.__name__.upper() == "AFFINE":
                self.z[c] = copy(x) # for debugging
                c += 1
            i += 1
        return x

    def loss(self, x, t, train_flag): # x = input, t = target data
        y = self.predict(x, train_flag)
        loss_wo_weight_decay = self.lastLayer.forward(y, t)

        loss_weight_decay = 0.
        if not self.weight_decay is None:
            for i in range(self.n_layers):
                loss_weight_decay += 0.5 * self.weight_decay * np.sum(self.params[f'W{i+1}'] **2)

        return loss_wo_weight_decay + loss_weight_decay

    def accuracy(self, x, t):
        y = self.predict(x, train_flag=False)
        y_score = np.argmax(y, axis=1)
        t_score = np.argmax(t, axis=1)
        ar = np.sum(y_score == t_score) / len(y_score)
        return ar

    def numerical_gradient_net(self, x, t):
        loss_W = lambda W: self.loss(x, t, train_flag=True)
        grads = {}
        for i in range(self.n_layers):
            grads[f'W{i+1}'] = numerical_gradient(loss_W, self.params[f'W{i+1}'])
            grads[f'b{i+1}'] = numerical_gradient(loss_W, self.params[f'b{i+1}'])

            if self.use_batch_norm:
                grads[f'gamma{i+1}'] = numerical_gradient(loss_W, self.params[f'gamma{i+1}'])
                grads[f'beta{i+1}'] = numerical_gradient(loss_W, self.params[f'beta{i+1}'])

        return grads
    
    def gradient(self, x, t):
        # forward
        self.loss(x, t, train_flag=True)

        # backward
        dout = 1.
        dout = self.lastLayer.backward(dout) 
        rev_layers = list(self.layers.values())
        rev_layers.reverse()
        for layer in rev_layers:
            dout = layer.backward(dout)
        # populate results
        grads = {}
        for i in range(self.n_layers):
            if self.weight_decay is None:
                grads[f'W{i+1}'] = self.layers[f'Affine{i+1}'].dW
                grads[f'b{i+1}'] = self.layers[f'Affine{i+1}'].db
            else:
                grads[f'W{i+1}'] = self.layers[f'Affine{i+1}'].dW + self.weight_decay * self.params[f'W{i+1}']
                grads[f'b{i+1}'] = self.layers[f'Affine{i+1}'].db

            if self.use_batch_norm:
                grads[f'gamma{i+1}'] = self.layers[f'BatchNorm{i+1}'].dgamma
                grads[f'beta{i+1}'] = self.layers[f'BatchNorm{i+1}'].dbeta
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
