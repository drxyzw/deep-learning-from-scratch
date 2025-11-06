import numpy as np
import numbers

class SGD:
    def __init__(self, lr = 0.01):
        self.lr = lr
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]

class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
    def update(self, params, grads):
        if self.v == None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
        for key in params.keys():
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]
            params[key] += self.v[key]

class AdaGrad:
    def __init__(self, lr = 0.01):
        self.lr = lr
        self.h = None
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / np.sqrt(self.h[key] + 1.e-7)

class RMSProp: # root of mean square prop
    def __init__(self, lr = 0.01, decay_rate=0.0):
        self.lr = lr
        self.dr = decay_rate
        self.h = None
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        for key in params.keys():
            self.h[key] += -self.dr * self.h[key] + self.dr * grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / np.sqrt(self.h[key] + 1.e-7)

class Adam: # Adaptive Momentum estimation
    def __init__(self, lr = 0.01, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.v = None
        self.h = None
        self.iter = 0
    def update(self, params, grads):
        self.iter += 1
        if self.h is None:
            self.v = {}
            self.h = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
                self.h[key] = np.zeros_like(val)
        for key in params.keys():
            self.v[key] += -self.beta1 * self.v[key] + self.beta1 * grads[key]
            self.h[key] += -self.beta2 * self.h[key] + self.beta2 * grads[key] * grads[key]
            v_eff = self.v[key] / (1. - self.beta1**self.iter)
            h_eff = self.h[key] / (1. - self.beta2**self.iter)
            params[key] -= self.lr * v_eff / np.sqrt(h_eff + 1.e-7)

if __name__ == "__main__":
    import numpy as np
    from dataset.mnist import load_mnist
    from ch05.code_5_7_2_backward_propabgation import TwoLayerNet
    import pandas as pd


    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    input_size=784
    hidden_size=50
    output_size=10
    weight_init_std_type=0.
    # weight_init_std_type=0.01
    # weight_init_std_type="Xavier"
    # weight_init_std_type="He"
    if isinstance(weight_init_std_type, numbers.Number):
        weight_init_std = weight_init_std_type
    elif weight_init_std_type.upper() == "XAVIER":
        weight_init_std = {}
        weight_init_std["W1"] = 1. / np.sqrt(input_size)
        weight_init_std["W2"] = 1. / np.sqrt(hidden_size)
    elif weight_init_std_type.upper() == "HE":
        weight_init_std = {}
        weight_init_std["W1"] = np.sqrt(2. / input_size)
        weight_init_std["W2"] = np.sqrt(2. / hidden_size)

    network = TwoLayerNet(input_size=input_size, hidden_size=hidden_size, output_size=output_size, weight_init_std=weight_init_std)

    iters_num = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    iter_per_epoch = max(train_size / batch_size, 1.)


    # optimizer = SGD(lr=learning_rate)
    # optimizer = Momentum(lr=learning_rate)
    # optimizer = AdaGrad(lr=learning_rate)

    optimizer = SGD(lr = 0.95)
    # optimizer = Momentum(lr = 0.1, momentum=0.9)
    # optimizer = AdaGrad(lr = 0.03)
    # optimizer = RMSProp(lr = 0.005, decay_rate=0.05)
    # optimizer = Adam(lr = 0.005, beta1=0.9, beta2=0.99)

    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        grad = network.gradient(x_batch, t_batch)
        optimizer.update(network.params, grad)

        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print(train_acc, test_acc)
        else:
            train_acc_list.append(np.nan)
            test_acc_list.append(np.nan)


    # summarize output in a dataframe to export it to a file
    df_output = pd.DataFrame({"loss": train_loss_list, "train_acc": train_acc_list, "test_acc": test_acc_list})
    df_output.to_excel(f"./ch06/learning_backward_{optimizer.__class__.__name__}_weight_init_{weight_init_std_type}.xlsx", index=True)
