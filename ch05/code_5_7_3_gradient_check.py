import numpy as np
from dataset.mnist import load_mnist
from code_5_7_2_backward_propabgation import TwoLayerNet
import csv

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient_net(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

# average of accuracy rates
for key in grad_numerical.keys():
    diff = np.average( np.abs(grad_backprop[key] - grad_numerical[key]) )
    print(key + ": " + str(diff))
    with open(key + ".csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "backprop", "numerical"])
        for i in range(len(grad_backprop[key])):
            writer.writerow({"name": key, "backprop": grad_backprop[key][i], "numerical": grad_numerical[key][i]})

