import sys, os
from dataset.mnist import load_mnist
import numpy as np
from PIL import Image
import pickle
from datetime import datetime

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

def get_data(normalize):
    # takes a few mins at initial call
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=normalize, one_hot_label=False)
    return x_test, t_test

# print(x_train.shape) # 60k, 786 (=28**2)
# print(t_train.shape) # 60k, label data
# print(x_test.shape)  # 10k, 786
# print(t_test.shape)  # 10k

def init_network():
    with open("ch03/sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    z3 = sofrmax(a3)
    return z3

# img = x_train[0]
# label = t_train[0]
# print(label)

# print(img.shape) # 784
# img = img.reshape(28, 28)
# print(img.shape)
# img_show(img)

x, t = get_data(normalize=True)
network = init_network()

starttime = datetime.now()
batch_size = 100
accuracy_count = 0
for i in range(0, len(x), batch_size):
    # img_show(x[i])
    y = predict(network, x[i:(i+batch_size)])
    p = np.argmax(y, axis=1)
    # if p == t[i]:
    #     accuracy_count += 1
    accuracy_count += np.sum(p == t[i:(i+batch_size)])
endtime = datetime.now()

print(f"accuracy count: {float(accuracy_count)/len(x)}") # 0.9352
print(f"elapsed time: {(endtime - starttime).total_seconds()} sec")
# ~0.4 sec without batch
# ~0.05 sec with batch=100
