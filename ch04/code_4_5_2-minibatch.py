from dataset.mnist import load_mnist
from ch04.code_4_5_1_twoLayerNet import TwoLayerNet
import numpy as np
from tqdm import tqdm
import csv

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []

# hyperparameters
# iters_num = 10000
iters_num = 10
train_size = x_train.shape[0]
# batch_size = 100
batch_size = 10
learning_rate = 0.1

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
for i in tqdm(range(iters_num)):
    # preparing mini batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # grad
    grad = network.numerical_gradient_net(x_batch, t_batch)
    # grad = network.gradient(x_batch, t_batch)

    # updating params
    for key in ("W1", "b1", "W2", "b2"):
        network.params[key] -= learning_rate * grad[key]

    # logging a track of learning
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append({"id": i, "loss": loss})

with open("TwoLayerNetLearning.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=['id', 'loss'])
    writer.writeheader()
    writer.writerows(train_loss_list)

