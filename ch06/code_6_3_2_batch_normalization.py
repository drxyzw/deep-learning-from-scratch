import numpy as np
import numbers
from ch06.code_6_2_1_initial_weight import *

if __name__ == "__main__":
    import numpy as np
    from dataset.mnist import load_mnist
    from ch05.code_5_7_2_backward_propabgation import MultiLayerNet, FiveLayerNet
    import pandas as pd


    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    input_size=784
    hidden_size=50
    output_size=10
    weight_init_std=0.1
    # network = FiveLayerNet(input_size=input_size, hidden_size=hidden_size, output_size=output_size, weight_init_std=weight_init_std,
    #                        activation="ReLU")
    use_batch_norm = True
    network = MultiLayerNet(n_layers=5, input_size=input_size, hidden_size=hidden_size, output_size=output_size, weight_init_std=weight_init_std,
                           use_batch_norm=use_batch_norm, activation="ReLU")

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

    optimizer = SGD(lr = 0.01)
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
    batch_normalization = "none" if not hasattr(network, "use_batch_norm") else "on" if network.use_batch_norm else "off"
    df_output.to_excel(f"./ch06/five_learning_backward_{optimizer.__class__.__name__}_batch_normalization_{batch_normalization}.xlsx", index=True)
