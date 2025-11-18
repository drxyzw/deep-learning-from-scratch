import numpy as np
import numbers
from ch06.code_6_2_1_initial_weight import *

if __name__ == "__main__":
    import numpy as np
    from dataset.mnist import load_mnist
    from ch05.code_5_7_2_backward_propabgation import MultiLayerNet
    import pandas as pd


    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    
    # reduce train data for overfitting experiment
    x_train = x_train[:300]
    t_train = t_train[:300]

    input_size=784
    hidden_size=100 # 50 # for overfitting experiment
    output_size=10
    n_layers= 9 #7 # 5 # for overfitting experiment
    weight_init_std_type="He"
    if isinstance(weight_init_std_type, numbers.Number):
        weight_init_std = weight_init_std_type
    elif weight_init_std_type.upper() == "XAVIER":
        weight_init_std = {}
        weight_init_std["W1"] = 1. / np.sqrt(input_size)
        for i in range(1, n_layers):
            weight_init_std[f"W{i+1}"] = 1. / np.sqrt(input_size)
    elif weight_init_std_type.upper() == "HE":
        weight_init_std = {}
        weight_init_std["W1"] = np.sqrt(2. / input_size)
        for i in range(1, n_layers):
            weight_init_std[f"W{i+1}"] = np.sqrt(2. / hidden_size)
    # network = FiveLayerNet(input_size=input_size, hidden_size=hidden_size, output_size=output_size, weight_init_std=weight_init_std,
    #                        activation="ReLU")
    use_batch_norm = False
    use_dropout=True
    dropout_ratio=0.0 # must be 0 or positive -- In training, it masks nodes to be droppped by rand > dropout ratio. In test, signal is scaled by * (1-dropout radio)
    network = MultiLayerNet(n_layers=n_layers,
                            input_size=input_size, hidden_size=hidden_size, output_size=output_size, weight_init_std=weight_init_std,
                           use_batch_norm=use_batch_norm, activation="ReLU", 
                           weight_decay=0., use_dropout=use_dropout, dropout_ratio=dropout_ratio)

    iters_num = 1000 # 1 mil iterations for test
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.01

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    iter_per_epoch = max(train_size / batch_size, 1.)


    # optimizer = SGD(lr=learning_rate)
    # optimizer = Momentum(lr=learning_rate)
    # optimizer = AdaGrad(lr=learning_rate)

    optimizer = SGD(lr = learning_rate)
    # optimizer = Momentum(lr = 0.1, momentum=0.9)
    # optimizer = AdaGrad(lr = 0.03)
    # optimizer = RMSProp(lr = 0.005, decay_rate=0.05)
    # optimizer = Adam(lr = 0.005, beta1=0.9, beta2=0.99)
    rand_gen = np.random.RandomState(42)

    for i in range(iters_num):
        batch_mask = rand_gen.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        grad = network.gradient(x_batch, t_batch)
        optimizer.update(network.params, grad)

        loss = network.loss(x_batch, t_batch, train_flag=True)
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
    df_output.to_excel(f"./ch06/multi_learning_overfitting_batch_normalization_{batch_normalization}_dropout_{str(dropout_ratio) if use_dropout else "false"}.xlsx", index=True)
