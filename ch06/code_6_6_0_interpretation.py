import pandas as pd
import numpy as np
import numbers
import os
from multiprocessing import Pool
import matplotlib.pyplot as plt

from dataset.mnist import load_mnist

from ch06.code_6_2_1_initial_weight import *
from ch05.code_5_7_2_backward_propabgation import MultiLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
# for accerelation
# x_train = x_train[:500]
# t_train = t_train[:500]
# divide training data to training and validation data
idx_train_shuffle = list(range(x_train.shape[0]))
np.random.shuffle(idx_train_shuffle)
x_train, t_train = x_train[idx_train_shuffle], t_train[idx_train_shuffle]
validation_data_rate = 0.2
n_val = int(validation_data_rate * x_train.shape[0])
x_val = x_train[:n_val]
t_val = t_train[:n_val]
x_train = x_train[n_val:]
t_train = t_train[n_val:]
def __train(lr, weight_decay):
    input_size=784
    hidden_size=50 # for overfitting experiment
    output_size=10
    n_layers=5 # for overfitting experiment
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
    network = MultiLayerNet(n_layers=n_layers,
                            input_size=input_size, hidden_size=hidden_size, output_size=output_size, weight_init_std=weight_init_std,
                           use_batch_norm=use_batch_norm, activation="ReLU", weight_decay=weight_decay)
    iters_num = 10000 # 1 mil iterations for test
    train_size = x_train.shape[0]
    batch_size = 100
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    iter_per_epoch = int(max(train_size / batch_size, 1))
    optimizer = SGD(lr = lr)
    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
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
        # else:
        #     train_acc_list.append(np.nan)
        #     test_acc_list.append(np.nan)
    return train_acc_list, test_acc_list, network.params

if __name__=="__main__":
    # optimizing hyperparameter
    hyperparam_trial = 1 # 100
    results_val = {}
    results_train = {}
    train_accs = []
    val_accs = []
    lrs = []
    weight_decays = []
    results = []
    hyperparams = []
    params = {}
    # for _ in range(hyperparam_trial):
    #     # set random hyperparameters
    #     weight_decay = 10 ** np.random.uniform(-8, -4)
    #     lr = 10 ** np.random.uniform(-6, -2)
    #     hyperparam = (lr, weight_decay)
    #     hyperparams.append(hyperparam)
    lr = 0.00995634412233812
    weight_decay = 4.17889769878053E-08
    hyperparam = (lr, weight_decay)
    hyperparams.append(hyperparam)

    TRAIN_NETWORK = False
    directory = "./ch06/interpretation"

    if TRAIN_NETWORK:
        with Pool(processes=os.cpu_count()-1) as p:
            results = p.starmap(__train, hyperparams)
        # results.append(result)
        for i, result in enumerate(results):
            lr, weight_decay = hyperparams[i]
            train_acc_list, val_acc_list, params = result
            key = "lr: " + str(lr) + ", weight decay: " + str(weight_decay)
            print("val acc: " + str(val_acc_list[-1]) + " | " + key)
            results_train[key] = train_acc_list
            results_val[key] = val_acc_list
            lrs.append(lr)
            weight_decays.append(weight_decay)
            train_accs.append(train_acc_list[-1])
            val_accs.append(val_acc_list[-1])
            for key, param in params.items():
                df = pd.DataFrame(param)
                os.makedirs(directory, exist_ok=True)
                df.to_excel(f"{directory}/{key}.xlsx", index=False)
    else:
        for filename in os.listdir(directory):
            ext = os.path.splitext(filename)[-1]
            if ext.upper() == ".XLSX":
                key = os.path.splitext(filename)[0] # remove extension
                fullpath = os.path.join(directory, filename)
                df = pd.read_excel(fullpath)
                df_values = df.values
                params[key] = df_values

    for key, value in sorted(params.items()):
        if key.startswith("W"): # only interested in weight, not in bias "b"
            svd = np.linalg.svd(value)
            U = svd.U
            S = svd.S
            Vh = svd.Vh
            rnk = S.shape[0]
            singular_input_vecs = []
            singular_output_vecs = []
            singular_values = []
            for r in range(rnk):
                singular_input_vec = U[:, r]
                singular_input_vecs.append(singular_input_vec)
                singular_output_vec = Vh[r]
                singular_output_vecs.append(singular_output_vec)
            singular_values = S
            print(f"svd for {key} is done")

            # sort by absolute value of singular value
            idx = np.argsort(-np.abs(singular_values))
            singular_values = singular_values[idx]
            singular_input_vecs = np.array(singular_input_vecs)[idx]
            singular_output_vecs = np.array(singular_output_vecs)[idx]

            # Chart
            graph_draw_num = rnk
            col_num = 10
            row_num = int(np.ceil(graph_draw_num / col_num))
            font_size=8
            font_color="white"
            params = {"ytick.color" : font_color,
                      "xtick.color" : font_color,
                      "axes.labelcolor" : font_color,
                      "axes.edgecolor" : font_color,
                      "font.size": font_size,          # default text size
                      "axes.titlesize": font_size,     # title size
                      "axes.labelsize": font_size,     # axis label size
                      "xtick.labelsize": font_size,    # x-tick label size
                      "ytick.labelsize": font_size,    # y-tick label size
                      "legend.fontsize": font_size,    # legend
                     }
            plt.rcParams.update(params)
            fig, axes = plt.subplots(row_num, col_num)
            fig.set_facecolor('black')
            axes = axes.flatten()
            if key == "W1":
                for r in range(rnk):
                    print(key + " " + str(r) + "-th input vectors")
                    axes[r].set_title(f"{singular_values[r]:.2f}(No.{r+1}-in)", color="w")
                    axes[r].matshow(singular_input_vecs[r].reshape(28, 28), cmap="gray")
                    if r % col_num: axes[r].set_yticklabels([])
                    axes[r].set_xticklabels([])
                # plt.subplots_adjust(hspace=0.1)
                plt.show()
                outputfig = os.path.join(directory, f"{key}_input.png")
                plt.savefig(outputfig)
            print("== End of plot ==")