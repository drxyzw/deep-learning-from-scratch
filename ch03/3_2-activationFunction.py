import numpy as np

def step_function(x):
    temp = x > 0
    return temp.astype(np.int64)

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

if __name__ == "__main__":
    x = np.array([-1., 1., 2.])
    y = sigmoid(x)
    # y = step_function(x)
    print(y)
