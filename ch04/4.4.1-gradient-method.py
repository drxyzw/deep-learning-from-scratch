from utils import *
import numpy as np

x_init = np.array([-3., 4.])
f = lambda x: x[0] ** 2 + x[1]**2
x, xs = gradient_decent(f, x_init, lr = 0.1, output_history=True)

print(f"solution: {x}")

