import matplotlib.pyplot as plt
from ch06.code_6_1_0_optimization import SGD, Momentum, AdaGrad, RMSProp, Adam
import pandas as pd
import numpy as np
from copy import copy

h = 1.e-4
z = lambda x, y: x**2/20 + y**2
def grad(params):
    dout = {}
    dout["x"] = (z(params["x"]+h, params["y"]) - z(params["x"]-h, params["y"])) / (2. * h)
    dout["y"] = (z(params["x"], params["y"]+h) - z(params["x"], params["y"]-h)) / (2. * h)
    return dout
params = {"x": -7., "y": 2.}
# optimizer = SGD(lr = 0.95)
# optimizer = Momentum(lr = 0.01, momentum=0.9)
# optimizer = AdaGrad(lr = 0.5)
# optimizer = RMSProp(lr = 0.5, decay_rate=0.2)
optimizer = Adam(lr = 0.5, beta1=0.9, beta2=0.999)
dicts = [copy(params)]
for i in range(1000):
    dz = grad(params)
    optimizer.update(params, dz)
    dicts.append(copy(params))
df = pd.DataFrame(dicts)
df.to_excel("./ch06/SGD.xlsx")

x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
Z = z(X, Y)
plt.figure()
plt.contour(X, Y, Z, levels=20, colors="black")
plt.plot(df["x"], df["y"], linestyle="-", marker="o")
plt.show()
