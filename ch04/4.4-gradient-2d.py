import matplotlib.pyplot as plt
import numpy as np
from utils import *

x = np.linspace(-2, 2, 17)
y = np.linspace(-2, 2, 17)

X, Y = np.meshgrid(x, y)
z = lambda x:  x[0]**2 + x[1]**2
# plot of surface
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, Y, Z=z(np.array([X, Y])))
plt.show()

# plot of gradient
grad = numerical_gradient(z, np.array([X.flatten(), Y.flatten()]))
# h = 1.-4
# U = (z(X + h, Y) - z(X - h, Y)) / 2.  / h
# V = (z(X, Y + h) - z(X, Y - h)) / 2.  / h
U = grad[0].reshape(X.shape)
V = grad[1].reshape(Y.shape)

plt.quiver(X, Y, -U, -V)
plt.show()

