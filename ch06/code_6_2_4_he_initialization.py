from ch05.code_5_7_2_backward_propabgation import FiveLayerNet
import numpy as np
import matplotlib.pyplot as plt

node_num_per_layer = 100
init_weight_amplitude = np.sqrt(2. / node_num_per_layer) # He initialization
x = np.random.randn(1000, node_num_per_layer)

network = FiveLayerNet(input_size=node_num_per_layer, hidden_size=node_num_per_layer, output_size=node_num_per_layer,
                       weight_init_std=init_weight_amplitude, activation="ReLU")

y = network.predict(x)
activations = network.z
fig, axes = plt.subplots(1, len(activations), sharey=True)

# histogram
for i, a in activations.items():
    axes[i].set_title(str(i+1) + "-layer")
    axes[i].hist(a.flatten(), 30, range=(0,1))
plt.show()
