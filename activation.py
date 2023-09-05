import numpy as np

layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.81, 0.2],
                 [1.41, 1.051, 0.026]]


exp_values= np.exp(layer_outputs)      # or E = math.e    exp_values= layer_outputs ** E

norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)

x= np.sum(norm_values, axis=1, keepdims=True)

print(x)

#print(norm_values)                                               

