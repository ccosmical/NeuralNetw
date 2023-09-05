import numpy as np

np.random.seed(0)

X= [[1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, 0.8]]

class Layer_Dense:

    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output= np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Activation_Softmax:

    def forward(self, inputs):
        exp_values = np.exp(exp_values - np.max(inputs, axis=1, keepdims=True))
        norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = norm_values #norm_values was called probability distrubition


class Loss:            # this class is a common loss class for every kind of loss calculation
    def calculate(self, output, y):
        sample_losses= self.forward(output, y) #forward method is gonna be defined in the certain loss function class that you will choose which is the one below for us
        batch_loss= np.mean(sample_losses)    #sample_losses was a vector of losses so we want just 1 value
        return batch_loss

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape)==1:
            correct_confidences = y_pred_clipped [range(samples), y_true]
        elif len(y_true.shape)==2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

        


layer1 = Layer_Dense(4,5)
activation1 = Activation_ReLU()
print(layer1.weights)
layer1.forward(X)
#print(layer1.output)

activation1.forward(layer1.output)
#print(activation1.output)