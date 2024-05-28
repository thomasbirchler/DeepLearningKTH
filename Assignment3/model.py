import numpy as np

class NeuralNetworkInitializer:
    def __init__(self, layer_dims, seed=400):
        """
        Initializes the parameters for a neural network with variable number of layers.

        Args:
        layer_dims (list): List of integers where the ith element represents the number of neurons in the ith layer.
                          The first element is the input dimension and the last element is the output dimension.
        seed (int): Seed for the random number generator to ensure reproducibility.
        """
        np.random.seed(seed)
        self.parameters = {}
        self.num_layers = len(layer_dims) - 1  # Number of layers excluding the input layer

        # Initialize weights and biases for each layer
        for i in range(1, self.num_layers + 1):
            self.parameters[f'W{i}'] = np.random.randn(layer_dims[i], layer_dims[i-1]) * (1 / np.sqrt(layer_dims[i-1]))
            self.parameters[f'b{i}'] = np.random.randn(layer_dims[i], 1) * 0.01
            self.parameters[f'X_{i-1}'] = np.zeros((layer_dims[i], 1))

    def get_parameters(self):
        """
        Returns the initialized parameters.

        Returns:
        dict: A dictionary containing the weights and biases for each layer.
        """
        return self.parameters
    
    def forward_pass(self, X):
        """
        Performs a forward pass through the neural network.

        Args:
        X (numpy.ndarray): Input data of shape (input_dim, n_samples).

        Returns:
        numpy.ndarray: Predicted probabilities of shape (output_dim, n_samples).
        """
        num_layers = len(self.parameters) // 2
        
        self.parameters['X_0'] = X

        for i in range(1, num_layers):
            W = self.parameters[f'W{i}']
            b = self.parameters[f'b{i}']
            X = self.parameters[f'X_{i-1}']
            Z = np.dot(W, X) + b
            self.parameters[f'X_{i}'] = np.maximum(0, Z)
        W = self.parameters[f'W{num_layers}']
        b = self.parameters[f'b{num_layers}']
        X = self.parameters[f'X_{num_layers-1}']
        Z = np.dot(W, X) + b
        return self.softmax(Z)
    
    def backward_pass(self, Y_predicted, Y_true):
        grads = self.parameters.copy()
        G = np.zeros((Y_true.shape[0], Y_true.shape[1], self.num_layers + 1))
        G[:, :, self.num_layers] = -(Y_true - Y_predicted)

        for i in range(self.num_layers, 0, -1):
            X = self.parameters[f'X_{i}']
            self.grads[f'W{i}'] = 1/self.num_layers * np.dot(G[:, :, i], X.T)
            self.grads[f'b{i}'] = 1/self.num_layers * np.sum(G[:, :, i], axis=1, keepdims=True)
            G[:, :, i - 1] = np.dot(self.parameters[f'W{i}'].T, G[:, :, i])
            G[:, :, i - 1] = G[:, :, i - 1] * (X > 0)
        self.parameters[f'W1'] = 1/self.num_layers * np.dot(G, self.parameters['X_0'].T)
        self.parameters[f'b1'] = 1/self.num_layers * np.sum(G, axis=1, keepdims=True)

        return grads




def softmax(x):
    e_x = np.exp(x - np.max(x, axis=0))
    return e_x / np.sum(e_x, axis=0)
