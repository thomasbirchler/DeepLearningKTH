import numpy as np

class NeuralNetwork:
    def __init__(self, layer_dims, batch_size, GDparams, seed=400, initalizer='He'):
        """
        Initializes the parameters for a neural network with variable number of layers.

        Args:
        layer_dims (list): List of integers where the ith element represents the number of neurons in the ith layer.
                          The first element is the input dimension and the last element is the output dimension.
        seed (int): Seed for the random number generator to ensure reproducibility.
        """
        np.random.seed(seed)
        self.parameters = {}
        self.layer_dim = layer_dims
        self.num_layers = len(layer_dims)
        self.batch_size = batch_size
        self.lambda_ = GDparams['lambda']
        self.running_mu = {}
        self.running_var = {}

        

        # Initialize weights and biases for each layer
        for i in range(1, self.num_layers):
            
            if initalizer == 'He':
                initializer_multiplier = np.sqrt(2 / layer_dims[i])
            elif initalizer == 'Xavier':
                initializer_multiplier = np.sqrt(1 / layer_dims[i])

            self.parameters[f'W{i}'] = np.random.randn(layer_dims[i], layer_dims[i-1]) * initializer_multiplier
            self.parameters[f'b{i}'] = np.random.randn(layer_dims[i], 1) * initializer_multiplier
            self.parameters[f'X_{i-1}'] = np.zeros((layer_dims[i-1], batch_size))
            # For batch normalization
            self.parameters[f'Z{i}'] = np.ones((layer_dims[i], batch_size))
            self.parameters[f'Z_hat{i}'] = np.ones((layer_dims[i], batch_size))
            self.parameters[f'gamma{i}'] = np.ones((layer_dims[i], 1))
            self.parameters[f'beta{i}'] = np.zeros((layer_dims[i], 1))
            self.running_mu[i] = np.zeros((layer_dims[i], 1))
            self.running_var[i] = np.ones((layer_dims[i], 1))


        self.parameters[f'X_{self.num_layers-1}'] = np.zeros((layer_dims[-1], batch_size))
        
        
    def get_parameters(self):
        """
        Returns the initialized parameters.

        Returns:
        dict: A dictionary containing the weights and biases for each layer.
        """
        return self.parameters
    
    def forward_pass(self, X, training=False):
        """
        Performs a forward pass through the neural network.

        Args:
        X (numpy.ndarray): Input data of shape (input_dim, n_samples).

        Returns:
        numpy.ndarray: Predicted probabilities of shape (output_dim, n_samples).
        """
        
        self.parameters['X_0'] = X
        num_layers = self.num_layers


        for i in range(1, num_layers-1):
            W = self.parameters[f'W{i}']
            b = self.parameters[f'b{i}']
            X = self.parameters[f'X_{i-1}']
            Z = np.dot(W, X) + b
            self.parameters[f'Z{i}'] = Z
            
            mu = np.mean(Z, axis=1, keepdims=True)
            var = np.var(Z, axis=1, keepdims=True)
            Z_norm = (Z - mu) / np.sqrt(var + 1e-8)
            self.parameters[f'Z_hat{i}'] = Z_norm
            Z_tilde = self.parameters[f'gamma{i}'] * Z_norm + self.parameters[f'beta{i}']

            self.parameters[f'X_{i}'] = np.maximum(0, Z_tilde)

            # Update running averages
            if training == True:
                self.running_mu[i] = 0.9 * self.running_mu[i] + 0.1 * mu
                self.running_var[i] = 0.9 * self.running_var[i] + 0.1 * var

        W = self.parameters[f'W{num_layers-1}']
        b = self.parameters[f'b{num_layers-1}']
        X = self.parameters[f'X_{num_layers-2}']
        Z = np.dot(W, X) + b

        Y_predicted = self.softmax(Z)

        return Y_predicted
    

    def backward_pass(self, Y_predicted, Y_true):
        grads = self.parameters.copy()
        G = -(Y_true - Y_predicted)

        for i in range(self.num_layers-1, 0, -1):
            
            if i == self.num_layers-1:
                X = self.parameters[f'X_{i-1}']
                grads[f'W{i}'] = 1/self.batch_size * np.dot(G, X.T) + (2 * self.lambda_ * self.parameters[f'W{i}'])
                grads[f'b{i}'] = 1/self.batch_size * np.sum(G, axis=1, keepdims=True)
                G = np.dot(self.parameters[f'W{i}'].T, G)
                G = G * (X > 0)
                continue

            else:
                X = self.parameters[f'X_{i-1}']
                grads[f'gamma{i}'] = np.sum(G * self.parameters[f'Z_hat{i}'], axis=1, keepdims=True) / self.batch_size
                grads[f'beta{i}'] = np.sum(G, axis=1, keepdims=True) / self.batch_size
                G = G * self.parameters[f'gamma{i}']
                G = self.batch_norm_backward(G, self.parameters[f'Z{i}'], self.running_mu[i], self.running_var[i])

                grads[f'W{i}'] = 1/self.batch_size * np.dot(G, X.T) + (2 * self.lambda_ * self.parameters[f'W{i}'])
                grads[f'b{i}'] = 1/self.batch_size * np.sum(G, axis=1, keepdims=True)
                G = np.dot(self.parameters[f'W{i}'].T, G)
                G = G * (X > 0)

                # Z = self.parameters[f'Z{i}']
                # Z_norm = (Z - self.running_mu[i]) / np.sqrt(self.running_var[i] + 1e-8)
                # dZ_tilde = G * self.parameters[f'gamma{i}']
                # dgamma = np.sum(dZ_tilde * Z_norm, axis=1, keepdims=True)
                # dbeta = np.sum(dZ_tilde, axis=1, keepdims=True)
                # dZ_norm = dZ_tilde / np.sqrt(self.running_var[i] + 1e-8)
                # G = dZ_norm
            

        #grads[f'W1'] = 1/self.batch_size * np.dot(G, self.parameters['X_0'].T) + (2 * self.lambda_ * self.parameters['W1'])
        #grads[f'b1'] = 1/self.batch_size * np.sum(G, axis=1, keepdims=True)

        return grads
    
    def batch_norm_backward(self, G, S, mu, v, epsilon=1e-8):

        n = G.shape[1]
        sigma1 = (v + epsilon) ** -0.5
        sigma2 = (v + epsilon) ** -1.5

        G1 = G * sigma1
        G2 = G * sigma2
        D = S - mu

        c = np.sum(G2 * D, axis=1, keepdims=True)

        G = G1 -  np.dot(G1, np.ones((n, n))) / n - D * c / n

        return G

    
    def update_weights(self, X, Y, eta):
        """
        Update the weights and biases of the neural network using the gradients.

        Args:
        X (numpy.ndarray): Input data of shape (input_dim, n_samples).
        Y (numpy.ndarray): True labels (one-hot encoded) of shape (output_dim, n_samples).
        eta (float): Learning rate.
        """
        Y_predicted = self.forward_pass(X, training=True)
        grads = self.backward_pass(Y_predicted, Y)

        for i in range(1, self.num_layers):
            self.parameters[f'W{i}'] -= eta * grads[f'W{i}']
            self.parameters[f'b{i}'] -= eta * grads[f'b{i}']
            self.parameters[f'gamma{i}'] -= eta * grads[f'gamma{i}']
            self.parameters[f'beta{i}'] -= eta * grads[f'beta{i}']


    

    def compute_loss_cost(self, X, Y_true):
        """
        Computes the loss and cross-entropy cost.
        
        Args:
            X (numpy.ndarray): Input data of shape (input_dim, n_samples).
            Y (numpy.ndarray): True labels (one-hot encoded), shape (output_dim, n_samples).

        Returns:
            float: Cross-entropy loss.
            float: Cross-entropy cost.
        """

        P = self.forward_pass(X)

        m = Y_true.shape[1]
        loss = -np.sum(Y_true * np.log(P + 1e-9)) / m  # Add a small value to prevent log(0)
        second_term = 0
        for i in range(1, self.num_layers-1):
            second_term += np.sum(np.square(self.parameters[f'W{i}']))
        cost = loss + self.lambda_ * second_term
        
        return loss, cost
    
    
    def compute_accuracy(self, X, Y):
        """
        Computes the accuracy of the neural network model.
        
        Args:
            X (numpy.ndarray): Input data of shape (input_dim, n_samples).
            Y (numpy.ndarray): True labels (one-hot encoded) of shape (output_dim, n_samples).
        
        Returns:
            float: Accuracy of the model as a percentage.
        """
        Y_predicted = self.forward_pass(X)
        correct = np.sum(np.argmax(Y, axis=0) == np.argmax(Y_predicted, axis=0))
        accuracy = correct / Y.shape[1] * 100
        return accuracy
    
    

    def softmax(self, x):
        for i in range(x.shape[1]):
            e_x = np.exp(x[:, i] - np.max(x[:, i]))
            x[:, i] = e_x / np.sum(e_x, axis=0)
        return x


    def compute_grads_num_slow(self, X, Y, lambda_, h=1e-5):
            """ Compute gradients numerically slowly for weights and biases over multiple batches.
            
            Args:
                X_batches (list of numpy.ndarray): List of arrays, each array is one batch of input data.
                Y_batches (list of numpy.ndarray): List of arrays, each array is one batch of labels.
                lambda_ (float): Regularization parameter.
                h (float): Small change to apply to parameters.
            
            Returns:
                dict: Dictionary containing averaged gradients for 'W' and 'b' across all batches.
            """
            
            grads = {f'W{i}': np.zeros_like(self.parameters[f'W{i}']) for i in range(1, self.num_layers)}
            grads.update({f'b{i}': np.zeros_like(self.parameters[f'b{i}']) for i in range(1, self.num_layers)})

            

            #for X, Y in zip(X_batches, Y_batches):
            
            for i in range(1, self.num_layers):
                # Gradients for biases
                b_grads = np.zeros_like(self.parameters[f'b{i}'])
                for j in range(len(b_grads)):
                    old_val = self.parameters[f'b{i}'][j, :].copy()
                    
                    self.parameters[f'b{i}'][j, 0] = old_val + h
                    _, c2 = self.forward_pass(X, Y)

                    self.parameters[f'b{i}'][j, :] = old_val - h
                    _, c1 = self.forward_pass(X, Y)

                    self.parameters[f'b{i}'][j, :] = old_val
                    b_grads[j] = (c2 - c1) / (2 * h)
                #grads[f'b{i}'] += b_grads / len(X)
                grads[f'b{i}'] = b_grads
                
                # Gradients for weights
                W_grads = np.zeros_like(self.parameters[f'W{i}'])
                for idx in np.ndindex(W_grads.shape):
                    old_val = self.parameters[f'W{i}'][idx]
                    
                    self.parameters[f'W{i}'][idx] = old_val + h
                    _, c2 = self.forward_pass(X, Y)

                    self.parameters[f'W{i}'][idx] = old_val - h
                    _, c1 = self.forward_pass(X, Y)

                    self.parameters[f'W{i}'][idx] = old_val
                    W_grads[idx] = (c2 - c1) / (2 * h)
                #grads[f'W{i}'] += W_grads / len(X)
                grads[f'W{i}'] = W_grads


            return grads
    

    def compute_grad_difference(self, ana_grad, num_grad):
        """
        Compute the relative and absolut error between the analytical and numerical gradients.
        
        Args:
            ana_grad (dict): Analytical gradients.
            num_grad (dict): Numerical gradients.
        
        Returns:
            numpy.ndarray: Relative error between the analytical and numerical gradients.
            numpy.ndarray: Absolute error between the analytical and numerical gradients.
        """
        difference_abs = {}
        difference_rel = {}

        for key in ana_grad.keys():
            if key in num_grad:
                difference_abs[key] = ana_grad[key] - num_grad[key]
                difference_rel[key] = np.abs(ana_grad[key] - num_grad[key]) / np.maximum(1e-6, np.abs(ana_grad[key]) + np.abs(num_grad[key]))
            

        print('relative error:', difference_rel)
        print('absolut error:', difference_abs)
        return difference_rel, difference_abs
    
