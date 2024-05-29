import numpy as np

class NeuralNetwork:
    def __init__(self, layer_dims, batch_size, GDparams, reg=0.0, seed=400):
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

        # Initialize weights and biases for each layer
        for i in range(1, self.num_layers):
            self.parameters[f'W{i}'] = np.random.randn(layer_dims[i], layer_dims[i-1]) * (1 / np.sqrt(layer_dims[i]))
            #self.parameters[f'b{i}'] = np.random.randn(layer_dims[i], batch_size) * 0.01
            self.parameters[f'b{i}'] = np.random.randn(layer_dims[i], 1) * 0.01
            self.parameters[f'X_{i-1}'] = np.zeros((layer_dims[i-1], batch_size))
        self.parameters[f'X_{self.num_layers-1}'] = np.zeros((layer_dims[-1], batch_size))
        
        
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
        
        self.parameters['X_0'] = X
        num_layers = self.num_layers


        for i in range(1, num_layers-1):
            W = self.parameters[f'W{i}']
            b = self.parameters[f'b{i}']
            X = self.parameters[f'X_{i-1}']
            Z = np.dot(W, X) + b
            self.parameters[f'X_{i}'] = np.maximum(0, Z)
        W = self.parameters[f'W{num_layers-1}']
        b = self.parameters[f'b{num_layers-1}']
        X = self.parameters[f'X_{num_layers-2}']
        Z = np.dot(W, X) + b

        Y_predicted = self.softmax(Z)

        return Y_predicted
    

    def backward_pass(self, Y_predicted, Y_true):
        grads = self.parameters.copy()
        G = -(Y_true - Y_predicted)

        for i in range(self.num_layers-1, 1, -1):
            X = self.parameters[f'X_{i-1}']
            # grads[f'W{i}'] = 1/self.num_layers * np.dot(G, X.T)
            # grads[f'b{i}'] = 1/self.num_layers * np.sum(G, axis=1, keepdims=True)
            grads[f'W{i}'] = 1/self.batch_size * np.dot(G, X.T)
            grads[f'b{i}'] = 1/self.batch_size * np.sum(G, axis=1, keepdims=True)
            G = np.dot(self.parameters[f'W{i}'].T, G)
            G = G * (X > 0)
        # self.parameters[f'W1'] = 1/self.num_layers * np.dot(G, self.parameters['X_0'].T)
        # self.parameters[f'b1'] = 1/self.num_layers * np.sum(G, axis=1, keepdims=True)
        self.parameters[f'W1'] = 1/self.batch_size * np.dot(G, self.parameters['X_0'].T)
        self.parameters[f'b1'] = 1/self.batch_size * np.sum(G, axis=1, keepdims=True)

        return grads
    
    def update_weights(self, X, Y, eta):
        """
        Update the weights and biases of the neural network using the gradients.

        Args:
        X (numpy.ndarray): Input data of shape (input_dim, n_samples).
        Y (numpy.ndarray): True labels (one-hot encoded) of shape (output_dim, n_samples).
        eta (float): Learning rate.
        """
        Y_predicted = self.forward_pass(X)
        grads = self.backward_pass(Y_predicted, Y)
        for i in range(1, self.num_layers):
            self.parameters[f'W{i}'] -= eta * grads[f'W{i}']
            self.parameters[f'b{i}'] -= eta * grads[f'b{i}']

    

    def compute_loss(self, X, Y_true):
        """
        Computes the cross-entropy loss.
        
        Args:
            X (numpy.ndarray): Input data of shape (input_dim, n_samples).
            Y (numpy.ndarray): True labels (one-hot encoded), shape (output_dim, n_samples).

        Returns:
            float: Cross-entropy loss.
        """

        P = self.forward_pass(X)

        m = Y_true.shape[1]
        first_term = -np.sum(Y_true * np.log(P + 1e-9)) / m  # Add a small value to prevent log(0)
        second_term = 0
        for i in range(1, self.num_layers-1):
            second_term += np.sum(np.square(self.parameters[f'W{i}']))
        loss = first_term + self.lambda_ * second_term
        
        return loss
    
    
    def compute_accuracy(self, X, Y):
        """
        Computes the accuracy of the neural network model.
        
        Args:
            X (numpy.ndarray): Input data of shape (input_dim, n_samples).
            Y (numpy.ndarray): True labels (one-hot encoded) of shape (output_dim, n_samples).
        
        Returns:
            float: Accuracy of the model as a percentage.
        """
        Y_predicted = self.forward_pass(X, Y)
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
    
