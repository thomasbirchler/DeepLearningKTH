import numpy as np
from model import NeuralNetwork

def compute_grads_num_slow(self, X_batches, Y_batches, lambda_, NN, h=1e-5):
        """ Compute gradients numerically slowly for weights and biases over multiple batches.
        
        Args:
            X_batches (list of numpy.ndarray): List of arrays, each array is one batch of input data.
            Y_batches (list of numpy.ndarray): List of arrays, each array is one batch of labels.
            lambda_ (float): Regularization parameter.
            h (float): Small change to apply to parameters.
        
        Returns:
            dict: Dictionary containing averaged gradients for 'W' and 'b' across all batches.
        """
        self.parameters = NN.get_parameters()


        grads = {f'W{i}': np.zeros_like(self.parameters[f'W{i}']) for i in range(1, self.num_layers + 1)}
        grads.update({f'b{i}': np.zeros_like(self.parameters[f'b{i}']) for i in range(1, self.num_layers + 1)})


        for X, Y in zip(X_batches, Y_batches):
            for i in range(1, self.num_layers + 1):
                # Gradients for biases
                b_grads = np.zeros_like(self.parameters[f'b{i}'])
                for j in range(len(b_grads)):
                    old_val = self.parameters[f'b{i}'][j]
                    
                    self.parameters[f'b{i}'][j] = old_val + h
                    c2 = self.compute_cost(X, Y, lambda_)

                    self.parameters[f'b{i}'][j] = old_val - h
                    c1 = self.compute_cost(X, Y, lambda_)

                    self.parameters[f'b{i}'][j] = old_val
                    b_grads[j] = (c2 - c1) / (2 * h)
                grads[f'b{i}'] += b_grads / len(X_batches)
                
                # Gradients for weights
                W_grads = np.zeros_like(self.parameters[f'W{i}'])
                for idx in np.ndindex(W_grads.shape):
                    old_val = self.parameters[f'W{i}'][idx]
                    
                    self.parameters[f'W{i}'][idx] = old_val + h
                    c2 = self.compute_cost(X, Y, lambda_)

                    self.parameters[f'W{i}'][idx] = old_val - h
                    c1 = self.compute_cost(X, Y, lambda_)

                    self.parameters[f'W{i}'][idx] = old_val
                    W_grads[idx] = (c2 - c1) / (2 * h)
                grads[f'W{i}'] += W_grads / len(X_batches)

        return grads