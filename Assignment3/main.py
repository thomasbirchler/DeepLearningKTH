from data_utils import load_and_preprocess_data
#from training import train_model
from model import NeuralNetwork
from config.settings import GDparams, FILENAMES_TRAIN, FILENAMES_TEST
import numpy as np


def main():
    X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, Y_test, y_test = load_and_preprocess_data(FILENAMES_TRAIN, FILENAMES_TEST)


    #input_size = X_train.shape[0]
    #output_size = Y_train.shape[0]

    input_size = 12
    output_size = 10
    batch_size = 4

    layer_dims = [input_size, 6, 12, 23, output_size]

    nn = NeuralNetwork(layer_dims, batch_size, output_size)

    X_sample = X_train[:input_size, :batch_size]
    Y_sample = Y_train[:output_size, :batch_size]
    output = nn.forward_pass(X_sample, Y_sample)
    #print(output)

    lambda_ = 0.0

    grads_num = nn.compute_grads_num_slow(X_sample, Y_sample, lambda_)
    Y_predicted, _ = nn.forward_pass(X_sample, Y_sample)
    grads_ana = nn.backward_pass(Y_predicted, Y_sample)
    
    rel_error, abs_error = nn.compute_grad_difference(grads_ana, grads_num)

    print('relative error:', rel_error)
    print('absolut error:', abs_error)


if __name__ == "__main__":
    main()
