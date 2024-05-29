from data_utils import load_and_preprocess_data
#from training import train_model
from model import NeuralNetwork
from config.settings import GDparams, FILENAMES_TRAIN, FILENAMES_TEST
import numpy as np
from config.settings import GDparams
from training import train_model


def main():
    X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, Y_test, y_test = load_and_preprocess_data(FILENAMES_TRAIN, FILENAMES_TEST)

    input_size = X_train.shape[0]
    output_size = Y_train.shape[0]
    batch_size = 64

    layer_dims = [input_size, 50, output_size]


    nn = NeuralNetwork(layer_dims, batch_size, GDparams)



    #X_sample = X_train[:input_size, :batch_size]
    #Y_sample = Y_train[:output_size, :batch_size]
    #output = nn.forward_pass(X_sample, Y_sample)
    #print(output)

    #Y_predicted, _ = nn.forward_pass(X_sample, Y_sample)
    

    train_model(nn, X_train, Y_train, X_val, Y_val, GDparams)

    print('Finito!')


if __name__ == "__main__":
    main()
