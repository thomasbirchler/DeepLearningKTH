from data_utils import load_and_preprocess_data
from model import NeuralNetwork
from config.settings import GDparams, FILENAMES_TRAIN, FILENAMES_TEST
from training import train_model
from plotting import plot_losses, plot_costs, plot_accuracies


def main():
    X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, Y_test, y_test = load_and_preprocess_data(FILENAMES_TRAIN, FILENAMES_TEST)

    input_size = X_train.shape[0] 
    input_size = 30
    output_size = Y_train.shape[0]
    batch_size = GDparams['batch_size']

    layer_dims = [input_size, 10, 10, output_size]


    nn = NeuralNetwork(layer_dims, batch_size, GDparams)



    
    #Y_predicted, _ = nn.forward_pass(X_sample, Y_sample)
    

    #loss_train, loss_val, cost_train, cost_val, accuracy_train, accuracy_val = train_model(nn, X_train, Y_train, X_val, Y_val, GDparams)

    #accuracy_train = nn.compute_accuracy(X_train, Y_train)
    #accuracy_val = nn.compute_accuracy(X_val, Y_val)
    #accuracy_test = nn.compute_accuracy(X_test, Y_test)

    # print(f'Accuracy on train set: {accuracy_train}')
    # print(f'Accuracy on validation set: {accuracy_val}')
    # print(f'Accuracy on test set: {accuracy_test}')

    # plot_losses(loss_train, loss_val, GDparams)
    # plot_costs(cost_train, cost_val, GDparams)
    # plot_accuracies(accuracy_train, accuracy_val)


    print('Finito!')


if __name__ == "__main__":
    main()
