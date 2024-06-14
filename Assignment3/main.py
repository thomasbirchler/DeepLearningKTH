from data_utils import load_and_preprocess_data
from model import NeuralNetwork
from config.settings import GDparams, FILENAMES_TRAIN, FILENAMES_TEST
from training import train_model
from plotting import plot_losses, plot_costs, plot_accuracies


def main():
    X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, Y_test, y_test = load_and_preprocess_data(FILENAMES_TRAIN, FILENAMES_TEST)

    input_size = X_train.shape[0] 
    input_size = 20
    output_size = Y_train.shape[0]
    batch_size = GDparams['batch_size']

    layer_dims = [input_size, 50, 50, output_size]


    nn = NeuralNetwork(layer_dims, batch_size, GDparams)


    grads_num = nn.compute_grads_num_slow(X_train[:input_size, :20], Y_train[:, :20], 1e-5)
    Y_predicted = nn.forward_pass(X_train[:input_size, :20])
    grads_ana = nn.backward_pass(Y_predicted, Y_train[:, :20])

    nn.compute_grad_difference(grads_num, grads_ana)

    

    # loss_train, loss_val, cost_train, cost_val, accuracy_train, accuracy_val = train_model(nn, X_train, Y_train, X_val, Y_val, GDparams)

    # accuracy_train = nn.compute_accuracy(X_train, Y_train)
    # accuracy_val = nn.compute_accuracy(X_val, Y_val)
    # accuracy_test = nn.compute_accuracy(X_test, Y_test)

    # print(f'Accuracy on train set: {accuracy_train}')
    # print(f'Accuracy on validation set: {accuracy_val}')
    # print(f'Accuracy on test set: {accuracy_test}')

    # plot_losses(loss_train, loss_val, GDparams)
    # plot_costs(cost_train, cost_val, GDparams)
    # plot_accuracies(accuracy_train, accuracy_val, GDparams)


    print('Finito!')


if __name__ == "__main__":
    main()
