# Project Name: Deep Learning, Classification of CiFAR-Dataset
#
# Description: This project will train and test a one layer network with multiple outputs to classify images
# from the CIFAR-10 dataset. The network is trained using mini-batch gradient descent applied to a cost function
# that computes the cross-entropy loss of the classifier applied to the labelled training data and an L 2
# regularization term on the weight matrix.
#
# Author: Thomas Birchler
# Import necessary modules/packages
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
# for load_batch function
import pickle
import shutil

# Define global constants (if any)
# Define any global constants that you'll be using throughout your project
# For example:
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
filenames_train = (
    'data_batch_1',
    'data_batch_2',
    'data_batch_3',
    'data_batch_4',
    'data_batch_5',
)
filenames_test = ('test_batch', )


# Define functions/classes (if any)
def main():
    lambda_ = 2.15714286e-03
    n_batch = 100
    eta = 0
    eta_min = 0.00001
    eta_max = 0.1
    n_epochs = 0
    cycles = 3
    eta_s = 800
    GDparams = [n_batch, eta, n_epochs, eta_min, eta_max, cycles, eta_s, lambda_]
    file_name_attribution = set_file_name_attribution(lambda_, GDparams, eta_s*cycles)
    recompute = False


    X_all, Y_all, y_all = load_batch(filenames_train)

    X_train, Y_train, y_train = X_all[:, 0:49000], Y_all[:, 0:49000], y_all[0:49000]
    X_val, Y_val, y_val = X_all[:, 49000:50000], Y_all[:, 49000:50000], y_all[49000:50000]
    X_test, Y_test, y_test = load_batch(filenames_test)

    X_train = preprocess_images(X_train)
    X_val = preprocess_images(X_val)
    X_test = preprocess_images(X_test)

    initialize_parameters(Y_train.shape[0], X_train.shape[0])

    # l_min = 1.4e-04
    # l_max = -1
    # l_max = 7.2e-03
    # l_values = np.linspace(l_min, l_max, 8)
    # lambdas = np.power(10, l_values)
    # l = 8

    # Iterate over lambdas and perform some operation
    # i = 1
    # for l in l_values:
    #     print(f'trying following lambda: {i}/{len(l_values)}')
    #     i += 1
    # mini_batch_gradient_descent(X_train, Y_train, X_val, Y_val, GDparams, file_name_attribution, lambda_)

    # ComputeGradsNumSlow(X_train[0:20, 0:10], Y_train[:, 0:10], lambda_)

    # plot_loss_or_cost("loss", file_name_attribution)
    # plot_loss_or_cost('cost', file_name_attribution)
    # plot_accuracy(file_name_attribution)

    # plot_gradient_difference(X_train[0:20, 0:25], Y_train[:, 0:25], lambda_, file_name_attribution)

    print("Hello, world!")
    # plot_search()
    accuracy_test(X_test, Y_test)


def accuracy_test(X, Y):
    W1 = np.load('Weights/W1_final.npy')
    W2 = np.load('Weights/W2_final.npy')
    b1 = np.load('Weights/b1_final.npy')
    b2 = np.load('Weights/b2_final.npy')

    accuracy = compute_accuracy(X, Y, W1, W2, b1, b2)

    print(accuracy)


def plot_search():
    data = np.load('Search/CoarseToFine.npy')
    x = []
    y = []
    print(data)
    for i in range(data.shape[0]):
        if i % 2 == 0 and i > 15:
            x.append(data[i])
            # print(data[i])
        elif i > 15:
            y.append(data[i])

    plt.plot(x, y)
    plt.xscale('log')
    plt.xlabel('lambda')
    plt.ylabel('best accuracy')
    plt.title('best accuracy for each lambda training')
    plt.show()


def plot_update_rate(rate):
    x = range(1, len(rate) + 1)

    plt.plot(x, rate, linestyle='-', label=f'update rate')

    # Add labels and title
    plt.xlabel('update step')
    plt.ylabel('update rate')
    plt.title(f'update rate per update step')

    # Show legend
    plt.legend()

    # Show grid
    plt.grid(True)

    plt.show()


def plot_accuracy(file_name_attribution):
    label = 'accuracy'
    y1 = np.load(f'LossCostAccuracy/{label}_test__{file_name_attribution}.npy')
    y2 = np.load(f'LossCostAccuracy/{label}_val__{file_name_attribution}.npy')
    y3 = np.load(f'LossCostAccuracy/{label}_batch__{file_name_attribution}.npy')
    print(y1[-1])
    print(y2[-1])
    # print(y3[-1])
    x = range(1, len(y1) + 1)
    n = 100

    # Plot the training
    # plt.plot(x, y1, marker='o', linestyle='-', label=f'Training {label}')
    plt.plot(x[::n], y1[::n], linestyle='-', label=f'Training {label}')

    # Plot the validation
    # plt.plot(x, y2, marker='o', linestyle='-', label=f'Validation {label}')
    plt.plot(x[::n], y2[::n], linestyle='-', label=f'Validation {label}')

    # Plot the batch
    plt.plot(x[::n], y3[::n], linestyle='-', label=f'Batch {label}')

    # Add labels and title
    plt.xlabel('update step')
    plt.ylabel(f'{label}')
    plt.title(f'{label} per update step')

    # Show legend
    plt.legend()

    # Show grid
    plt.grid(True)

    file_name = f'Plots/{label}__{file_name_attribution}.png'

    # Save the plot as an image file (e.g., PNG)
    plt.savefig(file_name)

    # Show plot
    plt.show()


def set_file_name_attribution(lambda_, GDparams, update_steps):
    lambda_ = '{:01.4f}'.format(lambda_)
    lambda_ = lambda_.replace('.', '_')

    # update_steps = '{:03d}'.format(GDparams[5]*GDparams[6]*2)

    n_batch = '{:03d}'.format(GDparams[0])

    # eta = '{:03.3f}'.format(eta)
    # eta = eta.replace('.', '_')

    # file_name_attribution = f"lamda{lambda_}-update_steps{update_steps}-nbatch{n_batch}-eta{eta}"
    file_name_attribution = f"lamda{lambda_}-update_steps{update_steps}-nbatch{GDparams[0]}"

    return file_name_attribution


def shuffle_batch(X, Y):
    # Generate a random permutation of indices
    indices = np.random.permutation(X.shape[1])

    # Shuffle both input features and target labels based on the same permutation
    X_shuffled = X[:, indices]
    Y_shuffled = Y[:, indices]

    return X_shuffled, Y_shuffled


def plot_loss_or_cost(label, file_name_attribution):
    y1 = np.load(f'LossCostAccuracy/{label}_test__{file_name_attribution}.npy')
    y2 = np.load(f'LossCostAccuracy/{label}_val__{file_name_attribution}.npy')
    y3 = np.load(f'LossCostAccuracy/{label}_batch__{file_name_attribution}.npy')

    x = range(1, len(y1) + 1)
    n = 100

    # Plot the training
    # plt.plot(x, y1, marker='o', linestyle='-', label=f'Training {label}')
    plt.plot(x[::n], y1[::n], linestyle='-', label=f'Training {label}')

    # Plot the validation
    # plt.plot(x, y2, marker='o', linestyle='-', label=f'Validation {label}')
    plt.plot(x[::n], y2[::n], linestyle='-', label=f'Validation {label}')

    # Plot the batch
    # plt.plot(x, y2, marker='o', linestyle='-', label=f'Validation {label}')
    plt.plot(x[::n], y3[::n], linestyle='-', label=f'Batch {label}')

    # Add labels and title
    plt.xlabel('update step')
    plt.ylabel(f'{label}')
    plt.title(f'{label} per update step')

    # Show legend
    plt.legend()

    # Show grid
    plt.grid(True)

    file_name = f'Plots/{label}__{file_name_attribution}.png'

    # Save the plot as an image file (e.g., PNG)
    plt.savefig(file_name)

    # Show plot
    plt.show()


def mini_batch_gradient_descent(X_train, Y_train, X_val, Y_val, GDparams, file_name_attribution, lambda_):
    n_batch = GDparams[0]
    eta = GDparams[1]
    n_epochs = GDparams[2]
    cycles = GDparams[5]
    eta_s = GDparams[6]
    # lambda_ = GDparams[7]

    # eta_s = 2 * int(X_train.shape[1] / n_batch)

    loss_train = []
    cost_train = []
    accuracy_train = []
    loss_val = []
    cost_val = []
    accuracy_val = []
    loss_batch = []
    cost_batch = []
    accuracy_batch = []
    performance_best = 0

    W1 = np.load('Weights/W1.npy')
    W2 = np.load('Weights/W2.npy')
    b1 = np.load('Weights/b1.npy')
    b2 = np.load('Weights/b2.npy')

    for cycle in range(cycles):
        X_train, Y_train = shuffle_batch(X_train, Y_train)

        for step in range(int(2*eta_s)):
            start = step%(X_train.shape[1]/n_batch) * n_batch
            end = start + n_batch
            start = int(start)
            end = int(end)

            X_batch = X_train[:, start:end]
            Y_batch = Y_train[:, start:end]

            eta = cyclical_learning_rate(GDparams, step, eta_s)
            W1_grad, W2_grad, b1_grad, b2_grad = compute_gradients(X_batch, Y_batch, lambda_, W1, W2, b1, b2)

            W1, W2, b1, b2 = update_weights(eta, W1_grad, W2_grad, b1_grad, b2_grad, W1, W2, b1, b2)

            loss_train.append(compute_loss(X_train, Y_train, W1, W2, b1, b2))
            cost_train.append(compute_cost(X_train, Y_train, W1, W2, b1, b2, lambda_))
            accuracy_train.append(compute_accuracy(X_train, Y_train, W1, W2, b1, b2))

            loss_val.append(compute_loss(X_val, Y_val, W1, W2, b1, b2))
            cost_val.append(compute_cost(X_val, Y_val, W1, W2, b1, b2, lambda_))
            accuracy_validation = compute_accuracy(X_val, Y_val, W1, W2, b1, b2)
            accuracy_val.append(accuracy_validation)

            loss_batch.append(compute_loss(X_batch, Y_batch, W1, W2, b1, b2))
            cost_batch.append(compute_cost(X_batch, Y_batch, W1, W2, b1, b2, lambda_))
            accuracy_batch.append(compute_accuracy(X_batch, Y_batch, W1, W2, b1, b2))


            if accuracy_validation > performance_best:
                performance_best = accuracy_validation
                print(f'best performance: {performance_best}, step: {step}/{2*eta_s}')

            if step % (100) == 0:
                print(f'step: {step}/{2 * eta_s}')
                print(f'cycle: {cycle}/{cycles}')
                print(f'accuracy validation: {accuracy_validation}')

        print(f'accuracy_train: {accuracy_train[-1]}')
        print(f'accuracy_validation: {accuracy_val[-1]}')

    np.save('Weights/W1_final.npy', W1)
    np.save('Weights/W2_final.npy', W2)
    np.save('Weights/b1_final.npy', b1)
    np.save('Weights/b2_final.npy', b2)

    # file_name_attribution += f'l-{lambda_}'
    save_loss_cost_accuracy_test_and_validation(file_name_attribution, loss_train, cost_train, accuracy_train, loss_val,
                                                cost_val, accuracy_val, loss_batch, cost_batch, accuracy_batch)

    # save_lambda_and_performance(lambda_, performance_best)

    return


def save_lambda_and_performance(lambda_, performance):
    data = []
    try:
        data = np.load('Search/CoarseToFine.npy')
    except FileNotFoundError:
        print("File not found. Please check the file path.")

    appending_data = [lambda_, performance]
    data = np.append(data, appending_data, axis=0)

    np.save('Search/CoarseToFine.npy', data)
    return


def cyclical_learning_rate(GDparam, step, eta_s):
    eta_min = GDparam[3]
    eta_max = GDparam[4]

    if step <= eta_s:
        eta = eta_min + step / eta_s * (eta_max - eta_min)
        return eta

    elif eta_s < step:
        eta = eta_max - (step-eta_s) / eta_s * (eta_max - eta_min)
        return eta

    print('Error: no eta calculated')
    return


def save_loss_cost_accuracy_test_and_validation(file_name_attribution, loss_test, cost_test, accuracy_test, loss_val,
                                                cost_val, accuracy_val, loss_batch, cost_batch, accuracy_batch):
    np.save(f'LossCostAccuracy/loss_test__{file_name_attribution}.npy', loss_test)
    np.save(f'LossCostAccuracy/cost_test__{file_name_attribution}.npy', cost_test)
    np.save(f'LossCostAccuracy/accuracy_test__{file_name_attribution}.npy', accuracy_test)
    np.save(f'LossCostAccuracy/loss_val__{file_name_attribution}.npy', loss_val)
    np.save(f'LossCostAccuracy/cost_val__{file_name_attribution}.npy', cost_val)
    np.save(f'LossCostAccuracy/accuracy_val__{file_name_attribution}.npy', accuracy_val)
    np.save(f'LossCostAccuracy/loss_batch__{file_name_attribution}.npy', loss_batch)
    np.save(f'LossCostAccuracy/cost_batch__{file_name_attribution}.npy', cost_batch)
    np.save(f'LossCostAccuracy/accuracy_batch__{file_name_attribution}.npy', accuracy_batch)


def update_weights(eta, W1_grad, W2_grad, b1_grad, b2_grad, W1, W2, b1, b2):
    W1 -= eta * W1_grad
    W2 -= eta * W2_grad
    b1 -= eta * b1_grad
    b2 -= eta * b2_grad
    return W1, W2, b1, b2


def compute_gradients(X, Y, lambda_, W1, W2, b1, b2):
    N = X.shape[1]
    Y_estimated = evaluate_classifier(X, W1, W2, b1, b2)

    g = -(Y - Y_estimated)
    W2_grad = g @ np.maximum(0, W1 @ X + b1).T / N + 2 * lambda_ * W2
    b2_grad = np.sum(g, axis=1).reshape(10, 1) / N
    g = W2.T @ g
    g = g * (W1 @ X + b1 > 0)
    W1_grad = g @ X.T / N + 2 * lambda_ * W1
    b1_grad = np.sum(g, axis=1).reshape(50, 1) / N

    return W1_grad, W2_grad, b1_grad, b2_grad


def compute_accuracy(X, Y, W1, W2, b1, b2):
    Y_estimated = evaluate_classifier(X, W1, W2, b1, b2)
    y_estimated = np.argmax(Y_estimated, axis=0)
    y_true = np.argmax(Y, axis=0)

    correct_labeled = np.sum(y_true.flatten() == y_estimated.flatten())
    accuracy = correct_labeled / Y.shape[1]

    return accuracy


def compute_loss(X, Y_true, W1, W2, b1, b2):
    return compute_cost(X, Y_true, W1, W2, b1, b2, 0)


def compute_cost(X, Y_true, W1, W2, b1, b2, lambda_, epsilon=0.000001):
    Y_estimated = evaluate_classifier(X, W1, W2, b1, b2)
    # Add a small constant to predicted probabilities to avoid taking the log of zero
    Y_estimated = epsilon if np.any(np.abs(Y_estimated)) < epsilon else Y_estimated

    loss_cross_entropy = (1 / Y_true.shape[1]) * -np.sum(Y_true * np.log(Y_estimated))
    loss_regularization = lambda_ * np.sum(np.square(W1[:, 0:X.shape[0]])) * np.sum(np.square(W2))
    cost = loss_cross_entropy + loss_regularization

    return cost


def evaluate_classifier(X, W1, W2, b1, b2):
    s1 = np.dot(W1[:, 0:X.shape[0]], X) + b1
    h = np.maximum(0, s1)
    s = np.dot(W2, h) + b2
    p = softmax(s)
    return p


def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def load_batch(filename):
    """ Copied from the dataset website """

    X_all = np.empty((3*1024, 0))
    y_all = np.empty((0,))

    for i in range(len(filename)):
        with open('../Dataset/cifar-10-batches-py/' + filename[i], 'rb') as fo:
            data_dict = pickle.load(fo, encoding='bytes')

            # Use 'b' prefix for byte strings
            X = data_dict[b'data']
            X = X.T
            y = data_dict[b'labels']
            y = np.array(y)
            # Y = np.eye(10)[y]
            # Y = Y.T

            X_all = np.concatenate((X_all, X), axis=1)
            y_all = np.concatenate((y_all, y), axis=0)

    y_all = y_all.astype(int)
    Y = np.eye(10)[y_all]
    Y = Y.T

    return X_all, Y, y_all


def preprocess_images(X):
    # Normalization and conversion to double
    normalized_img_rgb = X / 255.0
    X = normalized_img_rgb.astype(np.float64)

    # Calculate mean along the first dimension (axis=0) to get mean X
    mean_X = np.mean(X, axis=0)
    # Calculate standard deviation along the first dimension (axis=0) with ddof=0 to get std X
    std_X = np.std(X, axis=0, ddof=0)
    # Subtracting the mean and scaling to unit variance
    X = (X - mean_X) / std_X

    return X


def initialize_parameters(K, d):
    hidden_layers = 50

    np.random.seed(400)

    # Initialize W with Gaussian random values
    W1 = np.random.randn(hidden_layers, d) * (1 / np.sqrt(d))
    W2 = np.random.randn(K, hidden_layers) * (1 / np.sqrt(hidden_layers))
    # Initialize b with Gaussian random values
    b1 = np.random.randn(hidden_layers, 1) * 0.001
    b2 = np.random.randn(K, 1) * 0.001
    b1 = np.array(b1)

    np.save('Weights/W1.npy', W1)
    np.save('Weights/W2.npy', W2)
    np.save('Weights/b1.npy', b1)
    np.save('Weights/b2.npy', b2)

    return


def ComputeGradsNumSlow(X, Y, lamda, h=0.00001):
    """ Converted from matlab code """
    W1 = np.load('Weights/W1.npy')
    W2 = np.load('Weights/W2.npy')
    b1 = np.load('Weights/b1.npy')
    b2 = np.load('Weights/b2.npy')

    W1 = W1[0:20, 0:20]
    W2 = W2[:, 0:20]
    b1 = b1[0:20, :]
    b2 = b2[:, 0:20]

    grad_W1 = np.zeros(W1.shape)
    grad_W2 = np.zeros(W2.shape)
    grad_b1 = np.zeros((b1.shape))
    grad_b2 = np.zeros((b2.shape))

    for i in range(b1.shape[0]):
        b1_try = np.array(b1)
        b1_try[i] -= h
        Y_estimated = evaluate_classifier(X)
        c1 = compute_cost(X, Y, lamda)

        b1_try = np.array(b1)
        b1_try[i] += h
        Y_estimated = evaluate_classifier(X)
        c2 = compute_cost(X, Y, lamda)

        grad_b1[i] = (c2 - c1) / (2 * h)

    for i in range(b2.shape[0]):
        b2_try = np.array(b2)
        b2_try[i] -= h
        c1 = compute_cost(X, Y, lamda)

        b2_try = np.array(b2)
        b2_try[i] += h
        c2 = compute_cost(X, Y, lamda)

        grad_b2[i] = (c2 - c1) / (2 * h)

    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            W1_try = np.array(W1)
            W1_try[i, j] -= h
            Y_estimated = evaluate_classifier(X)
            c1 = compute_cost(X, Y, lamda)

            W1_try = np.array(W1)
            W1_try[i, j] += h
            Y_estimated = evaluate_classifier(X)
            c2 = compute_cost(X, Y, lamda)

            grad_W1[i, j] = (c2 - c1) / (2 * h)
        return


if __name__ == "__main__":
    main()
    sys.exit()
