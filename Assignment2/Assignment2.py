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
filenames_test = 'test_batch'


# Define functions/classes (if any)
def main():

    lambda_ = 0
    n_batch = 100
    eta = 0.001
    n_epochs = 5
    GDparams = [n_batch, eta, n_epochs]
    file_name_attribution = set_file_name_attribution(lambda_, n_epochs, n_batch, eta)
    recompute = False

    X_train, Y_train, y_train = load_batch(filenames_train[0])
    X_val, Y_val, y_val = load_batch(filenames_train[1])
    X_test, Y_test, y_test = load_batch(filenames_test)

    X_train = preprocess_images(X_train)
    X_val = preprocess_images(X_val)
    X_test = preprocess_images(X_test)

    initialize_parameters(Y_train.shape[0], X_train.shape[0])

    mini_batch_gradient_descent(X_train, Y_train, X_val, Y_val, GDparams, lambda_, file_name_attribution)


    ComputeGradsNumSlow(X_train[0:20, 0:10], Y_train[:, 0:10], )


    plot_loss_or_cost(n_epochs, "loss", file_name_attribution)
    plot_loss_or_cost(n_epochs, 'cost', file_name_attribution)

    # visualize_W(n_epochs, file_name_attribution)

    # difference_between_gradient_calculation(X_train[0:20, 0:25], Y_train[:, 0:25], lambda_, n_epochs, file_name_attribution)
    # plot_gradient_difference(X_train[0:20, 0:25], Y_train[:, 0:25], lambda_, file_name_attribution)

    print("Hello, world!")


def plot_gradient_difference(X, Y, lambda_, file_name_attribution):
    gradient_differences = []
    epochs = range(1, 40)
    for epoch in epochs:
        e = '{:03d}'.format(epoch)
        grad_W_slow = np.load(f'Weights/ComputeSlow/W__{file_name_attribution}__epoch{e}.npy')
        grad_W_slow = grad_W_slow[0:Y.shape[0], 0:X.shape[0]]

        W = np.load(f'Weights/W__{file_name_attribution}__epoch{e}.npy')
        b = np.load(f'Weights/b__{file_name_attribution}__epoch{e}.npy')
        Y_estimated = evaluate_classifier(X)
        compute_gradients(X, Y, Y_estimated, W[0:Y.shape[0], 0:X.shape[0]], lambda_)

        gradient_difference = np.abs(grad_W_slow - grad_W).mean()
        gradient_differences.append(gradient_difference)

    # Plotting the gradient differences
    plt.plot(epochs, gradient_differences, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Gradient Difference')
    plt.title('Difference Between Gradients for Each Epoch')
    plt.grid(True)
    plt.savefig(f'Plots/gradient_difference__{file_name_attribution}.png')
    plt.show()


def difference_between_gradient_calculation(X, Y, lambda_, n_epochs, file_name_attribution):
    for epoch in range(1, 40):
        print(f"gradient calculation {epoch}/40")
        epoch_ = '{:03d}'.format(epoch)
        W = np.load(f'Weights/W__{file_name_attribution}__epoch{epoch_}.npy', allow_pickle=True)
        b = np.load(f'Weights/b__{file_name_attribution}__epoch{epoch_}.npy', allow_pickle=True)
        grad_W, grad_b = ComputeGradsNumSlow(X, Y, W[0:Y.shape[0], 0:X.shape[0]], b[0:Y.shape[0]], lambda_)
        np.save(f'Weights/ComputeSlow/W__{file_name_attribution}__epoch{epoch_}.npy', grad_W)
        np.save(f'Weights/ComputeSlow/b__{file_name_attribution}__epoch{epoch_}.npy', grad_b)



def set_file_name_attribution(lambda_, n_epochs, n_batch, eta):
    lambda_ = '{:01.4f}'.format(lambda_)
    lambda_ = lambda_.replace('.', '_')

    n_epochs = '{:03d}'.format(n_epochs)

    n_batch = '{:03d}'.format(n_batch)

    eta = '{:03.3f}'.format(eta)
    eta = eta.replace('.', '_')

    file_name_attribution = f"lamda{lambda_}-nepochs{n_epochs}-nbatch{n_batch}-eta{eta}"
    return file_name_attribution


def shuffle_batch(X, Y):
    # Generate a random permutation of indices
    indices = np.random.permutation(X.shape[1])

    # Shuffle both input features and target labels based on the same permutation
    X_shuffled = X[:, indices]
    Y_shuffled = Y[:, indices]

    return X_shuffled, Y_shuffled


def plot_loss_or_cost(n_epochs, label, file_name_attribution):
    y1 = np.load(f'LossCostAccuracy/{label}_test__{file_name_attribution}.npy')
    y2 = np.load(f'LossCostAccuracy/{label}_val__{file_name_attribution}.npy')
    x = range(1, len(y1)+1)

    # Plot the training
    plt.plot(x, y1, marker='o', linestyle='-', label=f'Training {label}')
    # Plot the validation
    plt.plot(x, y2, marker='o', linestyle='-', label=f'Validation {label}')

    # Add labels and title
    plt.xlabel('epoch')
    plt.ylabel(f'{label}')
    plt.title(f'{label} per epoch')

    # Show legend
    plt.legend()

    # Show grid
    plt.grid(True)

    file_name = f'Plots/{label}__{file_name_attribution}.png'

    # Save the plot as an image file (e.g., PNG)
    plt.savefig(file_name)

    # Show plot
    plt.show()


def mini_batch_gradient_descent(X_train, Y_train, X_val, Y_val, GDparams, lambda_, file_name_attribution):
    n_batch = GDparams[0]
    eta = GDparams[1]
    n_epochs = GDparams[2]

    loss_train = []
    cost_train = []
    accuracy_train = []
    loss_val = []
    cost_val = []
    accuracy_val = []

    for epoch in range(n_epochs):
        X_train, Y_train = shuffle_batch(X_train, Y_train)

        for j in range(int(X_train.shape[1]/n_batch)):
            start = j * n_batch
            end = (j+1) * n_batch - 1
            X_batch = X_train[:, start:end]
            Y_batch = Y_train[:, start:end]

            compute_gradients(X_batch, Y_batch, lambda_)
            update_weights(eta)

        loss_train.append(compute_loss(X_train, Y_train))
        cost_train.append(compute_cost(X_train, Y_train, lambda_))
        accuracy_train.append(compute_accuracy(X_train, Y_train))
        loss_val.append(compute_loss(X_val, Y_val))
        cost_val.append(compute_cost(X_val, Y_val, lambda_))
        accuracy_val.append(compute_accuracy(X_val, Y_val))

        print(f'epoch: {epoch+1}/{n_epochs}')
        print(f'accuracy_train: {accuracy_train[-1]}')
        print(f'accuracy_validation: {accuracy_val[-1]}')

    save_loss_cost_accuracy_test_and_validation(file_name_attribution, loss_train, cost_train, accuracy_train, loss_val, cost_val, accuracy_val)

    copy_weights(file_name_attribution)

    return


def copy_weights(file_name_attribution):
    shutil.copy('Weights/W1.npy', f'Weights/W/W1__{file_name_attribution}.npy')
    shutil.copy('Weights/W2.npy', f'Weights/W/W2__{file_name_attribution}.npy')
    shutil.copy('Weights/b1.npy', f'Weights/b/b1__{file_name_attribution}.npy')
    shutil.copy('Weights/b2.npy', f'Weights/b/b2__{file_name_attribution}.npy')


def save_loss_cost_accuracy_test_and_validation(file_name_attribution, loss_test, cost_test, accuracy_test, loss_val, cost_val, accuracy_val):
    np.save(f'LossCostAccuracy/loss_test__{file_name_attribution}.npy', loss_test)
    np.save(f'LossCostAccuracy/cost_test__{file_name_attribution}.npy', cost_test)
    np.save(f'LossCostAccuracy/accuracy_test__{file_name_attribution}.npy', accuracy_test)
    np.save(f'LossCostAccuracy/loss_val__{file_name_attribution}.npy', loss_val)
    np.save(f'LossCostAccuracy/cost_val__{file_name_attribution}.npy', cost_val)
    np.save(f'LossCostAccuracy/accuracy_val__{file_name_attribution}.npy', accuracy_val)


def update_weights(eta):
    W1_grad = np.load('Gradients/W1_grad.npy')
    W2_grad = np.load('Gradients/W2_grad.npy')
    b1_grad = np.load('Gradients/b1_grad.npy')
    b2_grad = np.load('Gradients/b2_grad.npy')

    W1 = np.load('Weights/W1.npy')
    W2 = np.load('Weights/W2.npy')
    b1 = np.load('Weights/b1.npy')
    b2 = np.load('Weights/b2.npy')

    W1 -= eta * W1_grad
    W2 -= eta * W2_grad
    b1 -= eta * b1_grad
    b2 -= eta * b2_grad

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


def function():
    pass


def compute_gradients(X, Y, lambda_):
    W1 = np.load('Weights/W1.npy')
    W2 = np.load('Weights/W2.npy')
    b1 = np.load('Weights/b1.npy')
    b2 = np.load('Weights/b2.npy')

    N = X.shape[1]

    Y_estimated = evaluate_classifier(X)

    g = -(Y - Y_estimated)
    W2_grad = g @ np.maximum(0, W1 @ X + b1).T / N + 2 * lambda_ * W2
    b2_grad = np.sum(g, axis=1).reshape(10, 1) / N
    g = W2.T @ g
    g = g * (W1 @ X + b1 > 0)
    W1_grad = g @ X.T / N + 2 * lambda_ * W1
    b1_grad = np.sum(g, axis=1).reshape(50, 1) / N
    # return grad_W_1, grad_b_1, grad_W_2, grad_b_2

    np.save('Gradients/W1_grad.npy', W1_grad)
    np.save('Gradients/W2_grad.npy', W2_grad)
    np.save('Gradients/b1_grad.npy', b1_grad)
    np.save('Gradients/b2_grad.npy', b2_grad)

    return


def compute_accuracy(X, Y):
    Y_estimated = evaluate_classifier(X)
    y_estimated = np.argmax(Y_estimated, axis=0)
    y_true = np.argmax(Y, axis=0)

    correct_labeled = np.sum(y_true.flatten() == y_estimated.flatten())
    accuracy = correct_labeled / Y.shape[1]

    return accuracy


def compute_loss(X, Y_true):
    return compute_cost(X, Y_true, 0)


def compute_cost(X, Y_true, lambda_, epsilon=0.000001):
    Y_estimated = evaluate_classifier(X)
    # Add a small constant to predicted probabilities to avoid taking the log of zero
    Y_estimated = epsilon if np.any(np.abs(Y_estimated)) < epsilon else Y_estimated

    W1 = np.load('Weights/W1.npy')
    W2 = np.load('Weights/W2.npy')

    loss_cross_entropy = (1 / Y_true.shape[1]) * -np.sum(Y_true * np.log(Y_estimated))
    loss_regularization = lambda_ * np.sum(np.square(W1[:, 0:X.shape[0]])) * np.sum(np.square(W2))
    cost = loss_cross_entropy + loss_regularization

    return cost


def evaluate_classifier(X):
    W1 = np.load('Weights/W1.npy')
    W2 = np.load('Weights/W2.npy')
    b1 = np.load('Weights/b1.npy')
    b2 = np.load('Weights/b2.npy')

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
    with open('../Dataset/cifar-10-batches-py/' + filename, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')

        # Use 'b' prefix for byte strings
        X = data_dict[b'data']
        X = X.T
        y = data_dict[b'labels']
        y = np.array(y)
        Y = np.eye(10)[y]
        Y = Y.T

        return X, Y, y


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
    W1 = np.random.randn(hidden_layers, d) * (1/np.sqrt(d))
    W2 = np.random.randn(K, hidden_layers) * (1/np.sqrt(hidden_layers))
    # Initialize b with Gaussian random values
    b1 = np.random.randn(hidden_layers, 1) * 0.001
    b2 = np.random.randn(K, 1) * 0.001
    b1 = np.array(b1)

    np.save('Weights/W1.npy', W1)
    np.save('Weights/W2.npy', W2)
    np.save('Weights/b1.npy', b1)
    np.save('Weights/b2.npy', b2)

    return


if __name__ == "__main__":
    main()
    sys.exit()
