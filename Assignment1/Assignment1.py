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

    lambda_ = 0.1
    n_batch = 100
    eta = 0.001
    n_epochs = 400
    GDparams = [n_batch, eta, n_epochs]
    file_name_attribution = set_file_name_attribution(lambda_, n_epochs, n_batch, eta)
    recompute = False

    X_train, Y_train, y_train = load_batch(filenames_train[0])
    X_val, Y_val, y_val = load_batch(filenames_train[1])
    X_test, Y_test, y_test = load_batch(filenames_test)

    X_train = preprocess_images(X_train)
    X_val = preprocess_images(X_val)
    X_test = preprocess_images(X_test)

    # X_train = X_train[0:50, 0:100]
    # Y_train = Y_train[:, 0:100]


    W, b = initialize_parameters(Y_train.shape[0], X_train.shape[0])

    W, b = mini_batch_gradient_descent(X_train, Y_train, GDparams[0], GDparams[1], GDparams[2], W, b, lambda_, file_name_attribution, recompute)

    compute_loss_and_cost_for_all_epochs(X_train, Y_train, 'train', GDparams[2], file_name_attribution, recompute)
    compute_loss_and_cost_for_all_epochs(X_val, Y_val, 'validation', GDparams[2], file_name_attribution, recompute)

    epoch_vec = list(range(0, n_epochs))

    loss_train = np.load(f'Loss/loss_train__{file_name_attribution}.npy')
    loss_val = np.load(f'Loss/loss_validation__{file_name_attribution}.npy')
    plot_loss_or_cost(epoch_vec, loss_train, loss_val, "loss", file_name_attribution)

    cost_train = np.load(f'Cost/cost_train__{file_name_attribution}.npy', allow_pickle=True)
    cost_val = np.load(f'Cost/cost_validation__{file_name_attribution}.npy', allow_pickle=True)
    plot_loss_or_cost(epoch_vec, cost_train, cost_val, "cost", file_name_attribution)

    visualize_W(n_epochs, file_name_attribution)

    accuracy = compute_accuracy(X_test, Y_test, W, b)
    print(f'Test Accuracy: {accuracy}')

    # print(X_train.shape)
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
        Y_estimated = evaluate_classifier(X, W[0:Y.shape[0], 0:X.shape[0]], b[0:Y.shape[0], :])
        grad_W, grad_b = compute_gradients(X, Y, Y_estimated, W[0:Y.shape[0], 0:X.shape[0]], lambda_)

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


def delete_existing_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)


def visualize_W(n_epoch, file_name_attribution):
    n_epoch = '{:03d}'.format(n_epoch)
    W = np.load(f'Weights/W__{file_name_attribution}__epoch{n_epoch}.npy')
    s_im = []

    for i in range(10):
        # Reshape W(i, :) to a 32x32x3 image
        im = np.reshape(W[i, :], (32, 32, 3),  order='F')

        # Normalize pixel values to range [0, 1]
        im_normalized = (im - np.min(im)) / (np.max(im) - np.min(im))

        # Rearrange dimensions to match MATLAB's default image format
        im_permuted = np.transpose(im_normalized, (1, 0, 2))

        # Append the transformed image to the list
        s_im.append(im_permuted)
        # s_im.append(im_normalized)

    # Visualize the images
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))

    for i, ax in enumerate(axes.flat):
        ax.imshow(s_im[i], interpolation='nearest')  # Display the i-th image
        ax.axis('off')  # Turn off axes
        ax.set_title(f'Class {i + 1}')  # Set title for the image

    file_name = f'Plots/weights_visualized__{file_name_attribution}.png'

    delete_existing_file(file_name)

    # Save the plot as an image file (e.g., PNG)
    plt.savefig(file_name)

    plt.show()


def set_file_name_attribution(lambda_, n_epochs, n_batch, eta):
    lambda_ = '{:01.4f}'.format(lambda_)
    lambda_ = lambda_.replace('.', '_')

    n_epochs = '{:03d}'.format(n_epochs)

    n_batch = '{:03d}'.format(n_batch)

    eta = '{:03.4f}'.format(eta)
    eta = eta.replace('.', '_')

    file_name_attribution = f"lamda{lambda_}-nepochs{n_epochs}-nbatch{n_batch}-eta{eta}"
    return file_name_attribution


def shuffle_batch(X, Y):
    """
    Shuffle the input features and corresponding target labels within a batch.

    Parameters:
        X (numpy.ndarray): Input features (batch_size, num_features).
        Y (numpy.ndarray): Target labels (batch_size, num_classes).

    Returns:
        numpy.ndarray: Shuffled input features.
        numpy.ndarray: Shuffled target labels.
    """
    # Generate a random permutation of indices
    indices = np.random.permutation(X.shape[1])

    # Shuffle both input features and target labels based on the same permutation
    X_shuffled = X[:, indices]
    Y_shuffled = Y[:, indices]

    return X_shuffled, Y_shuffled


def compute_loss_and_cost_for_all_epochs(X, Y, label, n_epoch, file_name_attribution, recompute):
    # folder_path = 'Weights/'
    # files_in_folder = os.listdir(folder_path)
    # pattern_W = 'W_epoch'
    # pattern_b = 'b_epoch'
    # W_matching_files = [filename for filename in files_in_folder if pattern_W in filename]
    # b_matching_files = [filename for filename in files_in_folder if pattern_b in filename]
    # num_matching_files = len(W_matching_files)
    # for epoch in range(num_matching_files):
    if recompute or not os.path.exists(f'Loss/loss_{label}__{file_name_attribution}.npy'):
        if os.path.exists(f'Loss/loss_{label}__{file_name_attribution}.npy'):
            os.remove(f'Loss/loss_{label}__{file_name_attribution}.npy')
        if os.path.exists(f'Cost/cost_{label}__{file_name_attribution}.npy'):
            os.remove(f'Cost/cost_{label}__{file_name_attribution}.npy')

        for epoch in range(n_epoch):
            format_epoch = '{:03d}'.format(epoch + 1)
            W = np.load(f'Weights/W__{file_name_attribution}__epoch{format_epoch}.npy')
            b = np.load(f'Weights/b__{file_name_attribution}__epoch{format_epoch}.npy')

            save_loss(X, Y, W, b, label, file_name_attribution)
            save_cost(X, Y, W, b, 0, label, file_name_attribution)


def plot_loss_or_cost(x, y1, y2, label, file_name_attribution):
    # Plot the training cost
    plt.plot(x, y1, marker='o', linestyle='-', label=f'Training {label}')
    # Plot the validation cost
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

    delete_existing_file(file_name)

    # Save the plot as an image file (e.g., PNG)
    plt.savefig(file_name)

    # Show plot
    plt.show()


def mini_batch_gradient_descent(X, Y, n_batch, eta, n_epochs, W, b, lambda_, file_name_attribution, recompute):
    file_path_W = f'Weights/W__{file_name_attribution}__epoch_000.npy'
    file_path_b = f'Weights/b__{file_name_attribution}__epoch_000.npy'
    if os.path.exists(file_path_W) and not recompute:
        format_epoch = '{:03d}'.format(n_epochs)
        file_path_W = f'Weights/W__{file_name_attribution}__epoch{format_epoch}.npy'
        file_path_b = f'Weights/b__{file_name_attribution}__epoch{format_epoch}.npy'
        W = np.load(file_path_W)
        b = np.load(file_path_b)
        return W, b
    else:
        delete_existing_file(file_path_W)
        np.save(file_path_W, W)
        delete_existing_file(file_path_b)
        np.save(file_path_b, b)

        for epoch in range(n_epochs):
            X, Y = shuffle_batch(X, Y)

            for j in range(int(X.shape[1]/n_batch)):
                start = j * n_batch
                end = (j+1) * n_batch - 1
                X_batch = X[:, start:end]
                Y_batch = Y[:, start:end]

                Y_estimated = evaluate_classifier(X_batch, W, b)
                W_gradient, b_gradient = compute_gradients(X_batch, Y_batch, Y_estimated, W, lambda_)
                W, b = update_weights(W, W_gradient, b, b_gradient, eta)

            format_epoch = '{:03d}'.format(epoch+1)
            delete_existing_file(f'Weights/W__{file_name_attribution}__epoch{format_epoch}.npy')
            np.save(f'Weights/W__{file_name_attribution}__epoch{format_epoch}.npy', W)
            delete_existing_file(f'Weights/b__{file_name_attribution}__epoch{format_epoch}.npy')
            np.save(f'Weights/b__{file_name_attribution}__epoch{format_epoch}.npy', b)

            # print("computing difference in gradients")
            # difference_between_gradient_calculation(X, Y, lambda_, n_epochs,
            #                                         file_name_attribution, epoch+1)

            print(f'epoch: {epoch+1}/{n_epochs}')
        return W, b


def save_loss(X, Y, W, b, file_name, file_name_attribution):
    Y_estimated = evaluate_classifier(X, W, b)
    loss = compute_loss(X, Y, Y_estimated, W, b)
    try:
        loss_array = np.load(f'Loss/loss_{file_name}__{file_name_attribution}.npy')
    except FileNotFoundError:
        loss_array = np.array([])
    loss_array = np.append(loss_array, loss)
    np.save(f'Loss/loss_{file_name}__{file_name_attribution}.npy', loss_array)


def save_cost(X, Y, W, b, lambda_, file_name, file_name_attribution):
    Y_estimated = evaluate_classifier(X, W, b)
    cost = compute_cost(X, Y, Y_estimated, W, b, lambda_)
    try:
        cost_array = np.load(f'Cost/cost_{file_name}__{file_name_attribution}.npy')
    except FileNotFoundError:
        cost_array = np.array([])
    cost_array = np.append(cost_array, cost)
    np.save(f'Cost/cost_{file_name}__{file_name_attribution}.npy', cost_array)


def update_weights(W, W_gradient, b, b_gradient, eta):
    W -= eta * W_gradient
    b = np.array(b)
    b_gradient = np.array(b_gradient)
    b = b - eta * b_gradient
    return W, b


def ComputeGradsNumSlow(X, Y, W, b, lamda, h=0.00001):
    """ Converted from matlab code """
    no = W.shape[0]
    d = X.shape[0]

    grad_W = np.zeros(W.shape)
    grad_b = np.zeros((W.shape[0], 1))

    for i in range(b.shape[0]):
        b_try = np.array(b)
        b_try[i] -= h
        Y_estimated = evaluate_classifier(X, W, b_try)
        c1 = compute_cost(X, Y, Y_estimated, W, b_try, lamda)

        b_try = np.array(b)
        b_try[i] += h
        Y_estimated = evaluate_classifier(X, W, b_try)
        c2 = compute_cost(X, Y, Y_estimated, W, b_try, lamda)

        grad_b[i] = (c2 - c1) / (2 * h)

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_try = np.array(W)
            W_try[i, j] -= h
            Y_estimated = evaluate_classifier(X, W_try, b)
            c1 = compute_cost(X, Y, Y_estimated, W_try, b, lamda)

            W_try = np.array(W)
            W_try[i, j] += h
            Y_estimated = evaluate_classifier(X, W_try, b)
            c2 = compute_cost(X, Y, Y_estimated, W_try, b, lamda)

            grad_W[i, j] = (c2 - c1) / (2 * h)

    return grad_W, grad_b


def compute_gradients(X, Y, Y_estimated, W, lambda_):
    Y_delta = Y_estimated - Y

    gradient_W = np.dot(Y_delta, X.T) / X.shape[1]
    gradient_W += 2 * lambda_ * W

    gradient_b = np.sum(Y_delta, axis=1) / Y_delta.shape[1]
    gradient_b = gradient_b.reshape(-1, 1)

    return gradient_W, gradient_b


def compute_accuracy(X, Y, W, b):
    y_true = np.argmax(Y, axis=0)
    Y_estimated = evaluate_classifier(X, W, b)
    Y_estimated = np.array(Y_estimated)
    y_estimated = np.argmax(Y_estimated, axis=0)

    correct_labeled = np.sum(y_true.flatten() == y_estimated.flatten())
    accuracy = correct_labeled / y_true.shape[0]
    return accuracy


def compute_loss(X, Y_true, Y_estimated, W, b):
    return compute_cost(X, Y_true, Y_estimated, W, b, 0)


def compute_cost(X, Y_true, Y_estimated, W, b, lambda_, epsilon=0.000001):
    # Add a small constant to predicted probabilities to avoid taking the log of zero
    Y_estimated = np.maximum(Y_estimated, epsilon)
    loss_cross_entropy = (1 / Y_true.shape[1]) * -np.sum(Y_true * np.log(Y_estimated))
    loss_regularization = lambda_ * np.sum(np.square(W))
    loss = loss_cross_entropy + loss_regularization
    return loss


def evaluate_classifier(X, W, b):
    s = np.dot(W, X) + b
    p = softmax(s)
    # y = np.argmax(p, axis=0)
    return p


def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def load_batch(filename):
    """ Copied from the dataset website """
    with open('Dataset/cifar-10-batches-py/' + filename, 'rb') as fo:
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
    np.random.seed(400)
    # Initialize W with Gaussian random values
    W = np.random.randn(K, d) * 0.01
    # Initialize b with Gaussian random values
    b = np.random.randn(K, 1) * 0.01
    b = np.array(b)
    return W, b


if __name__ == "__main__":
    main()
    sys.exit()
