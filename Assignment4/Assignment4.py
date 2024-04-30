import collections
import numpy as np
import copy


class RNN:
    def __init__(self, m, K, sig=0.01):
        """
        Initialize the RNN model's parameters.
        :param m: int, Number of hidden units.
        :param K: int, Dimensionality of the output/input (since they are the same).
        :param sig: float, Standard deviation factor for weight initialization.
        """
        # Initialize bias vectors as zero vectors
        self.b = np.zeros((m, 1))  # Bias for hidden layer
        self.c = np.zeros((K, 1))  # Bias for output layer

        # Initialize weight matrices with random values from a normal distribution
        self.U = np.random.randn(m, K) * sig  # Weight matrix from input to hidden
        self.W = np.random.randn(m, m) * sig  # Weight matrix from hidden to hidden
        self.V = np.random.randn(K, m) * sig  # Weight matrix from hidden to output


class Gradients:
    """ Class to store gradients for RNN parameters """
    def __init__(self, rnn_model):
        self.U = np.zeros_like(rnn_model.U)  # Gradient for weight matrix U
        self.W = np.zeros_like(rnn_model.W)  # Gradient for weight matrix W
        self.V = np.zeros_like(rnn_model.V)  # Gradient for weight matrix V
        self.b = np.zeros_like(rnn_model.b)  # Gradient for bias vector b
        self.c = np.zeros_like(rnn_model.c)  # Gradient for bias vector c


def compare_gradients(numerical_grad, analytical_grad):
    for param_name in vars(analytical_grad).keys():
        num_grad = getattr(numerical_grad, param_name)
        ana_grad = getattr(analytical_grad, param_name)
        difference = num_grad - ana_grad
        # print('Size of num_grad for param', param_name, num_grad.shape)
        # print('Size of ana_grad for param', param_name, ana_grad.shape)
        print(f'Difference between numerical and analytical gradients for param {param_name}: {difference}')
        # TODO: plot difference


def compute_grads_num(X, Y, RNN, h):
    grads = Gradients(RNN)
    h = np.array([h])

    # for j in vars(grads).keys():
    #     print(getattr(grads, j))

    for param_name in vars(RNN).keys():
        # print(f'Param_name {param_name}')
        # print(f"Computing numerical gradient for {param_name}")
        grad = compute_grad_num_slow(X, Y, param_name, RNN, h)
        setattr(grads, param_name, grad)

    return grads


def compute_grad_num_slow(X, Y, param_name, RNN, h):
    # Get the parameter as a numpy array
    param = getattr(RNN, param_name)
    # print(param)
    n = param.size
    grad = np.zeros(param.shape)

    # It's useful to flatten and reshape for generic parameter dimensions
    param_flat = param.flatten()
    for i in range(n):
        RNN_try = copy.deepcopy(RNN)

        # Perturb the parameter down
        param_flat[i] -= h
        setattr(RNN_try, param_name, param_flat.reshape(param.shape))
        l1, _, _ = forward_pass(RNN_try, X, Y, h)

        # Perturb the parameter up
        param_flat[i] += 2 * h
        setattr(RNN_try, param_name, param_flat.reshape(param.shape))
        l2, _, _ = forward_pass(RNN_try, X, Y, h)

        # Reset the parameter
        param_flat[i] -= h
        grad.flat[i] = (l2 - l1) / (2 * h)

    return grad


def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def synthesize_sequence(RNN, h0, x0, n, char_to_ind, ind_to_char):
    """
    Synthesize a sequence of characters from the RNN based on the initial hidden state and input.

    :param RNN: The RNN model with U, W, V, b, c parameters.
    :param h0: Initial hidden state (numpy array of shape (m, 1)).
    :param x0: First dummy input vector (one-hot encoded, numpy array of shape (K, 1)).
    :param n: Length of the sequence to generate.
    :param char_to_ind: Dictionary mapping characters to indices.
    :param ind_to_char: Dictionary mapping indices to characters.
    :return: Generated sequence of characters.
    """
    K = x0.shape[0]
    h = h0
    x = x0
    generated_sequence = []

    for t in range(n):
        # Forward pass through the RNN to get new hidden state and output probabilities
        h = np.tanh(np.dot(RNN.W, h) + np.dot(RNN.U, x) + RNN.b)
        o = np.dot(RNN.V, h) + RNN.c
        p = softmax(o)

        # Sample a character index from the probability distribution
        cp = np.cumsum(p.ravel())  # Flatten p and compute cumulative sum
        a = np.random.rand()  # Random draw from a uniform distribution [0, 1)
        idx = np.where(cp - a > 0)[0][0]  # Find the first index where cumulative sum exceeds a

        # Update x to be the one-hot encoding of the sampled character index
        x = np.zeros((K, 1))
        x[idx] = 1

        # Append sampled character to the output sequence
        generated_sequence.append(ind_to_char[idx])

    return ''.join(generated_sequence)


def one_hot_encode(char, char_to_ind, K):
    """ Returns a one-hot encoded vector for character `char`."""
    one_hot = np.zeros((K, 1))
    one_hot[char_to_ind[char]] = 1
    return one_hot


def prepare_data(book_data, seq_length, char_to_ind, K):
    """ Prepare the data matrices X and Y for the sequence length."""
    X_chars = book_data[:seq_length]
    Y_chars = book_data[1:seq_length + 1]

    # Initialize X and Y matrices
    X = np.zeros((K, seq_length))
    Y = np.zeros((K, seq_length))

    for i in range(seq_length):
        X[:, i] = one_hot_encode(X_chars[i], char_to_ind, K).flatten()
        Y[:, i] = one_hot_encode(Y_chars[i], char_to_ind, K).flatten()

    return X, Y


def forward_pass(RNN, X, Y, h0):
    """Perform the RNN forward pass and compute the cross-entropy loss."""
    m, K = RNN.W.shape[0], RNN.V.shape[0]
    _, seq_length = X.shape  # K x seq_length
    h = np.zeros((m, seq_length + 1))
    p = np.zeros((K, seq_length))
    h[:, 0] = h0.flatten()
    # h[:, 0] = h0
    loss = 0

    # Forward pass
    for t in range(seq_length):
        xt = X[:, t].reshape(-1, 1)
        h[:, t + 1] = np.tanh(np.dot(RNN.W, h[:, t].reshape(-1, 1)) + np.dot(RNN.U, xt) + RNN.b).squeeze()
        ot = np.dot(RNN.V, h[:, t + 1].reshape(-1, 1)) + RNN.c
        yt_hat = softmax(ot.flatten())
        p[:, t] = softmax(ot).squeeze()

        loss -= np.log(yt_hat[Y[:, t].astype(bool)][0])  # Cross-entropy loss

    return loss, h, p


def backward_pass(RNN, grads, x, y, h, p):
    """Compute gradients for the backward pass."""
    # Initialize gradients as zero
    grads.U, grads.W, grads.V, grads.b, grads.c = np.zeros_like(RNN.U), np.zeros_like(RNN.W), np.zeros_like(
        RNN.V), np.zeros_like(RNN.b), np.zeros_like(RNN.c)

    dnext_h = np.zeros_like(h[:, 0].reshape(-1, 1))

    for t in reversed(range(y.shape[1])):
        # Gradient of cross-entropy wrt output probabilities
        dy = p[:, t].reshape(-1, 1)
        dy[y[:, t].argmax()] -= 1

        # Gradients of parameters wrt loss
        grads.V += np.dot(dy, h[:, t + 1].reshape(1, -1))
        grads.c += dy

        # Backprop into h
        dh = np.dot(RNN.V.T, dy) + dnext_h
        dnext_h = dh * (1 - np.tanh(h[:, t + 1].reshape(-1, 1)) ** 2)

        grads.W += np.dot(dnext_h, h[:, t].reshape(1, -1))
        grads.U += np.dot(dnext_h, x[:, t].reshape(-1, 1).T)
        grads.b += dnext_h

    return grads


def update_parameters(RNN, grads, learning_rate=0.01):
    """Update RNN parameters using SGD."""
    for param, grad in zip([RNN.U, RNN.W, RNN.V, RNN.b, RNN.c],
                           [grads.U, grads.W, grads.V, grads.b, grads.c]):
        param -= learning_rate * grad


def clip_gradients(grads, clip_value=5):
    """Clip the gradients to mitigate exploding gradients."""
    for param in [grads.U, grads.W, grads.V, grads.b, grads.c]:
        np.clip(param, -clip_value, clip_value, out=param)


def read_book(filename):
    """Read the content of a book from a text file."""
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()


def preprocess_text(text):
    """Preprocess the text to extract unique characters and create mappings."""
    # Using OrderedDict to maintain the order of characters as first appearance
    unique_chars = collections.OrderedDict.fromkeys(text)
    char_to_ind = {char: idx for idx, char in enumerate(unique_chars)}
    ind_to_char = {idx: char for char, idx in char_to_ind.items()}
    K = len(unique_chars)
    return char_to_ind, ind_to_char, K


def set_hyperparams():
    dimensionality_hidden_layer = 5
    eta = 0.1
    seq_length = 2

    return [dimensionality_hidden_layer, eta, seq_length]


def main():
    hyper_params = set_hyperparams()

    # Path to the text file containing the book's text
    book_name = 'data/goblet_book.txt'

    # Read and preprocess the book data
    book_data = read_book(book_name)
    char_to_ind, ind_to_char, K = preprocess_text(book_data)
    X, Y = prepare_data(book_data, hyper_params[2], char_to_ind, K)

    # Print some example mappings to demonstrate
    # print("Character to Index Mapping (partial):", list(char_to_ind.items())[:10])
    # print("Index to Character Mapping (partial):", list(ind_to_char.items())[:10])
    # print(f'Unique chars: {K}')

    rnn_model = RNN(hyper_params[0], K)
    h0 = np.zeros((hyper_params[0], 1))
    x0 = np.zeros((K, 1))
    x0[0] = 1  # Assume first character is a dummy with index

    loss, h, p = forward_pass(rnn_model, X, Y, h0)
    # print('Loss: ', loss)

    # generated_text = synthesize_sequence(rnn_model, h0, x0, 20, char_to_ind, ind_to_char)
    # print("Generated Sequence:", generated_text)

    numerical_grad = compute_grads_num(X, Y, rnn_model, 0.0001)

    gradients = Gradients(rnn_model)
    analytical_gradients = backward_pass(rnn_model, gradients, X, Y, h, p)

    compare_gradients(numerical_grad, analytical_gradients)

    # clip_gradients(gradients)

    # update_parameters(rnn_model, gradients, learning_rate=0.01)


if __name__ == '__main__':
    main()
