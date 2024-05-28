import numpy as np
from model import forward_pass
from plotting import plot_accuracy, plot_loss

def train_model(X_train, Y_train, X_val, Y_val, W1, W2, b1, b2, GDparams):
    # Add logic for training including updating weights and plotting
    pass

def compute_accuracy(X, Y, W1, W2, b1, b2):
    Y_pred = forward_pass(X, W1, W2, b1, b2)
    predictions = np.argmax(Y_pred, axis=0)
    labels = np.argmax(Y, axis=0)
    return np.mean(predictions == labels)
