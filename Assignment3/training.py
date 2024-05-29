import numpy as np
from plotting import plot_accuracy, plot_loss
from model import NeuralNetwork

def train_model(nn, X_train, Y_train, X_val, Y_val, GDparams):
    """
    Train the model using mini-batch gradient descent.
    
    Args:
        nn (NeuralNetwork): Neural network model.
        X_train (numpy.ndarray): Training data.
        Y_train (numpy.ndarray): Training labels.
        X_val (numpy.ndarray): Validation data.
        Y_val (numpy.ndarray): Validation labels.
    """
    
    '''
    GDparams = {
    'lambda': 2.15714286e-03,
    'n_batch': 100,
    'eta_min': 0.00001,
    'eta_max': 0.1,
    'n_epochs': 50,
    'cycles': 3,
    'eta_s': 800,
    'update_steps': 1000
    }
    '''

    n_batch = GDparams['n_batch']
    epochs = GDparams['n_epochs']
    update_steps = GDparams['update_steps']
    k = 2
    eta_s = k * X_train.shape[1] / n_batch

    step = 0
    epoch = 0
    loss_train = 0
    loss_val = 0

    for e in range(epochs):
        # TODO: Shuffle the data
        for j in range(0, X_train.shape[1], n_batch):
            X_batch = X_train[:, j:j+n_batch]
            Y_batch = Y_train[:, j:j+n_batch]


            eta = cyclical_learning_rate(GDparams, step, eta_s)
            
            nn.update_weights(X_batch, Y_batch, eta)
            loss_t = nn.compute_loss(X_batch, Y_batch)
            loss_v = nn.compute_loss(X_val, Y_val)

            if e == 0:
                loss_train = loss_t
                loss_val = loss_v
            else:
                loss_train = np.append(loss_train, loss_t)
                loss_val = np.append(loss_val, loss_v)

            print(f'Epoch: {epoch}, Step: {step}, Loss: {loss_t}, eta: {eta}')
            
            step += 1
        epoch += 1


    


def cyclical_learning_rate(GDparam, step, eta_s):
    eta_min = GDparam['eta_min']
    eta_max = GDparam['eta_max']

    step = step % (2 * eta_s)
    
    if step <= eta_s:
        eta = eta_min + step / eta_s * (eta_max - eta_min)
        return eta

    elif eta_s < step:
        eta = eta_max - (step-eta_s) / eta_s * (eta_max - eta_min)
        return eta

    print('Error: no eta calculated')
    return