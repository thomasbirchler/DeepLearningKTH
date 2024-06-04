import numpy as np

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

    #n_batch = GDparams['n_batch']
    epochs = GDparams['n_epochs']
    update_steps = GDparams['update_steps']
    batch_size = GDparams['batch_size']
    
    #k = 2
    #eta_s = k * X_train.shape[1] / batch_size
    eta_s = GDparams['eta_s']

    step = 0
    epoch = 0
    loss_train = [0]
    loss_val = [0]
    cost_train = [0]
    cost_val = [0]
    accuracy_train = [0]
    accuracy_val = [0]

    for e in range(epochs):
        # TODO: Shuffle the data
        X_train, Y_train = shuffle_batch(X_train, Y_train)
        for j in range(0, X_train.shape[1], batch_size):
            X_batch = X_train[:, j:j+batch_size]
            Y_batch = Y_train[:, j:j+batch_size]


            eta = cyclical_learning_rate(GDparams, step, eta_s)
            
            nn.update_weights(X_batch, Y_batch, eta)
            
            #accuracy_train, accuracy_val, cost_train, cost_val, loss_train, loss_val = update_accuracy_cost_loss(
            #    nn, X_batch, Y_batch, X_val, Y_val, step, accuracy_train, accuracy_val, cost_train, cost_val, loss_train, loss_val)

            loss_t, cost_t = nn.compute_loss_cost(X_batch, Y_batch)
            loss_v, cost_v = nn.compute_loss_cost(X_val, Y_val)
            accuracy_t = nn.compute_accuracy(X_batch, Y_batch)
            accuracy_v = nn.compute_accuracy(X_val, Y_val)

            if step == 0:
                loss_train = loss_t
                cost_train = cost_t
                loss_val = loss_v
                cost_val = cost_v
                accuracy_train = accuracy_t
                accuracy_val = accuracy_v
            else:

                loss_train = np.append(loss_train, loss_t)
                loss_val = np.append(loss_val, loss_v)
                cost_train = np.append(cost_train, cost_t)
                cost_val = np.append(cost_val, cost_v)
                accuracy_train = np.append(accuracy_train, accuracy_t)
                accuracy_val = np.append(accuracy_val, accuracy_v)



            if step % 100 == 99:
                print(f'Epoch: {epoch}, Step: {step}, Loss: {loss_train[-1]:.3f}, Cost: {cost_train[-1]:.3f} eta: {eta:.5f}')
            
            step += 1
            if step == update_steps:
                return loss_train, loss_val, cost_train, cost_val, accuracy_train, accuracy_val
        epoch += 1
        if epoch == epochs:
            return loss_train, loss_val, cost_train, cost_val, accuracy_train, accuracy_val
    
    return loss_train, loss_val, cost_train, cost_val, accuracy_train, accuracy_val


    


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


def shuffle_batch(X, Y):
    # Generate a random permutation of indices
    indices = np.random.permutation(X.shape[1])

    # Shuffle both input features and target labels based on the same permutation
    X_shuffled = X[:, indices]
    Y_shuffled = Y[:, indices]

    return X_shuffled, Y_shuffled


def update_accuracy_cost_loss(nn, X_batch, Y_batch, X_val, Y_val, step, accuracy_train, accuracy_val, cost_train, cost_val, loss_train, loss_val):
    """
    Update the accuracy and cost of the model.
    
    Args:
        nn (NeuralNetwork): Neural network model.
        X (numpy.ndarray): Input data.
        Y (numpy.ndarray): Target labels.
    """

    loss_t, cost_t = nn.compute_loss_cost(X_batch, Y_batch)
    loss_v, cost_v = nn.compute_loss_cost(X_val, Y_val)
    accuracy_t = nn.compute_accuracy(X_batch, Y_batch)
    accuracy_v = nn.compute_accuracy(X_val, Y_val)

    loss_train_ma = moving_average(loss_train)
    loss_val_ma = moving_average(loss_val)
    cost_train_ma = moving_average(cost_train)
    cost_val_ma = moving_average(cost_val)
    accuracy_train_ma = moving_average(accuracy_train)
    accuracy_val_ma = moving_average(accuracy_val)

    ratio_old = 0
    ratio_new = 1 - ratio_old

    # Update lists
    if step == 0:
        loss_train = loss_t
        cost_train = cost_t
        loss_val = loss_v
        cost_val = cost_v
        accuracy_train = accuracy_t
        accuracy_val = accuracy_v
    elif step < 11:
        loss_train = np.append(loss_train, loss_t)
        loss_val = np.append(loss_val, loss_v)
        cost_train = np.append(cost_train, cost_t)
        cost_val = np.append(cost_val, cost_v)
        accuracy_train = np.append(accuracy_train, accuracy_t)
        accuracy_val = np.append(accuracy_val, accuracy_v)
    else:
        loss_train[-1] = ratio_old * loss_train_ma + ratio_new * loss_t
        loss_val[-1] = ratio_old * loss_val_ma + ratio_new * loss_v
        cost_train[-1] = ratio_old * cost_train_ma + ratio_new * cost_t
        cost_val[-1] = ratio_old * cost_val_ma + ratio_new * cost_v
        accuracy_train[-1] = ratio_old * accuracy_train_ma + ratio_new * accuracy_t
        accuracy_val[-1] = ratio_old * accuracy_val_ma + ratio_new * accuracy_v
    
    return accuracy_train, accuracy_val, cost_train, cost_val, loss_train, loss_val

def moving_average(lst):
        if type(lst) == np.ndarray and len(lst) <= 9:
            return np.mean(lst)
        elif type(lst) == np.ndarray:
            return np.mean(lst[-10:])
        else:
            return lst