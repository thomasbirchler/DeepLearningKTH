import numpy as np
import pickle

def load_batch(filename, data_path='Assignment3/data/cifar-10-batches-py/'):
    X_all = np.empty((3 * 1024, 0))
    y_all = np.empty((0,))

    for file in filename:
        with open(data_path + file, 'rb') as fo:
            data_dict = pickle.load(fo, encoding='bytes')
            X = data_dict[b'data'].T
            y = np.array(data_dict[b'labels'])
            X_all = np.concatenate((X_all, X), axis=1)
            y_all = np.concatenate((y_all, y))

    Y = np.eye(10)[y_all.astype(int)].T
    return X_all, Y, y_all

def preprocess_images(X):
    normalized_X = X / 255.0
    mean_X = np.mean(normalized_X, axis=0)
    std_X = np.std(normalized_X, axis=0)
    return (normalized_X - mean_X) / std_X

def load_and_preprocess_data(filenames_train, filenames_test):
    
    X_all, Y_all, y_all = load_batch(filenames_train)

    proportion = int(0.8 * X_all.shape[1])
    
    X_train, Y_train, y_train = X_all[:, :proportion], Y_all[:, :proportion], y_all[:proportion]
    X_val, Y_val, y_val = X_all[:, proportion:-1], Y_all[:, proportion:-1], y_all[proportion:-1]
    X_test, Y_test, y_test = load_batch(filenames_test)

    X_train = preprocess_images(X_train)
    X_val = preprocess_images(X_val)
    X_test = preprocess_images(X_test)

    return X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, Y_test, y_test
