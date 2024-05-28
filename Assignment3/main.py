from data_utils import load_and_preprocess_data
from training import train_model
from model import initialize_parameters
from config.settings import GDparams, FILENAMES_TRAIN, FILENAMES_TEST

def main():
    X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, Y_test, y_test = load_and_preprocess_data(FILENAMES_TRAIN, FILENAMES_TEST)
    W1, W2, b1, b2 = initialize_parameters(Y_train.shape[0], X_train.shape[0])
    train_model(X_train, Y_train, X_val, Y_val, W1, W2, b1, b2, GDparams)

if __name__ == "__main__":
    main()
