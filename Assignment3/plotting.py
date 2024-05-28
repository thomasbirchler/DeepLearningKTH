import matplotlib.pyplot as plt

def plot_accuracy(accuracy_history):
    plt.figure()
    plt.plot(accuracy_history)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

def plot_loss(loss_history):
    plt.figure()
    plt.plot(loss_history)
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
