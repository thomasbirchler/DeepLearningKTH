import matplotlib.pyplot as plt

def plot_accuracies(accuracy_train, accuracy_val):
    plt.figure()
    plt.plot(accuracy_train, label='Train')
    plt.plot(accuracy_val, label='Validation')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('accuracy_plot.png')
    plt.show()
    plt.close()

def plot_losses(loss_train, loss_val, GDparams):
    batch_size = GDparams['n_batch']
    plt.figure()
    plt.plot(loss_train, label='Train')
    plt.plot(loss_val, label='Validation')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel(f'Update Step (Batch Size = {batch_size})')
    plt.legend(loc='upper left')
    plt.savefig('loss_plot.png')
    plt.show()
    plt.close()

def plot_costs(cost_train, cost_val, GDparams):
    batch_size = GDparams['n_batch']
    plt.figure()
    plt.plot(cost_train, label='Train')
    plt.plot(cost_val, label='Validation')
    plt.title('Model Cost')
    plt.ylabel('Cost')
    plt.xlabel(f'Update Step (Batch Size = {batch_size})')
    plt.legend(loc='upper left')
    plt.savefig('cost_plot.png')
    plt.show()
    plt.close()

