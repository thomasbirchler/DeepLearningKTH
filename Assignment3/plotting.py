import matplotlib.pyplot as plt


def plot_accuracies(accuracy_train, accuracy_val, GDparams):
    filename = create_filename('accuracy_plot_', GDparams)

    plt.figure()
    plt.plot(accuracy_train, label='Train')
    plt.plot(accuracy_val, label='Validation')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(filename)
    plt.show()
    plt.close()

def plot_losses(loss_train, loss_val, GDparams):
    filename = create_filename('loss_plot_', GDparams)

    batch_size = GDparams['batch_size']
    plt.figure()
    plt.plot(loss_train, label='Train')
    plt.plot(loss_val, label='Validation')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel(f'Update Step (Batch Size = {batch_size})')
    plt.legend(loc='upper left')
    plt.savefig(filename)
    plt.show()
    plt.close()

def plot_costs(cost_train, cost_val, GDparams):
    filename = create_filename('cost_plot_', GDparams)

    batch_size = GDparams['batch_size']
    plt.figure()
    plt.plot(cost_train, label='Train')
    plt.plot(cost_val, label='Validation')
    plt.title('Model Cost')
    plt.ylabel('Cost')
    plt.xlabel(f'Update Step (Batch Size = {batch_size})')
    plt.legend(loc='upper left')
    plt.savefig(filename)
    plt.show()
    plt.close()


def create_filename(label, GDparam):
    filename = f'Assignment3/Documentation/{label}'
    #filename += f'_lr_{GDparam["eta_max"]}'
    #filename += f'_batch_{GDparam["n_batch"]}'
    #filename += f'_epochs_{GDparam["n_epochs"]}'
    filename += f'_updatesteps_{str(GDparam["update_steps"])}',
    filename += f'_batchsize_{GDparam["batch_size"]}'
    filename += f'_lambda_{GDparam["lambda"]}'
    filename += f'_batchnorm_{GDparam["batch_norm"]}'
    filename += f'_layers_{GDparam["n_layers"]}'
    filename += '.png'
    return filename
