import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import random

def load_batch(filename, data_path=None):
    script_dir = os.path.dirname(__file__)  # Get the directory where the script is located
    print(script_dir)
    data_path = os.path.join(script_dir, 'data', 'cifar-10-batches-py', filename)
    
    with open(data_path, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def preprocess(dataset):
    features = np.array(dataset[b"data"]).T[:, :].astype(np.float32)
    label_array = np.array(dataset[b"labels"])[:]
    encoded_labels = create_one_hot(label_array).T
    print(features.shape, "features shape")
    
    # Normalize the features
    mean_values = np.mean(features, axis=1).reshape(3072, 1)
    features = features - mean_values
    standard_deviation = np.std(features, axis=1).reshape(3072, 1)
    features = features / standard_deviation

    return features, encoded_labels, label_array

def create_one_hot(labels):
    one_hot = np.zeros((len(labels), 10))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot


def set_initial_network_params(feature_set, target_set, architecture, initializer='he', sigma=None):
    weights = []
    biases = []
    input_dim = feature_set.shape[0]

    np.random.seed(1)
    
    for index, layer_size in enumerate(architecture):
        fan_in = input_dim if index == 0 else architecture[index - 1]
        fan_out = architecture[index]

        if initializer == 'he':
            stddev = np.sqrt(2.0 / fan_in)
        elif initializer == 'xavier':
            stddev = np.sqrt(2.0 / (fan_in + fan_out))
        elif initializer == 'sigma':
            stddev = sigma
        else:
            raise ValueError("Unsupported initializer. Use 'he' or 'xavier'.")

        weight_matrix = np.random.normal(0, stddev, (layer_size, fan_in))
        bias_vector = np.zeros((layer_size, 1))
        
        weights.append(weight_matrix)
        biases.append(bias_vector)

    print(f"Initializer: {initializer.capitalize()}")
    print("Weights array length:", len(weights))
    print("Biases array length:", len(biases))

    return weights, biases


def compute_classifier_output(inputs, weights, biases, scale_params, shift_params, means=None, variances=None):
    current_input = inputs
    pre_activations = []
    outputs = []
    normalization_means = []
    normalization_vars = []
    s_hat_list = []
    outputs.append(inputs)

    # Cache for debugging or further analysis
    batch_outputs = [inputs]

    if means is None and variances is None:
        for index, (weight, bias) in enumerate(zip(weights, biases)):

            pre_activation = weight @ current_input + bias
            
            if index == len(weights) - 1:
                return np.exp(pre_activation) / np.sum(np.exp(pre_activation), axis=0), s_hat_list, pre_activations, outputs, normalization_means, normalization_vars
                
            
            pre_activations.append(pre_activation)

            mean = np.mean(pre_activation, axis=1).reshape(-1, 1)
            normalization_means.append(mean)
            
            var = np.var(pre_activation, axis=1).reshape(-1, 1)
            normalization_vars.append(var)

            s_hat = (pre_activation - mean) / np.sqrt(var + 1e-8)
            s_hat_list.append(s_hat)

            s_tilde = scale_params[index] * s_hat + shift_params[index]

            s = np.maximum(0, s_tilde)
            outputs.append(s)
            current_input = s

    else:
        for index, (weight, bias) in enumerate(zip(weights, biases)):
            
            pre_activation = weight @ current_input + bias

            if index == len(weights) - 1:
                return np.exp(pre_activation) / np.sum(np.exp(pre_activation), axis=0)
            
            s_hat = (pre_activation - means[index]) / np.sqrt(variances[index] + 1e-8)
            s_tilde = scale_params[index] * s_hat + shift_params[index]
            current_input = np.maximum(0, s_tilde)


def calculate_cost(features, targets, weights, biases, scale_params, shift_params, regularization_strength, means=None, variances=None):
    num_samples = features.shape[1]
    if means is None and variances is None:
        probabilities, _, _, _, _, _ = compute_classifier_output(features, weights, biases, scale_params, shift_params)
    else:
        probabilities = compute_classifier_output(features, weights, biases, scale_params, shift_params, means=means, variances=variances)

    cross_entropy_loss = -np.log(np.diag(targets.T @ probabilities))
    regularization_term = sum(np.sum(w**2) for w in weights) * regularization_strength
    #average_loss = np.mean(cross_entropy_loss) + regularization_term
    average_loss = np.sum(cross_entropy_loss) / num_samples
    return average_loss + regularization_term, average_loss


def compute_accuracy(features, actual_labels, weights, biases, scale_params, shift_params, means=None, variances=None, batch_norm=True):
    if means is None and variances is None:
        probabilities, _, _, _, _, _ = compute_classifier_output(features, weights, biases, scale_params, shift_params)
    else:
        probabilities = compute_classifier_output(features, weights, biases, scale_params, shift_params, means, variances)

    predicted_labels = np.argmax(probabilities, axis=0)
    accuracy = np.sum(predicted_labels == actual_labels) / len(actual_labels)
    return accuracy


def compute_gradients(inputs, targets, weights, biases, scale_params, shift_params, regularization_strength, batch_norm=True):
    # Obtain outputs and caches from the forward pass
    probabilities, normalized_outputs, pre_activations, activations, means, variances = compute_classifier_output(
        inputs, weights, biases, scale_params, shift_params)

    num_samples = inputs.shape[1]
    num_layers = len(weights)

    gradient_weights = []
    gradient_biases = []
    gradient_scales = []
    gradient_shifts = []

    # Gradient of the loss with respect to the output of the network
    delta = -(targets - probabilities)

    for layer_index in range(num_layers - 1, -1, -1):
        # Gradients for the last layer
        if layer_index == num_layers - 1:
            gradient_weights.append(delta @ activations[layer_index].T / num_samples + 2 * regularization_strength * weights[layer_index])
            #gradient_biases.append(np.sum(delta, axis=1, keepdims=True) / num_samples)
            gradient_biases.append(np.sum(delta, axis=1).reshape(-1, 1) / num_samples)
            delta = weights[layer_index].T @ delta
            delta *= (activations[layer_index] > 0)  # ReLU derivative
        
        else:
            # Gradients for batch normalization parameters
            gradient_scales.append(np.sum(delta * normalized_outputs[layer_index], axis=1).reshape(-1, 1) / num_samples)
            gradient_shifts.append(np.sum(delta, axis=1).reshape(-1, 1) / num_samples)

            delta = delta * scale_params[layer_index]

            delta_1 = delta * (1 / np.sqrt(variances[layer_index] + 1e-8))
            delta_2 = delta * (variances[layer_index] + 1e-8) ** (-3 / 2)

            D = pre_activations[layer_index] - means[layer_index]
            c = np.sum(delta_2 * D, axis=1).reshape(-1, 1)
            delta = delta_1 - np.sum(delta_1, axis=1).reshape(-1, 1) / num_samples - (D * c / num_samples)
            
            gradient_weights.append(delta @ activations[layer_index].T / num_samples + 2 * regularization_strength * weights[layer_index])
            gradient_biases.append(np.sum(delta, axis=1).reshape(-1, 1) / num_samples)
            delta = weights[layer_index].T @ delta
            delta *= (activations[layer_index] > 0)  # ReLU derivative
            

    # Reversing the lists as gradients are collected in reverse order
    gradient_weights = gradient_weights[::-1]
    gradient_biases = gradient_biases[::-1]
    gradient_scales = gradient_scales[::-1]
    gradient_shifts = gradient_shifts[::-1]

    return gradient_weights, gradient_biases, gradient_scales, gradient_shifts, means, variances


def compute_numerical_gradients(inputs, targets, weights, biases, regularization_strength, epsilon, batch_norm=False):
    """ Calculate gradients numerically based on the definition of derivatives. """

    num_weights = len(weights)
    gradient_weights = [np.zeros_like(w) for w in weights]
    gradient_biases = [np.zeros_like(b) for b in biases]
    scale_params = []
    shift_params = []
    for i, layer in enumerate(weights):
            if i < len(weights) - 1:
                scale_params.append(np.ones((layer.shape[0], 1)))
                shift_params.append(np.zeros((layer.shape[0], 1)))

    # Calculate the original cost with the given parameters
    original_cost, _ = calculate_cost(inputs, targets, weights, biases, scale_params, shift_params, regularization_strength, batch_norm=batch_norm)
    print(f"Original Cost: {original_cost}")

    # Numerically estimate the gradient of weights
    for layer_idx in range(num_weights):
        for i in range(weights[layer_idx].shape[0]):
            for j in range(weights[layer_idx].shape[1]):
                # Perturb the current weight element
                weights[layer_idx][i, j] += epsilon
                cost_plus_epsilon, _ = calculate_cost(inputs, targets, weights, biases, scale_params, shift_params, regularization_strength, batch_norm=batch_norm)
                weights[layer_idx][i, j] -= epsilon

                # Compute the numerical gradient
                gradient_weights[layer_idx][i, j] = (cost_plus_epsilon - original_cost) / epsilon

    # Numerically estimate the gradient of biases
    for layer_idx in range(num_weights):
        for i in range(biases[layer_idx].shape[0]):
            # Perturb the current bias element
            biases[layer_idx][i] += epsilon
            cost_plus_epsilon, _ = calculate_cost(inputs, targets, weights, biases, scale_params, shift_params, regularization_strength, batch_norm=batch_norm)
            biases[layer_idx][i] -= epsilon

            # Compute the numerical gradient
            gradient_biases[layer_idx][i] = (cost_plus_epsilon - original_cost) / epsilon

    return gradient_weights, gradient_biases


def cyclic_learning_rate_training(features_train, targets_train, labels_train, features_val, targets_val, labels_val, weights, biases, regularization_strength, batch_size, batch_norm=True):
    steps_per_epoch = int(5 * (45000 / batch_size))
    print("Steps per epoch:", steps_per_epoch)
    learning_rate_min = 1e-5
    learning_rate_max = 1e-1
    total_cycles = 2
    total_samples = features_train.shape[1]
    print("Total training samples:", total_samples)

    training_costs = []
    validation_costs = []
    training_losses = []
    validation_losses = []
    training_accuracies = []
    validation_accuracies = []

    update_tracking = []
    learning_rate_tracking = []

    scale_params = []
    shift_params = []
    global_means = []
    global_variances = []

    # Initialize scale parameters and shift parameters for batch normalization
    if batch_norm:
        for i, layer in enumerate(weights):
            if i < len(weights) - 1:
                scale_params.append(np.ones((layer.shape[0], 1)))
                shift_params.append(np.zeros((layer.shape[0], 1)))
                global_means.append(np.zeros((layer.shape[0], 1)))
                global_variances.append(np.ones((layer.shape[0], 1)))

    for cycle in range(total_cycles):
        # Shuffle both input features and target labels based on the same permutation
        indices = np.random.permutation(features_train.shape[1])
        features_train = features_train[:, indices]
        targets_train = targets_train[:, indices]
        labels_train = labels_train[indices]

        for step in range(2 * steps_per_epoch):
            learning_rate = (learning_rate_min + (step / steps_per_epoch) * (learning_rate_max - learning_rate_min)) \
                            if step <= steps_per_epoch \
                            else (learning_rate_max - ((step - steps_per_epoch) / steps_per_epoch) * (learning_rate_max - learning_rate_min))
            learning_rate_tracking.append(learning_rate)


            start_index = (step * batch_size) % total_samples
            end_index = ((step + 1) * batch_size) % total_samples
            if end_index != 0:
                end_index = end_index
            else:
                end_index = total_samples

            batch_features = features_train[:, start_index:end_index]
            batch_targets = targets_train[:, start_index:end_index]

            grad_weights, grad_biases, grad_scales, grad_shifts, mean_updates, variance_updates = compute_gradients(
                batch_features, batch_targets, weights, biases, scale_params, shift_params, regularization_strength, batch_norm)

            # Update parameters
            for i in range(len(weights)):
                weights[i] -= learning_rate * grad_weights[i]
                biases[i] -= learning_rate * grad_biases[i]

            if batch_norm:
                for i in range(len(scale_params)):
                    scale_params[i] -= learning_rate * grad_scales[i]
                    shift_params[i] -= learning_rate * grad_shifts[i]

                for i in range(len(global_means)):
                    global_means[i] = 0.9 * global_means[i] + 0.1 * mean_updates[i]
                    global_variances[i] = 0.9 * global_variances[i] + 0.1 * variance_updates[i]

            
            # if step > 2210 and step % 2 == 0:
            #     if batch_norm:
            #         training_cost, training_loss = calculate_cost(features_train[:, ::10], targets_train[:, ::10], weights, biases, scale_params, shift_params, regularization_strength, global_means, global_variances)
            #         training_accuracy = compute_accuracy(features_train, labels_train, weights, biases, scale_params, shift_params, global_means, global_variances)
            #         print(f'Step: {step}, Training_cost: {training_cost}, Training_accuracy: {training_accuracy}')
            # #     if training_cost > 1.7:
            # #         pass


            if step % (steps_per_epoch / 5) == 0:
                if batch_norm:
                    training_cost, training_loss = calculate_cost(features_train[:, ::10], targets_train[:, ::10], weights, biases, scale_params, shift_params, regularization_strength, global_means, global_variances)
                    training_accuracy = compute_accuracy(features_train, labels_train, weights, biases, scale_params, shift_params, global_means, global_variances, batch_norm=batch_norm)
                    validation_cost, validation_loss = calculate_cost(features_val, targets_val, weights, biases, scale_params, shift_params, regularization_strength, global_means, global_variances)
                    validation_accuracy = compute_accuracy(features_val, labels_val, weights, biases, scale_params, shift_params, global_means, global_variances, batch_norm=batch_norm)
                else:
                    training_cost, training_loss = calculate_cost(features_train[:, ::10], targets_train[:, ::10], weights, biases, scale_params, shift_params, regularization_strength, batch_norm=batch_norm)
                    training_accuracy = compute_accuracy(features_train, labels_train, weights, biases, scale_params, shift_params, batch_norm=batch_norm)
                    validation_cost, validation_loss = calculate_cost(features_val, targets_val, weights, biases, scale_params, shift_params, regularization_strength, batch_norm=batch_norm)
                    validation_accuracy = compute_accuracy(features_val, labels_val, weights, biases, scale_params, shift_params, batch_norm=batch_norm)
                
                training_costs.append(training_cost)
                training_losses.append(training_loss)
                training_accuracies.append(training_accuracy)
                validation_costs.append(validation_cost)
                validation_losses.append(validation_loss)
                validation_accuracies.append(validation_accuracy)

                update_tracking.append(step + (2 * steps_per_epoch) * cycle)

                print("Step: {:5d}, Training Cost: {:.3f}, Training Accuracy: {:.3f}, Validation Accuracy: {:.3f}, Learning Rate: {:.5f}".format(step, training_cost, training_accuracy, validation_accuracy, learning_rate))


    return {
        "training_costs": training_costs,
        "validation_costs": validation_costs,
        "training_losses": training_losses,
        "validation_losses": validation_losses,
        "training_accuracies": training_accuracies,
        "validation_accuracies": validation_accuracies,
        "weights": weights,
        "biases": biases,
        "updates": update_tracking,
        "global_means": global_means,
        "global_variances": global_variances,
        "scale_params": scale_params,
        "shift_params": shift_params
    }

def visualize_images(image_data):
    """Display a grid of 25 random images from the reshaped dataset."""
    reshaped_data = image_data.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
    plt.figure(figsize=(10, 10))
    for index in range(25):
        plt.subplot(5, 5, index + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(reshaped_data[random.randint(0, 9999)])
    plt.show()


def display_weight_images(weight_matrix):
    """Create a montage displaying an image for each label in the weight matrix."""
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i in range(2):
        for j in range(5):
            index = i * 5 + j
            image = weight_matrix[index, :].reshape(32, 32, 3, order='F')
            scaled_image = (image - image.min()) / (image.max() - image.min())
            transposed_image = scaled_image.transpose(1, 0, 2)
            axes[i, j].imshow(transposed_image, interpolation='nearest')
            axes[i, j].set_title(f"Label={index}")
            axes[i, j].axis('off')
    plt.tight_layout()
    plt.show()


def plot_results(results):
    plt.figure()
    plt.plot(results['updates'], results['training_costs'], label='Training cost')
    plt.plot(results['updates'], results['validation_costs'], label='Validation cost')
    plt.title('Training and Validation Costs')
    plt.legend()
    plt.xlabel('Updates')
    plt.ylabel('Cost')
    plt.savefig('Assignment3/Documentation/cost_plot.png')

    plt.figure()
    plt.plot(results['updates'], results['training_accuracies'], label='Training accuracy')
    plt.plot(results['updates'], results['validation_accuracies'], label='Validation accuracy')
    plt.title('Training and Validation Accuracies')
    plt.legend()
    plt.xlabel('Updates')
    plt.ylabel('Accuracy')
    plt.savefig('Assignment3/Documentation/accuracy_plot.png')


def compare_num_ana_gradients(X_train, Y_train, architecture, lambda_, batch_norm=False):
    weights, biases = set_initial_network_params(X_train[:50, :50], Y_train[:, :50], architecture, initializer='he')
    scale_params = []
    shift_params = []
    for i, layer in enumerate(weights):
            if i < len(weights) - 1:
                scale_params.append(np.ones((layer.shape[0], 1)))
                shift_params.append(np.zeros((layer.shape[0], 1)))

    gradient_weights, gradient_biases, gradient_scales, gradient_shifts, means, variances = \
    compute_gradients(X_train[:50, :50], Y_train[:, :50], weights, biases, scale_params, shift_params, lambda_, batch_norm=batch_norm)

    grad_weights_num , grad_biases_num = compute_numerical_gradients(X_train[:50, :50], Y_train[:, :50], weights, biases, lambda_, 1e-6, batch_norm=batch_norm)

    for i in range(len(weights)):
        print(gradient_weights[i]-grad_weights_num[i])
        print(gradient_biases[i]-grad_biases_num[i])


def main():
    # Dataset paths
    data_batch_files = [
        'data_batch_1',
        'data_batch_2',
        'data_batch_3',
        'data_batch_4',
        'data_batch_5'
    ]
    test_data_file = 'test_batch'

    # Load and preprocess training data
    training_data = [load_batch(file) for file in data_batch_files]
    test_data = load_batch(test_data_file)

    # Prepare training and test sets
    X_train, Y_train, labels_train = [
    np.concatenate(t, axis=1) if t[0].ndim > 1 else np.concatenate(t, axis=0)
    for t in zip(*[preprocess(d) for d in training_data])
]
    X_test, Y_test, labels_test = preprocess(test_data)

    # Display dataset shapes
    print("Training set shape:", X_train.shape, Y_train.shape, labels_train.shape)
    print("Test set shape:", X_test.shape, Y_test.shape, labels_test.shape)

    # Set hyperparameters
    #lambda_ = 0.005
    lambda_ = 0.005
    n_batch = 100
    architecture = [50, 50, 10]
    architecture = [50, 30, 20, 20, 10, 10, 10, 10]
    #sigmas = [1e-1, 1e-3, 1e-4]
    sigmas = [1e-1]
    val_size = 5000
    batch_norm = True

    # Split data into training and validation sets
    np.random.seed(0)
    indices = np.random.permutation(X_train.shape[1])
    X_val, Y_val, labels_val = X_train[:, indices[:val_size]], Y_train[:, indices[:val_size]], labels_train[indices[:val_size]]
    X_train, Y_train, labels_train = X_train[:, indices[val_size:]], Y_train[:, indices[val_size:]], labels_train[indices[val_size:]]

    # Training process
    for sigma in sigmas:
        weights, biases = set_initial_network_params(X_train, Y_train, architecture, initializer='he', sigma=sigma)
        result_dict = cyclic_learning_rate_training(X_train, Y_train, labels_train, X_val, Y_val, labels_val, weights, biases, lambda_, n_batch, batch_norm=batch_norm)
        
        # Evaluate on test set
        test_accuracy = compute_accuracy(X_test, labels_test, result_dict["weights"], result_dict["biases"], result_dict["scale_params"], result_dict["shift_params"], result_dict["global_means"], result_dict["global_variances"], batch_norm=batch_norm)
        print(f"Test accuracy with sigma={sigma}: {test_accuracy}")

        # Plotting the training and validation costs and accuracies
        plot_results(result_dict)



if __name__ == "__main__":
    main()