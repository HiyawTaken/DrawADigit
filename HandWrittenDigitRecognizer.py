import cupy as cp
import numpy as np
import struct

class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [cp.zeros((y, 1)) for y in sizes[1:]]
        self.weights = [cp.random.randn(y, x) / cp.sqrt(x) for x, y in zip(sizes[:-1], sizes[1:])]

def softmax(z):
    exp_z = cp.exp(z - cp.max(z, axis=0, keepdims=True))
    return exp_z / cp.sum(exp_z, axis=0, keepdims=True)

def activation_function(z):
    return cp.maximum(0, z)

def loss_function(y_hat, y):
    return -cp.sum(y * cp.log(y_hat + 1e-8)) / y.shape[1]

def one_hot_batch(y_batch, classes=10):
    batch_size = len(y_batch)
    result = cp.zeros((classes, batch_size))
    result[y_batch.astype(int), cp.arange(batch_size)] = 1
    return result

def weight_update(network, grad_weights, grad_biases, learning_rate):
    for i in range(len(network.weights)):
        network.weights[i] -= learning_rate * grad_weights[i]
        network.biases[i] -= learning_rate * grad_biases[i]

def forward_pass(network, inputs):
    outputs = []
    z_values = []
    curr_inputs = inputs

    for w, b in list(zip(network.weights, network.biases))[:-1]:
        z = cp.dot(w, curr_inputs) + b
        z_values.append(z)
        output = activation_function(z)
        outputs.append(output)
        curr_inputs = output

    w, b = network.weights[-1], network.biases[-1]
    z = cp.dot(w, curr_inputs) + b
    z_values.append(z)
    output = softmax(z)
    outputs.append(output)

    return outputs, z_values

def relu_derivative(z):
    return (z > 0).astype(float)

'''
BackPropagation
   What does this function receive as input? 
        We need the inputs ,Predicted outputs, actual outputs, z_values, weights, biases, loss
   What does it need to do step by step?
        go over the outputs reversed and calculate the partial derivatives and store the partial derivates
        we start by calulating the partial derivative of the loss from y_hat formula is (y_hat-y)
        then weight gradient is calculated by taking the derivative we just calc and multiplying by the output of the neuron
        grad_w for output layer = output layer delta * outputs from previous layer
        The bias gradient is exactly equal to the neuron gradient which is calculatred by adding all the partial derivatees of the weights that follows it in the last case it would js be y_hat
        When the gradient flows back through a hidden layer you multiply it by this derivative before computing that layer's weight gradients. This is called the delta for that layer
        the loop is stopped when the we have cleared the all the outputs
   What does it return?
        a list of gradient weight& bias
'''
def backpropagation(network, x, y, outputs, z_values, loss):
    grad_weights = []
    grad_biases = []

    delta = (outputs[-1] - y)
    grad_w_output = cp.dot(delta, outputs[-2].T) / y.shape[1]
    grad_b_output = cp.sum(delta, axis=1, keepdims=True) / y.shape[1]

    for i in reversed(range(network.num_layers - 2)):
        delta = cp.dot(network.weights[i + 1].T, delta) * relu_derivative(z_values[i])

        if i == 0:
            grad_w = cp.dot(delta, x.T) / y.shape[1]
        else:
            grad_w = cp.dot(delta, outputs[i - 1].T) / y.shape[1]

        grad_b = cp.sum(delta, axis=1, keepdims=True) / y.shape[1]
        grad_weights.insert(0, grad_w)
        grad_biases.insert(0, grad_b)

    grad_weights.append(grad_w_output)
    grad_biases.append(grad_b_output)

    return grad_weights, grad_biases
'''
Training
   What does this function receive as input& where does the data come from
        its needs the labeled data from our dataset, network, learning rate
   What are the exact steps that happen for every single image in the dataset, in order
        We will be using mini-batch training which means we will:
            1. We will shuffle and split our training dataset into a batch with a 1000 pictures each giving us
            60 subsets.
            2. then we will forward pass, backprop, accumlate gradients, then after a batch is completed, average the gradients and then update the weight 
            until the epoch is completed
            3. Repeat till loss converges
    How do you know if the network is learning, and when do you check
        We can js hv it print the loss after each subset
    What are the two things that define how long training runs
        loss and max epochs
   '''

def train(x_train, y_train, network, learning_rate, num_sections):
    epochs = 0
    max_epochs = 1000
    loss_history = []

    tolerance = 1e-3      # how small the loss change must be
    patience = 10          # how many epochs in a row it must stay small
    stable_count = 0

    while epochs < max_epochs:
        loss_list = []
        permutation = cp.random.permutation(x_train.shape[0])
        x_train_shuffled = x_train[permutation]
        y_train_shuffled = y_train[permutation]
        x_train_split = cp.array_split(x_train_shuffled, num_sections)
        y_train_split = cp.array_split(y_train_shuffled, num_sections)

        for x_batch, y_batch in zip(x_train_split, y_train_split):
            gradient_weights = []
            gradient_biases = []
            x_batch_matrix = x_batch.reshape(len(x_batch), 784).T
            y_batch_matrix = one_hot_batch(y_batch)
            predicted_output, predicted_z_values = forward_pass(network, x_batch_matrix)
            loss = loss_function(predicted_output[-1], y_batch_matrix )
            loss_list.append(loss)
            grad_weights, grad_biases = backpropagation(network, x_batch_matrix , y_batch_matrix, predicted_output, predicted_z_values, loss)
            gradient_weights.append(grad_weights)
            gradient_biases.append(grad_biases)

            weight_update(network, grad_weights, grad_biases, learning_rate)

        epochs += 1
        avg_loss = float(cp.mean(cp.array(loss_list)))
        loss_history.append(avg_loss)
        print(f"AVG LOSS: {avg_loss}, Epoch: {epochs}")

        if len(loss_history) > 1:
            loss_change = abs(loss_history[-2] - loss_history[-1])

            if loss_change < tolerance:
                stable_count += 1
            else:
                stable_count = 0

            if stable_count >= patience:
                print(f"Loss converged at epoch {epochs}")
                break

    return network

def accuracy(network, x_data, y_data, batch_size=1000):
    correct = 0
    total = x_data.shape[0]

    for i in range(0, total, batch_size):
        x_batch = x_data[i:i + batch_size]
        y_batch = y_data[i:i + batch_size]

        x_batch_matrix = x_batch.reshape(len(x_batch), 784).T

        predicted_output, predicted_z_values = forward_pass(network, x_batch_matrix)

        predictions = cp.argmax(predicted_output[-1], axis=0)

        correct += int(cp.sum(predictions == y_batch))

    return correct / total


def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, 784, 1) / 255.0
    return cp.array(images)

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return cp.array(labels)

def save_network(network, filename="mnist_network_weights.npz"):
    data = {}

    for i, w in enumerate(network.weights):
        data[f"weight_{i}"] = w

    for i, b in enumerate(network.biases):
        data[f"bias_{i}"] = b

    cp.savez(filename, **data)

def main():
    num_batches = 60
    learning_rate = 0.01
    loss_history = []

    network = Network([784, 128, 64, 10])
    x_train = load_mnist_images('mnist-dataset/train-images.idx3-ubyte')
    y_train = load_mnist_labels('mnist-dataset/train-labels.idx1-ubyte')
    x_test = load_mnist_images('mnist-dataset/t10k-images.idx3-ubyte')
    y_test = load_mnist_labels('mnist-dataset/t10k-labels.idx1-ubyte')

    network = train(x_train, y_train, network, learning_rate, num_batches)

    print("Starting testing ...")
    test_accuracy = accuracy(network, x_test, y_test)
    print(f"TEST ACCURACY: {test_accuracy * 100:.2f}%")

    save_file  = input("Save network? (y/n)")
    if save_file == "y":
        save_network(network)
    else:
        exit()

if __name__ == '__main__':
    main()