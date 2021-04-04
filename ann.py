import numpy as np
import random as rnd


class Network():

    def __init__(self, sizes):
        # sizes = vector with number of neurons in each layer
        self.num_layers = len(sizes)
        # stochastic gradient descent algorithm start params
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def sigmoid(self, z):
        # z is a vector, numpy applies formula elementwise
        return 1.0/(1.0 + np.exp(-z))

    def sigmoid_derivative(self, z):
        """Derivative of the sigmoid function."""
        return self.sigmoid(z)*(1-self.sigmoid(z))

    def feedforward(self, a):
        """Return the output of the network, a = input"""
        for b, w in zip(self.biases, self.weights):
            # a'= o(wa+b)
            a = self.sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Stochastic Gradient Descent
           training_data = list of tuples
           eta = learning rate
           test_data(optional) = if supplied the program will evaluate progress
           after each epoch
        """
        results = []
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            rnd.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size]
                            for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_batch(mini_batch, eta)
            if test_data:
                results.append("Epoch {0}: {1} / {2}".format(j,
                                                             self.evaluate(test_data), n_test))
            else:
                results.append("Epoch {0} complete".format(j))
        return results

    def update_batch(self, batch, eta):
        """ Update the network's wights and biases by applying
            gradient descent using backpropagation
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in batch:
            delta_nabla_b, delta_nabla_w = self.backpropagation(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(batch))*nw for w,
                        nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(batch))*nb for b,
                       nb in zip(self.biases, nabla_b)]

    def backpropagation(self, x, y):
        """ Return a tuple representing the gradient
            for the cost function C_x
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedfoward
        activation = x
        activations = [x]  # list to store all activations
        zs = []  # list to store all the vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(
            activations[-1], y)*self.sigmoid_derivative(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.sigmoid_derivative(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta)*sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
            return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activation, y):
        """Return the vector of partial derivatives for the output activations."""
        return (output_activation-y)
