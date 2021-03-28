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
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            rnd.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size]
                            for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batchs:
                self.update_batch(mini_batch, eta)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    def update_batch(self, batch, eta):
        """ Update the network's wights and biases by applying 
            gradient descent using backpropagation
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for b in self.weights]
        for x, y in batch:
            delta_nabla_b, delta_nabla_w = self.backpropagation(x, y)
            nabla_b = [nb+dnd for nb, dnd in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_b, delta_nabla_b)]
            self.weights = [w-(eta/len(mini_batch))*nw for w,
                            nw in zip(self.weights, nabla_w)]
            self.biases = [b-(eta/len(mini_batch))*nb for b,
                           nb in zip(self.biases, nabla_b)]


if __name__ == "__main__":
    ann = Network([2, 2, 1])
