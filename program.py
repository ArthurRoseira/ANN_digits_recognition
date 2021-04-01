import load_data
import ann


training_data, validation_data, test_data = load_data.mnist_loader()
net = ann.Network([784, 30, 10])
# using stochastic gradient descent to learn from the MNIST Training Data
# 30 epochs
# mini-batch size = 10
# learning rate = 3.0
net.SGD(list(training_data), 30, 10, 3.0, test_data=list(test_data))
