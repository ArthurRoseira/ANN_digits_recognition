import gzip
import pickle
import numpy as np


def mnist_loader(model=None):
    """ The data consists in a Tuple of with two entries.
        The First contains the actual training image 28x28(pixels)=784 array
        The Second is the digit (0...9) for the corresponding image
        model = 'ann' or 'svc'
    """
    with gzip.open('./data/mnist.pkl.gz', 'rb') as file_contents:
        tr_d, va_d, te_d = pickle.load(file_contents, encoding='latin1')
    if model == 'ann':
        training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
        training_results = [vectorized_result(y) for y in tr_d[1]]
        training_data = list(zip(training_inputs, training_results))
        validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
        validation_data = list(zip(validation_inputs, va_d[1]))
        test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
        test_data = list(zip(test_inputs, te_d[1]))
        return (training_data, validation_data, test_data)
    else:
        return (tr_d, va_d, te_d)


def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
