import load_data
import ann
from sklearn import svm
import numpy as np


import gzip
import pickle


training_data, validation_data, test_data = load_data.mnist_loader('ann')
net = ann.Network([784, 30, 10])
# using stochastic gradient descent to learn from the MNIST Training Data
# 30 epochs
# mini-batch size = 10
# learning rate = 3.0

a = list(test_data)
v = list(training_data)
results = net.SGD(v, 1, 10, 3.0, test_data=a)
print(results[-1])
res = net.predict(a[0])
print(a[0][1])
print(res)

# Comparison using SVM/SVC model
# with gzip.open('./data/mnist.pkl.gz', 'rb') as file_contents:
#     train_set, valid_set, test_set = pickle.load(
#         file_contents, encoding='latin1')
# train_x, train_y = train_set
# test_x are the images in the test_set, test_y are the corresponding digits
# represented in those images
# test_x, test_y = test_set
# no_images = len(train_x)
# reduced_training_set_x = train_x[0: no_images // 10]
# reduced_training_set_y = train_y[0: no_images // 10]
# model = svm.SVC(kernel='sigmoid')
# print(model.fit(reduced_training_set_x, reduced_training_set_y))
