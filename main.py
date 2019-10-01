from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import model as nn


# load data
mnist = datasets.fetch_openml('mnist_784', data_home="dataset")
data = mnist["data"]
labels = mnist["target"].reshape(1, len(mnist["target"]))



# plot the first 12 items
data = data.reshape((data.shape[0], 28, 28))
f, axarr = plt.subplots(3,4)
j = -1
for i in range(12): 
	if i % 4 == 0:
		j = j + 1
	axarr[j , i % 4].imshow(data[i], cmap=plt.cm.gray)

plt.show()


# normalize data
data = data.reshape((data.shape[0], data.shape[1] * data.shape[2]))
data = data / 255.0

# split 70% training 30% test
train_X = data[: int(data.shape[0] * 0.7), :].T
train_Y = labels[:, : int(data.shape[0] * 0.7)]

test_X = data[int(data.shape[0] * 0.7):, :].T
test_Y = labels[:, int(data.shape[0] * 0.7):]

m = train_Y.shape[1]
# train model to recognize the digit zero
neural_networks = []
accuracies = []
for d in range(10): 
	train_Y_d = np.zeros((1, m))
	test_Y_d = np.zeros((1, test_Y.shape[1]))

	train_Y_d[0][train_Y[0] == chr(48 + d)] = 1
	test_Y_d[0][test_Y[0] == chr(48 + d)] = 1

	layer_dims = [train_X.shape[0], 8, 4, 1]

	network = nn.nn_model(train_X, train_Y_d, layer_dims, learning_rate = 0.01, num_iterations = 500, lambd = 0.001, showfigures=False)
	accuracy = 100 * nn.calculate_accuracy(test_X, test_Y_d, network)
	
	neural_networks.append(network)
	accuracies.append(accuracy)

avgAccuracy = 0
for i in range(len(accuracies)):	
	avgAccuracy = avgAccuracy + accuracies[i]

avgAccuracy = avgAccuracy / len(accuracies)

print ("\r                                                           ")
print ("\rAverage Accuracy: " + str(avgAccuracy) + "%", end = "\n")

