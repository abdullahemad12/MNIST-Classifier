import numpy as np
from activation import *
import matplotlib.pyplot as plt


def init_adams(layer_dims):
	"""
	Initializes the adams parameters
	
	Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    dWl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    dbl -- bias vector of shape (layer_dims[l], 1)
	"""
	L = len(layer_dims)

	parameters = {}	

	for l in range(1, L):

		parameters["dW" + str(l)] = np.zeros((layer_dims[l], layer_dims[l-1]))
		parameters["db" + str(l)] = np.zeros((layer_dims[l], 1))
		
	return parameters

def init_parameters(layer_dims):

	"""
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """


	L = len(layer_dims)

	parameters = {}	

	for l in range(1, L):

		parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2 / layer_dims[l-1])
		parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))
		
	return parameters


def linear_forward(A, W, b): 
	"""
    Arguments:
    A -- numpy array (list) containing the elements from the previous layer
    
    Returns:
    Z -- result of W * A + b
    cache -- the inputs to the function
    """
	Z = np.dot(W, A) + b

	cache = (A, W, b)
	return (Z, cache)



def activation_forward(A_prev, W, b, activation): 
	"""
    Arguments:
    A -- numpy array (list) containing the elements from the previous layer
	W -- the parameters W 
	b -- the bias
	activation -- the activation as string either "sigmoid" or "relu"

    Returns:
    Z -- result of W * A + b
    cache -- the inputs to the function
    """

	activation_cache = None
	linear_cache = None 
	Z = None 
	A = None

	if activation == "relu":
		(A, linear_cache) = linear_forward(A_prev, W, b)
		(Z, activation_cache) = relu(A)

	elif activation == "sigmoid":
		(A, linear_cache) = linear_forward(A_prev, W, b)
		(Z, activation_cache) = sigmoid(A)

	cache = (linear_cache, activation_cache)
	return (Z, cache)


def forward_propagation(X, parameters): 
	"""
	performs a forward propagation
    Arguments:
	X -- from the input layer
	parameters -- the parameters W and b
   
 	Returns:
   	AL -- the output of the forward propagation
	caches -- the caches from performing the complete forward propagation
    """

	L = len(parameters) // 2

	A_prev = X
	
	caches = []

	for l in range(L-1):
		(A_prev, cache) = activation_forward(A_prev, parameters["W" + str(l+1)], parameters["b" + str(l+1)], "relu")
		caches.append(cache)

	
	(AL, cache) = activation_forward(A_prev, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid")
	
	caches.append(cache)

	return (AL, caches)



def compute_cost(Y, AL, lambd, parameters): 
	"""
	computes the cost of the current parameters 
    Arguments:
	Y -- the Y values of the data set
	AL -- the output of the forward_propagation
   
 	Returns:
   	J -- the cost computed from the cost function J
    """

	m = Y.shape[1]

	L = len(parameters) // 2

	regularization = 0

	for l in range(1, L+1):
		regularization = regularization + np.sum(np.square(parameters["W" + str(l)]))
	regularization = regularization * (lambd / (2 * m))

	J = - (1 / m) * np.sum((Y * np.log(AL)) + (1 - Y) * np.log(1 - AL))

	J = J + regularization
	return J


def linear_backward(dZ, linear_cache, lambd):
	
	"""
    Arguments:
    dZ -- the derivative calculated from the activation function
	linear_cache -- the cache from the forward propagation containing: A_L-1, W_L, b_L    

    Returns:
    dW -- dJ/dw the derivative of the cost with respect to W
                    db -- dJ/db the derivative of the cost with respect to b
                    dA_prev -- dJ/dA_l-1 the derivative of the previous layer
    """


	(A_prev, W, b) = linear_cache
	m = A_prev.shape[1]

	dW = (1 / m) * np.dot(dZ, A_prev.T)
	dW = dW + ((lambd / m) * W)
		
	db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

	dA_prev = np.dot(W.T, dZ)

	
	return (dW, db, dA_prev)


def activation_backward(dA, cache, activation, lambd):

	"""
    Arguments:
    dZ -- the derivative calculated from the activation function
	cache -- the cache from the forward propagation containing: linear_cache, activation_cache

    Returns:   
	dW -- the derivative with respect to W
	db -- the derivative with respect to b
	dA_prev -- the derivative with respect to A_prev
    """

	dW = None 
	db = None 
	dA_prev = None
	(linear_cache, Z) = cache
	
	if activation == "sigmoid":
		dg = sigmoid_backward(Z)
		dZ = dA * dg
		(dW, db, dA_prev) = linear_backward(dZ, linear_cache, lambd)
	elif activation == "relu":
		dg = relu_backward(Z)
		dZ = dA * dg 
		(dW, db, dA_prev) = linear_backward(dZ, linear_cache, lambd)

	return (dW, db, dA_prev)


def backward_propagation(AL, Y, caches, lambd):
	"""
    Arguments:
    parameters -- the parameters of the model
	AL -- the output from the forward propagation
	caches -- the caches from the forward propagation containing: linear_cache, activation_cache

    Returns:   
	grads -- the gradient descent values
    """

	L = len(caches)
	Y = Y.reshape(AL.shape)
	grads = {}
	
	dA_prev = - (Y / AL) + ((1 - Y) / (1 - AL))

	cache = caches[L-1]


	(dW, db, dA_prev) = activation_backward(dA_prev, cache, "sigmoid", lambd)

	grads["dW" + str(L)] = dW
	grads["db" + str(L)] = db


	for l in reversed(range(L-1)):

		cache = caches[l]

		(dW, db, dA_prev) = activation_backward(dA_prev, cache, "relu", lambd)

		grads["dW" + str(l+1)] = dW
		grads["db" + str(l+1)] = db

	return grads


def update_parameters(parameters, grads, learning_rate, V, S, t, beta1, beta2, epsilon):

	"""
	updates the parameters given the gradient descent
    Arguments:
    parameters -- the parameters of the model
	grads -- the gradient descent calculated from backward propagation
	learning_rate -- the learning_rate alpha

    Returns:   
	params -- the updated parameters
    """

	L = len(parameters) // 2

	for l in range(1, L):
		V["dW" + str(l)] = (beta1 * V["dW" + str(l)]) + ((1 - beta1) * grads["dW" + str(l)])
		V["db" + str(l)] = (beta1 * V["db" + str(l)]) + ((1 - beta1) * grads["db" + str(l)])

		S["dW" + str(l)] = (beta2 * S["dW" + str(l)]) + ((1 - beta2) * np.square(grads["dW" + str(l)]))
		S["db" + str(l)] = (beta2 * S["db" + str(l)]) + ((1 - beta2) * np.square(grads["db" + str(l)]))

		VWCorrected = V["dW" + str(l)] / (1 - (beta1 ** t))
		VbCorrected = V["db" + str(l)] / (1 - (beta1 ** t))

		SWCorrected = S["dW" + str(l)] / (1 - (beta2 ** t))
		SbCorrected = S["db" + str(l)] / (1 - (beta2 ** t))

		parameters["W" + str(l)] = parameters["W" + str(l)] - (learning_rate * (VWCorrected / np.sqrt(SWCorrected + epsilon)))
		parameters["b" + str(l)] = parameters["b" + str(l)] - (learning_rate * (VbCorrected / np.sqrt(SbCorrected + epsilon)))

	return (parameters, V, S)

def partition_set_into_epochs(X, Y, epoch):
	
	
	batch = []
	n_batches = X.shape[1] // epoch

	indices = np.arange(X.shape[1])
	np.random.shuffle(indices)

	X = X[:, indices]
	Y = Y[:, indices]

	for i in range(n_batches):
		cur_batch_X = X[:, i * epoch : (i + 1) * epoch]
		cur_batch_Y = Y[:, i * epoch : (i + 1) * epoch]
		batch.append((cur_batch_X, cur_batch_Y))

	if (n_batches * epoch) < X.shape[1]:
		cur_batch_X = X[:, n_batches * epoch : ]
		cur_batch_Y = Y[:, n_batches * epoch : ]
		batch.append((cur_batch_X, cur_batch_Y))

	return batch



def nn_model(X, Y, layer_dims, learning_rate = 0.01, num_iterations = 1500, lambd = 0, epoch = 300, showfigures=False, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):

	"""
	Creates and optimizes the parameters given the dimensions and dataset
    Arguments:
	X -- the data set
	Y -- the labels of the data set
	layer_dims -- the dimensions of the neural network
	learning_rate -- the learning rate alpha
    num_iterations -- the number of iterations 
	showfigures -- should be set to true in case the cost plot is to be shown

    Returns:   
	params -- the neural network as a numpy array
    """

	parameters = init_parameters(layer_dims)

	V = init_adams(layer_dims)
	S = init_adams(layer_dims)

	costs = []

	batches = partition_set_into_epochs(X, Y, epoch)
	n_batches = len(batches)
	for i in range(num_iterations):
		cost = 0 
		for j in range(n_batches):
			(batch_X, batch_Y) = batches[j]
			(AL, caches) = forward_propagation(batch_X, parameters)

			cost = compute_cost(batch_Y, AL, lambd, parameters)

			grads = backward_propagation(AL, batch_Y, caches, lambd)

			(parameters, V, S) = update_parameters(parameters, grads, learning_rate, V, S, i + 1, beta1, beta2, epsilon)
		if i % 15 == 0:
			print ("\rCost after iteration %i: %f" %(i, cost), end = "")
			if showfigures:
				costs.append(cost)

	if showfigures:
		# plot the cost
		plt.plot(np.squeeze(costs))
		plt.ylabel('cost')
		plt.xlabel('iterations (per hundreds)')
		plt.title("Learning rate =" + str(learning_rate))
		plt.show()


	return parameters


def predict(X, parameters): 
	"""
	Makes a prediction given the neural network and the input data
    Arguments:
	X -- the data set
	parameters -- the parameters returned by nn_model 
    
	Returns:   
	Y -- the predicition of the neural network
    """
	(AL, _) = forward_propagation(X, parameters)
	
	Y = np.zeros(AL.shape)
	Y[AL >= 0.5] = 1
	return Y

def calculate_accuracy(X, Y, parameters):
	"""
	calculates the accuracy on a test set
    Arguments:
	X -- the test set
	Y -- the labels of the test set
	parameters -- the parameters returned by nn_model 
    
	Returns:   
	accuracy -- the accuracy of the model on this test set as a number in this range [0, 1]
    """

	predictions = predict(X, parameters)
	m = Y.shape[0]
	accuracy =  1 - ((1 / m) * np.mean(np.abs(Y - predictions)))

	return accuracy 


