import numpy as np 


def sigmoid(A): 

	"""
    Arguments:
    A -- numpy array (list) containing the input to the sigmoid function
    
    Returns:
    Z -- the output of the sigmoid function
	cache -- the input A
    """
	Z = (1 / (1 + np.exp(-A)))
	cache = A
	return (Z, cache)




def relu(A):
	"""
    Arguments:
    A -- numpy array (list) containing the input to the relu function
    
    Returns:
    Z -- the output of the relu function
	cache -- the input A
    """

	Z = np.copy(A)

	Z[Z < 0] = 0
	
	cache = A
	return (Z, cache)


def sigmoid_backward(Z): 
	"""
    Arguments:
    Z -- numpy array (list) containing the input to the differentiated sigmoid function
    
    Returns:
    Z -- the output of the differentiated simgoid function
    """


	(s, _) = sigmoid(Z)

	ds = s * (1 - s)

	return ds


def relu_backward(Z):

	"""
    Arguments:
    Z -- numpy array (list) containing the input to the differentiated sigmoid function
    
    Returns:
    Z -- the output of the differentiated simgoid function
    """

	A = np.copy(Z)

	A[Z <= 0] = 0

	A[Z > 0] = 1 

	return A

