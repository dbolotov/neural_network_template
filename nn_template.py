#!/usr/local/EPD/bin/python
#Filename: nn_template.py

#Description



#Code is based on ml-class.org, Ex.4.

import sys, numpy as np
from numpy import mat, c_, r_, array, e, reshape, random, sqrt, unique
from scipy import optimize as op
import itertools


# Define functions

def sigmoid(z):
	g = 1./(1 + e**(-z))
	return g

def sigmoidGradient(z):
	#must convert to array first
	if type(z) != 'numpy.ndarray':
		z = array([z])
	f = 1./(1 + e**(-z))
	return f*(np.ones(f.shape[0]) - f)

def randInitializeWeights(L_in, L_out):
	#epsilon_init = 0.12
	epsilon_init = float(sqrt(6))/sqrt(L_in + L_out)
	return random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init

def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lam):
	#code
	
	return 0

def predict(Theta1, Theta2, X):
	m = X.shape[0]
	num_labels = size(Theta2,1)

	h1 = sigmoid((c_[np.ones(m), X]) * np.transpose(Theta1)) 
	h2 = sigmoid((c_[np.ones(m), h1]) * np.transpose(Theta2))
	
	#assign each row of output  to be max of each row of h2
	return h2.max(1)


# Neural Network script

# Input: feature columns followed by dependent class column

data = np.loadtxt('wine.csv', delimiter = ',')

# shuffle rows
random.shuffle(data)

# separate into features and class
X = array(data[:,:-1])
y = array(data[:,-1])
y = reshape(y,(len(y),1)) #reshape into 1 by len(y) array

train_frac = 0.75 #fraction of data to use for training

# Split input file into training and test files
test_rows = int(round(X.shape[0] * (1 - train_frac))) #num of rows in test set
X_test = X[:test_rows, :] #test set
y_test = y[:test_rows] #test set

X = X[test_rows:,:] #training set
y = y[test_rows:] #training set

m = X.shape[0]

# NN layer sizes
input_layer_size = X.shape[1]
hidden_layer_size = 40
num_labels = unique(y).shape[0] #output layer









