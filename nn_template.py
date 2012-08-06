#!/usr/local/EPD/bin/python
#Filename: nn_template.py

#Description



#Code is based on ml-class.org, Ex.4.

import sys, numpy as np
from numpy import mat, c_, r_, array, e, reshape
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

#procedure









