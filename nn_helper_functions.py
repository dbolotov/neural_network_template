#!/usr/local/EPD/bin/python
#Filename: nn_helper_functions.py

#Functions used in the main nn_template.py script

from numpy import e
import numpy as np

def sigmoid(z):
	''' (number array) -> number array
	
	Return sigmoid of input array

	'''
	g = 1./(1 + e**(-z))
	return g

def sigmoidGradient(z):
	''' (number array) -> number array
	
	Return gradient of sigmoid of input array

	'''
	if type(z) != np.ndarray: #must convert to array first
		z = array([z])
	f = 1./(1 + e**(-z))
	return f*(1-f)