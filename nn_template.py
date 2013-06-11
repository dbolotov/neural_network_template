#!/usr/local/EPD/bin/python
#Filename: nn_template.py

#Description:
#Regularized neural network with a single hidden layer. Suitable for multi-class classification.
#Input data must be numeric (continuous, not nominal), and contain no missing values.

#Code is based on ml-class.org, Ex.4.


from nn_helper_functions import *


start_time = time.time()


X,y,X_test,y_test = data_preprocess('datasets/fisher_iris.csv')

Theta1, Theta2 = nn_train(X,y)

p_train = pred_accuracy(Theta1, Theta2, X, y)

p_test = pred_accuracy(Theta1, Theta2, X_test, y_test)

print '\nAccuracy on training set: %g' % p_train
print 'Accuracy on test set: %g' % p_test

print "Program run time:", time.time() - start_time, "seconds"



