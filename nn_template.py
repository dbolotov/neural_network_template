#!/usr/local/EPD/bin/python
#Filename: nn_template.py

#Description:
#Regularized neural network with a single hidden layer. Suitable for multi-class classification.
#Input data must be numeric (continuous, not nominal), and contain no missing values.

#Code is based on ml-class.org, Ex.4.


from nn_helper_functions import *


start_time = time.time()


X_full,y_full = data_preprocess('datasets/fisher_iris.csv')


# Split input file into training and test files
train_frac = 0.70
test_rows = int(round(X_full.shape[0] * (1 - train_frac))) #num of rows in test set
X_test = X_full[:test_rows, :] #test set
y_test = y_full[:test_rows] #test set

X = X_full[test_rows:,:] #training set
y = y_full[test_rows:] #training set

print type(X)
print X.shape

Theta1, Theta2 = nn_train(X,y, lam=1, hidden_layer_size = 10)

p_train = pred_accuracy(Theta1, Theta2, X, y)

p_test = pred_accuracy(Theta1, Theta2, X_test, y_test)

print '\nAccuracy on training set: %g' % p_train
print 'Accuracy on test set: %g' % p_test

print "Total run time:", time.time() - start_time, "seconds"



