#!/usr/local/EPD/bin/python
#Filename: nn_template.py

#Description:
#Regularized neural network with a single hidden layer. Suitable for multi-class classification.
#Input data must be numeric (continuous, not nominal), and contain no missing values.

#Code is based on ml-class.org, Ex.4.


from nn_helper_functions import *


start_time = time.time()
print "\nScript started at %s \n" % time.strftime("%H:%M:%S",time.localtime(start_time))

# load, shuffle and normalize data
try:
	data_filename = sys.argv[1]
except:
	data_filename = 'fisher_iris.csv' #default dataset to use

print "Using %s\n" % data_filename
X_full,y_full = data_preprocess('datasets/' + data_filename)

# split data into training and test sets
X,y,X_test,y_test = split_data(X_full, y_full, train_frac = 0.70)

# learn network parameters
Theta1, Theta2 = nn_train(X,y, lam=1, hidden_layer_size = 10)

# evaluate performance
p_train = pred_accuracy(Theta1, Theta2, X, y)
p_test = pred_accuracy(Theta1, Theta2, X_test, y_test)
y_test_pred = predict(Theta1, Theta2, X_test)
cm = confusion_matrix(y_test, y_test_pred)

print '\nAccuracy on training set: %g' % p_train
print '\nAccuracy on test set: %g' % p_test
print '\nConfusion matrix:\n',cm


print "\nTotal run time: %.4f seconds" % (time.time() - start_time)



