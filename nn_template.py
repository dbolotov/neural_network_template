#!/usr/local/EPD/bin/python
#Filename: nn_template.py

#Description:
#Regularized neural network with a single hidden layer. Suitable for multi-class classification.
#Input data must be numeric (continuous, not nominal), and contain no missing values.

#Code is based on ml-class.org, Ex.4.


from nn_helper_functions import *


start_time = time.time()


X_full,y_full = data_preprocess('datasets/wdbc_data.csv')
# X_full,y_full = data_preprocess('dataset_07.csv')

X,y,X_test,y_test = split_data(X_full, y_full, train_frac = 0.70)


Theta1, Theta2 = nn_train(X,y, lam=1, hidden_layer_size = 10)

p_train = pred_accuracy(Theta1, Theta2, X, y)
p_test = pred_accuracy(Theta1, Theta2, X_test, y_test)

y_test_pred = predict(Theta1, Theta2, X_test)
cm = confusion_matrix(y_test, y_test_pred)

print '\nAccuracy on training set: %g' % p_train
print '\nAccuracy on test set: %g' % p_test

print '\nConfusion matrix:\n',cm

# print '\nTrue vs Predicted\n', c_[y_test, y_test_pred]


print "Total run time:", time.time() - start_time, "seconds"



