###Neural Network Template


####Description
This is a 3-layer neural network for classification using a set of continuous features, implemented in MATLAB and python.

The logistic sigmoid is used as the activation function. Weights are learned by minimizing a square error cost function with fmincg or fminunc (MATLAB/Octave) and fmin\_cg (python). The network is regularized.

The scripts split input data into training and test sets, and compute performance metrics after training (test and training set prediction accuracy and confusion matrix).

The code is based on Ex.4 of [ml-class.org](http://ml-class.org).

####Script steps
Read in .csv data file <br />
Randomize rows in dataset <br />
Define features (X) and class (y) <br />
Standardize features (subtract mean, divide by st dev) <br />
Split into training and test sets <br />
Define NN layers <br />
Initialize NN weights <br />
Minimize cost function <br />
Compute performance metrics <br />



####Constraints on input datasets
This template is suitable for data with a dependent variable designating 2 or more classes. The code works for numeric (not categorical) features.

The class column must be the last column in the input.

Each data row must be complete (no missing values), and every class must be represented in the training set.

Classes must be designated with consecutive integers, starting from 1 (for example, {1,2,3,4} but not {1,2,4}). Due to Octave/MATLAB syntax, '0' cannot be used to designate a class.

####Datasets used in development and testing
[Fisher's Iris](http://archive.ics.uci.edu/ml/datasets/Iris)<br />
[Wine](http://archive.ics.uci.edu/ml/datasets/Wine)<br />
[Breast Cancer Wisconsin (Diagnostic)](http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic\))<br />
[Vertebral Column](http://archive.ics.uci.edu/ml/datasets/Vertebral+Column)<br />
[Ionosphere](http://archive.ics.uci.edu/ml/datasets/Ionosphere)<br />

####Dataset processing notes
Fisher's Iris: The Iris classes {setosa, versicolor, virginica} were relabeled to {1,2,3}, respectively.

Wine: the class column has been moved to the last column in the dataset.<br />
Wisconsin Breast Cancer: the classes {malignant, benign} were relabeled to {1,2}, respectively. Class column has been moved to the end.<br />
Vertebral Column: the 3-class dataset was used. The classes {DH, SL, NO} were relabeled as {1,2,3}, respectively.<br />
Ionosphere: the 2nd feature column was removed, as all entries are zero. The classes {g,b} were relabeled as {1,2}, respectively.<br />

####Files
Python implementation script: nn\_template.py

MATLAB/Octave implementation script: nn\_template.m

Functions used by main .m script: sigmoid.m, sigmoidGradient.m, randInitializeWeights.m, nnCostFunction.m, fmincg.m, predict.m
