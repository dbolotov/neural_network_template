###Neural Network Template


####Description
This is a 3-layer neural network template for classification using a continuous feature set, implemented in MATLAB and python.

Weights are learned by minimizing a square error cost function with fmincg or fminunc (MATLAB/Octave) and fmin\_cg (python).

The script splits input data into training and test sets, and computes some performance metrics after training (test and training set prediction accuracy and confusion matrix).

The code is based on Ex.4 of [ml-class.org](http://ml-class.org).

####Constraints on input datasets
This template is suitable for data with continuous independent variables and a dependent variable designating 2 or more classes.

Each data row must be complete (no missing values).

Features must be numerical (continuous, not categorical).	

Classes must be designated with consecutive integers, starting from 1 (for example, {1,2,3,4} but not {1,2,4}). Due to Octave/MATLAB syntax, '0' cannot be used to designate a class.

Every class must be represented in the training set.

####Datasets used in development and testing
[Fisher's Iris](http://archive.ics.uci.edu/ml/datasets/Iris)  

[Wine](http://archive.ics.uci.edu/ml/datasets/Wine)  

[Breast Cancer Wisconsin (Diagnostic)](http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic\))

####Data processing steps
#####Fisher's Iris 
Labels of 1,2,3 have been assigned to setosa, versicolor, and virginica Iris classes, respectively.
#####Wine
The class column has been moved to the last column in the dataset.
#####Wisconsin Breast Cancer
Classes were relabeled to 1 (malignant) and 2 (benign). Class column has been moved to the end.

####Files
Python implementation script: nn\_template.py

MATLAB/Octave implementation script: nn\_template.m

Functions used by main .m script: sigmoid.m, sigmoidGradient.m, randInitializeWeights.m, nnCostFunction.m, fmincg.m, predict.m
