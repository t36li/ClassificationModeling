# /User/bin/python
# coding: utf-8
import os
import csv
import numpy as np
import pandas as pd
import random
import time
import pdb

def nearZeroVar(x_array, var_threshold):
	"""
	Given a ndarray, loop through columns and keep only those that have variance above var_threshold
	"""
	x_train = x_array
	for i in range(x_train.shape[1]):
		temp_col = x_train[:,i]
		temp_var = np.var(temp_col)
		if temp_var <= (var_threshold*(1-var_threshold)):
			x_train=np.delete(x_train, i, axis=1)
	return x_train
	
def computeNan(df):
	"""
	This functions returns a list computing the %NaN for each feature given a dataframe
	"""
	results=list()
	for col in df:
		results.append(float(df[col].isnull().sum())/len(df))
	return results

def computeGinis(x, y):
	"""
	This function takes in two vectors
	x: binary 0 and 1 labels
	y: a vector of numbers, prediction_probabilities, or 0/1 predictions

	return: a list of gini scores
	Note: absolute value of gini should be examined for selection
	"""
	from sklearn.metrics import roc_auc_score

	gini_list = list()

	for i in range(x.shape[1]):
		## Implementing METHOD ONE of Handling missing values: removal of rows ### (BEGIN)
		y_pred = x[:,i]
		gini_list.append(2*roc_auc_score(y, y_pred) - 1)

	return gini_list

def obtain_dummy_cols(df):
	"""
	given a dataframe, obtain the dummy columns (remove the target label after)
	dummy column defined as having only 2 different values
	"""
	
	dummy_cols=list()
	for col in df:
		if len(np.unique(np.array(df[col]))) == 2:
			dummy_cols.append(col)
	return dummy_cols

def ridge_dummy_regression(X,y,x_test,lambda_val=None):
	"""
	Train ridge L2 Logistic Regression on X,y. Then predict on x_test
	If lambda_val is provided, will just use this parameter for the L2 LR
	otherwise, will run 5-fold CV on C = log(-1.5, 1.5,5)

	This function returns a list of predicted probabilities as a list
	"""
	from sklearn.linear_model import LogisticRegression
	from sklearn.cross_validation import cross_val_score
	
	Cs=np.logspace(-1.5, 1.5, 5)
	lr = LogisticRegression(penalty='l2')
	cv_list=list()

	if not lambda_val:
		# Fit ridge to various choices of regularization parameter C to select best C
		for c in Cs:
			lr.C = c
			cv_score = cross_val_score(lr, X, y, scoring='roc_auc', cv=5)
			cv_list.append(np.mean(cv_score))

		print 'Best lambda based on Ridge Cross-Validation...'
		max_score=np.max(cv_list)
		lambda_val=Cs[cv_list.index(max_score)]
		print 1.0/lambda_val, max_score

	# Train LR with the optimized regularization parameter ###
	lr.C = lambda_val
	lr.fit(X,y)
	proba_lst = lr.predict_proba(x_test)[:,1]

	return proba_lst
