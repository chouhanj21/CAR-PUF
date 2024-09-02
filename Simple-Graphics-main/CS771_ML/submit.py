import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.linalg import khatri_rao

# You are allowed to import any submodules of sklearn that learn linear models e.g. sklearn.svm etc
# You are not allowed to use other libraries such as keras, tensorflow etc
# You are not allowed to use any scipy routine other than khatri_rao

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_map etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

     

def fun1(X):
    n = X.shape[0]
    Y = np.zeros((n*(n+1)//2,))  # Initialize Y with appropriate size
    
    # Calculate Y[k] for pairs of elements in X
    k = 0
    for i in range(n):
        for j in range(i+1, n):
            Y[k] = X[i][0] * X[j][0]
            k += 1
    
    # Add individual elements of X to Y
    for i in range(n):
        Y[k] = X[i][0]
        k += 1
    
    return Y

# Define the function fun2(X_all) in Python
def fun2(X_all):
    n = X_all.shape[0]
    V = np.zeros((n, 528))  # Initialize V with 528 dimensions for each X_all[i]
  
    for i in range(n):
        V[i] = fun1(X_all[i])
    
    return V


def calculate_X_single(challenge_vector):
    X = np.ones((len(challenge_vector), 1))  # Initialize X as a column vector
    for i in range(len(challenge_vector) - 1, -1, -1):
        di = 1 - 2 * challenge_vector[i]
        X[i] = di * X[i+1] if i < len(challenge_vector) - 1 else di
    return X

def calculate_X_all(challenges):
    n, m = challenges.shape
    X_all = np.zeros((n, m, 1))  # Initialize X_all for all challenges
    for i in range(n):
        X_all[i] = calculate_X_single(challenges[i])
    return X_all

################################
# Non Editable Region Starting #
################################
def my_map( X ):
################################
# Non Editable Region Ending   #
################################
    X_all = calculate_X_all(X)
    mapped_CRPS = fun2(X_all)
	# Use this method to create features.
	# It is likely that my_fit will internally call my_map to create features for train points
    return mapped_CRPS



# ################################
# # Non Editable Region Starting #
# ################################

def my_fit( X_train, y_train ):
    
# ################################
# # Non Editable Region Ending  #
# ################################
    mapped_CRPS = my_map(X_train)
    model = LogisticRegression()
    model.fit(mapped_CRPS , y_train)
    w = model.coef_
    b = model.intercept_
    return w.T[:,-1], b




