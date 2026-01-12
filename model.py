#Model function compiling all functions to create a model
#Uses logistic regression model

import numpy as np

from initialize_with_zeros import initialize_with_zeros
from optimize import optimize
from predict import predict

def model(X_train, Y_train, X_test, Y_test, num_iterations=200, learning_rate=0.5, print_cost=False):
    #initial parameters with zeros
    w, b = initialize_with_zeros(X_train.shape[0])
    #gradient descent
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    #get params w and b from dict "params"
    w = params["w"]
    b = params["b"]
    #predict test/train set samples
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)
    
    # Print train/test Errors
    if print_cost:
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d
