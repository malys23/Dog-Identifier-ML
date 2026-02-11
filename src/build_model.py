import numpy as np

from initialize_with_zeros import initialize_with_zeros
from optimize import optimize
from predict import predict

#Compiles all methods to build a logistic regression model
def build_model(X_train, Y_train, X_test, Y_test, num_iterations=200, learning_rate=0.5, print_cost=False):
    weights, bias = initialize_with_zeros(X_train.shape[0])
    #gradient descent
    parameters, gradients, costs = optimize(weights, bias, X_train, Y_train, num_iterations, learning_rate, print_cost)
    #get params w and b from dict "params"
    weights = parameters["weights"]
    bias = parameters["bias"]
    #predict test/train set samples
    Y_prediction_test = predict(weights, bias, X_test)
    Y_prediction_train = predict(weights, bias, X_train)
    
    # Print train/test Errors
    if print_cost:
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "weights" : weights, 
         "bias" : bias,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d
