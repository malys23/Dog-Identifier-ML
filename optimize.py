#helper function to optimize w and b by running gradient descent algorithm
#returns params, grads, costs (list of all costs from opitmization)
import copy
import numpy as np
from propagate import propagate

def optimize(w, b, X, Y, num_iterations = 100, learning_rate = 0.009, print_cost = False):
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)
    
    costs = []
    
    for i in range(num_iterations):
        #cost and grad calculation
        grads, cost = propagate(w, b, X, Y)
        #get derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        #update rule
        w = w - learning_rate * dw
        b = b - learning_rate * db
        #record costs
        if i % 100 == 0:
            costs.append(cost)
            #print cost every 100 training iterations
            if print_cost:
                print("Cost after iteration %i: %f" %(i, cost))
    
    params = {
        "w": w,
        "b": b
    }
    
    grads = {
        "dw": dw,
        "db": db
    }
    
    return params, grads, costs