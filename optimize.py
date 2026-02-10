#helper function to optimize w and b by running gradient descent algorithm
#returns params, grads, costs (list of all costs from opitmization)
import copy
from propagate import propagate

def optimize(weights, bias, X, Y, num_iterations = 100, learning_rate = 0.009, print_cost = False):
    weights = copy.deepcopy(weights)
    bias = copy.deepcopy(bias)
    
    costs = []
    
    for i in range(num_iterations):
        #cost and grad calculation
        grads, cost = propagate(weights, bias, X, Y)
        #get derivatives from grads
        dweights = grads["dweights"]
        dbias = grads["dbias"]
        #update rule
        weights = weights - learning_rate * dweights
        bias = bias - learning_rate * dbias
        #record costs
        if i % 100 == 0:
            costs.append(cost)
            #print cost every 100 training iterations
            if print_cost:
                print("Cost after iteration %i: %f" %(i, cost))
    
    params = {
        "weights": weights,
        "bias": bias
    }
    
    grads = {
        "dweights": dweights,
        "dbias": dbias
    }
    
    return params, grads, costs