import numpy as np

#Implements cost function and gradient for propagation
# w = weights, b = bias scalar, X = data size, Y=true label vector
#returns grads= dictionary of gradients of weights and bias (dw, db)
# and also cost - negative log-likelihood cost for logistic regression
def propagate(weights, bias, X, Y):
    m = X.shape[1]
    #Forward propagation
    A = 1 / (1 + np.exp(-1*(np.dot(weights.T, X)+bias)))
    cost = (-1/m) * np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))
    #Backward propagation
    dweights = (1/m) * np.dot(X, (A-Y).T)
    dbias = (1/m) * np.sum(A-Y)
    
    cost = np.squeeze(np.array(cost))
    
    grads = {
        "dweights": dweights,
        "dbias": dbias
    }
    
    return grads, cost