#helper function implements cost function and gradient for propagation
# w = weights, b = bias scalar, X = data size, Y=true label vector
#returns grads= dictionary of gradients of weights and bias (dw, db)
# and also cost - negative log-likelihood cost for logistic regression

import numpy as np

def propagate(w, b, X, Y):
    m = X.shape[1]
    #Forward propagation
    A = 1/(1+ np.exp(-1*(np.dot(w.T, X)+b)))
    cost = (-1/m) * np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))
    #Backward propagation
    dw = (1/m) * np.dot(X, (A-Y).T)
    db = (1/m) * np.sum(A-Y)
    
    cost = np.squeeze(np.array(cost))
    
    grads = {
        "dw": dw,
        "db": db
    }
    
    return grads, cost