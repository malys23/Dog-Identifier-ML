# function 

import numpy as np
def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    #Get vector "A" predicting prob of dog being present in the pic
    A = 1/(1+ np.exp(-1*(np.dot(w.T, X) + b)))
    
    for i in range(A.shape[1]):
        #Convert probs A[0, i] to actual preds p[0, i]
        if A[0, i] > 0.5 :
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0
    
    return Y_prediction