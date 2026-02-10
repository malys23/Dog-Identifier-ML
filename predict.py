import numpy as np

# function that predicts if label is 0 or 1 using learned logistic regression parameters (w,b)
def predict(weights, bias, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    weights = weights.reshape(X.shape[0], 1)
    #Get vector "A" predicting prob of dog being present in the pic
    A = 1/(1+ np.exp(-1*(np.dot(weights.T, X) + bias)))
    
    for i in range(A.shape[1]):
        #Convert probs A[0, i] to actual preds p[0, i]
        if A[0, i] > 0.5 :
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0
    
    return Y_prediction