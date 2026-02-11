import numpy as np

# Predicts if label is 0 or 1 using learned logistic regression parameters (weight, bias)
def predict(weights, bias, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    weights = weights.reshape(X.shape[0], 1)
    
    #Get vector "A" predicting probability of dog being present in the picture
    A = 1/(1+ np.exp(-1*(np.dot(weights.T, X) + bias)))
    
    #Convert probability A[0, i] to actual predictions p[0, i]
    for i in range(A.shape[1]):
        if A[0, i] > 0.5 :
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0
    
    return Y_prediction