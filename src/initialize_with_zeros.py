import numpy as np

def initialize_with_zeros(dim):
    weights = np.zeros((dim, 1))
    bias = 0.0
    return weights, bias