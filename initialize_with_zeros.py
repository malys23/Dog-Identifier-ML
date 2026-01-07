#helper function creates vector of zeros of shape (dim, 1) for w
# and initializes b to 0

import numpy as np

def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0.0
    return w, b