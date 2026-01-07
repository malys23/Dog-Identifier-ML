import numpy as np
import copy
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage

import pandas as pd

train_set_x_orig = np.load('train_dogvnondog/train_set_x.npy')
train_set_y = np.load('train_dogvnondog/train_set_y.npy')
test_set_x_orig = np.load('test_dogvnondog/test_set_x.npy')
test_set_y = np.load('test_dogvnondog/test_set_y.npy')
classes = np.load('train_dogvnondog/list_classes.npy')

#test to see if data loaded
index = 100
plt.imshow(train_set_x_orig[index])
print(train_set_y[index])
print ("y = " + str(train_set_y[index]) + ", it's a '" + classes[(train_set_y[index])] +  "' picture.")
plt.show()