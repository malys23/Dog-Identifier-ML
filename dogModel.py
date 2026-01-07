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

#Exercise 1

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]
train_set_y = train_set_y.reshape(1, m_train)
test_set_y = test_set_y.reshape(1, m_test)

print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("classes shape: " + str(classes.shape))