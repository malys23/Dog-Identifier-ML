import numpy as np
import copy
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage

import kagglehub
from kagglehub import KaggleDatasetAdapter
#need to get test and training data and import

#load the data (dog/non-dog)
dataset = kagglehub.dataset_download("danielshanbalico/dog-vs-not-dog")
print("Path to dataset files:", dataset)


#train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

