#modded from How to Classify Photos of Dogs and Cats (with 97% accuracy) by Jason Browniee
#load dogs vs not dogs dataset, reshape and save to a new file
import numpy as np
from os import listdir
from numpy import asarray
from numpy import save
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

#from matplotlib import pyplot
#from matplotlib.image import imread

#define location of dataset
folder1 = 'train/'
folder2 = 'test/'
photos, labels = list(), list()

#enumerate files in directory for training data
for file in listdir(folder1):
    #determine class
    output = 0
    if file.startswith('dog'):
        output = 1
    #load image (change folder # accordingly)
    photo = load_img(folder1 + file, target_size = (64, 64))
    #store
    photos.append(photo)
    labels.append(output)

#save the reshaped train photos as separate files
save('train_set_x.npy', photos)
save('train_set_y.npy', labels)

#############################################################

#repeat for test data
#clear photos and labels 
photos, labels = list(), list()

#enumerate files in directory for test data
for file in listdir(folder2):
    #determine class
    output = 0
    if file.startswith('dog'):
        output = 1
    #load image 
    photo = load_img(folder2 + file, target_size = (64, 64))
    #store
    photos.append(photo)
    labels.append(output)

#save the reshaped test photos as separate files
save('test_set_x.npy', photos)
save('test_set_y.npy', labels)

#test to check if images properly stored
#test_set_x_orig = np.load('test_set_x.npy')
#pyplot.imshow(test_set_x_orig[1])
#pyplot.show()