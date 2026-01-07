#modded from How to Classify Photos of Dogs and Cats (with 97% accuracy) by Jason Browniee
#load dogs vs not dogs dataset, reshape and save to a new file
from os import listdir
from numpy import asarray
from numpy import save
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

#define location of dataset
folder1 = 'train/'
folder2 = 'test/'
photos, labels = list(), list()

#enumerate files in directory
for file in listdir(folder2):
    #determine class
    output = 1
    if file.startswith('dog'):
        output = 1
    #load image
    photo = load_img(folder2 + file, target_size = (64, 64))
    print(photo)
    #convert to numpy array
    photo = img_to_array(photo)
    #store
    photos.append(photo)
    labels.append(output)
    
#convert to a numpy array
photos = asarray(photos)
labels = asarray(labels)
print(photos.shape, labels.shape)

#save the reshaped photos as separate files
save('test_set_x.npy', photos)
save('test_set_y.npy', labels)