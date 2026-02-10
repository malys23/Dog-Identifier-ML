from os import listdir
from numpy import asarray
from numpy import save
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

#modded from How to Classify Photos of Dogs and Cats (with 97% accuracy) by Jason Browniee
#load dogs vs not dogs dataset, reshape and save to a new file
folder_1 = 'train/'
folder_2 = 'test/'
photos, labels = list(), list()

#enumerate files in directory
#exchange folder # for whichever dataset you need
for file in listdir(folder_2):
    #determine class
    output = 1
    if file.startswith('dog'):
        output = 1
    #load image (change folder # accordingly)
    photo = load_img(folder_2 + file, target_size = (64, 64))
    print(photo)

    photo = img_to_array(photo)
    
    photos.append(photo)
    labels.append(output)

photos = asarray(photos)
labels = asarray(labels)
print(photos.shape, labels.shape)

#rename test or train accordingly
save('test_set_x.npy', photos)
save('test_set_y.npy', labels)