#testing and visualization of reshape to 64 x 64
from matplotlib import pyplot
from matplotlib.image import imread
from keras.preprocessing.image import load_img

#define location of dataset
folder = 'dataPreprocessing/train/'

#plot first few images
for i in range(9):
    #define subplot
    pyplot.subplot(330 + 1 + i)
    #define file name
    filename = folder + 'dog (' + str(i+1) + ').jpg'
    #load image pixels
    #image = imread(filename)
    photo = load_img(filename, target_size = (64, 64))
    #plot raw pixel data
    pyplot.imshow(photo)
#show the figure
pyplot.show()