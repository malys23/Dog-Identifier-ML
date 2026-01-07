#modded from How to Classify Photos of Dogs and Cats (with 97% accuracy) by Jason Browniee
from matplotlib import pyplot
from matplotlib.image import imread

#define location of dataset
folder = 'train/'

#plot first few images
for i in range(9):
    #define subplot
    pyplot.subplot(330 + 1, i)
    #define file name
    filename = folder + 'dog (' + str(i+1) + ').jpg'
    print(filename)
    #load image pixels
    image = imread(filename)
    #plot raw pixel data
    pyplot.imshow(image)

#show the figure
pyplot.show()