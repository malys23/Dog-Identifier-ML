from matplotlib import pyplot
from matplotlib.image import imread

#modded from How to Classify Photos of Dogs and Cats (with 97% accuracy) by Jason Browniee
folder = 'train/'

#plot first few images
for i in range(9):
    pyplot.subplot(330 + 1 + i)
    filename = folder + 'dog (' + str(i+1) + ').jpg'
    print(filename)
    image = imread(filename)
    pyplot.imshow(image)

pyplot.show()