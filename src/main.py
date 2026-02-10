import numpy as np
from build_model import build_model

def main():
    #set_x contains the images of dogs and non dogs
    #set_y contains the labels 0 (non dog) and 1 (dog) for each corresponding image in set_x  
    train_set_x_original = np.load('./data/train_dogvnondog/train_set_x.npy')
    train_set_y = np.load('./data/train_dogvnondog/train_set_y.npy')
    test_set_x_original = np.load('./data/test_dogvnondog/test_set_x.npy')
    test_set_y = np.load('./data/test_dogvnondog/test_set_y.npy')

    #1: find number of training examples, number oftest examples, and dimensions of the training image
    m_train = train_set_x_original.shape[0]
    m_test = test_set_x_original.shape[0]
    train_set_y = train_set_y.reshape(1, m_train)
    test_set_y = test_set_y.reshape(1, m_test)

    #2: Flatten images to single vectors 
    train_set_x_flatten = train_set_x_original.reshape(train_set_x_original.shape[0], -1).T
    test_set_x_flatten = test_set_x_original.reshape(test_set_x_original.shape[0], -1).T

    #3: Standardize the dataset (divide by 255, max val of pixel channel)
    train_set_x = train_set_x_flatten / 255
    test_set_x = test_set_x_flatten / 255

    #4: Run model
    logistic_regression_model = build_model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=3000, learning_rate=0.005, print_cost=True)

if __name__ == "__main__":
    main()
