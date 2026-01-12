# Dog-Identifier-ML
Logistic regression classifier to recognize dogs.
Results: Training accuracy is 100% while test accuracy is only 58%.
Learning Outcome: Sourcing and preprocessing data is an important step and may be time consuming

The model uses the following:
initialize_with_zeros.py - creates vector of zeros of shape (dim, 1) for w and initializes b to 0
propagate.py - implements cost function and gradient for propagation
optimize.py - optimize w and b by running gradient descent algorithm
predict.py - predicts if label is 0 or 1
model.py - Model function compiling all functions to create a model
dogModel.py - main function that organizes data and calls the model

Data taken from "Dog vs Not-Dog" by Daniel Shan Balico on kaggle.com
Functions and structure used from "Logistic Regression with a Neural Network mindset" from DeepLearning.AI