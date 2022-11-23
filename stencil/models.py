https://powcoder.com
代写代考加微信 powcoder
Assignment Project Exam Help
Add WeChat powcoder
import numpy as np
import matplotlib.pyplot as plt

class RegularizedLogisticRegression(object):
	'''Implements regularized logistic regression for binary classification.

	The weight vector w should be learned by minimizing the regularized risk
	log(1 + exp(-y <w, x>)) + \lambda \|w\|_2^2. In other words, the objective
	function is the log loss for binary logistic regression plus Tikhonov
	regularization with a coefficient of \lambda.
	'''

	def __init__(self):
		self.learningRate = 0.00001 # Please dont change this
		self.num_epochs = 100000 # Feel free to play around with this if you'd like, though this value will do

		#####################################################################
		#																	#
		#	MAKE SURE TO SET THIS TO THE OPTIMAL LAMBDA BEFORE SUMITTING	#
		#																	#
		#####################################################################
		
		self.lmbda = 0.1 # tune this parameter
		

	def train(self, X, Y):
		'''
        Trains the model, using stochastic gradient descent
        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: a 1D Numpy array containing the corresponding labels for each example
        @return:
            None
        '''
        
        #[TODO]


	def predict(self, X):
		'''
        Compute predictions based on the learned parameters and examples X
        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
        @return:
            A 1D Numpy array with one element for each row in X containing the predicted class.
        '''
        
        #[TODO]


	def accuracy(self,X, Y):
		'''
        Outputs the accuracy of the trained model on a given testing dataset X and labels Y.
        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: a 1D Numpy array containing the corresponding labels for each example
        @return:
            a float number indicating accuracy (between 0 and 1)
        '''
        
        #[TODO]


	def plotError(self, X_train, y_train, X_val, y_val):
		'''
		Produces a plot of the cost function on the training and validation
		sets with respect to the regularization parameter lambda. Use this function to determine
		a valid lambda
		@params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: a 1D Numpy array containing the corresponding labels for each example
		'''
		lambda_list = [10000, 1000, 100, 10, 1, 0.1, 0.01, 0.001]
		
		#[TODO] train model and calculate train and validation errors here for each lambda, then plot.


def sigmoid_function(x):
	return 1.0 / (1.0 + np.exp(-x))

