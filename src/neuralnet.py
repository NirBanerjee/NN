from __future__ import division
from fileIO import readDataFile
from vectorizer import generateVectors
import numpy as np
import sys

np.set_printoptions(precision=8, suppress=True)

#Main Method
if __name__ == '__main__':

	#Get all command line arguments
	trainingFile = sys.argv[1]
	validationFile = sys.argv[2]
	trainingLabels = sys.argv[3]
	validationLabels = sys.argv[4]
	metricsFile = sys.argv[5]
	numEpochs = int(sys.argv[6])
	hiddenUnits = int(sys.argv[7])
	initFlag = int(sys.argv[8])
	eta = float(sys.argv[9])

	#Read Training Data File
	trainingData = readDataFile(trainingFile)
	#Read Validation Data File
	validationData = readDataFile(validationFile)

	#Get Features and Labels from Test Data
	trainingFeatures, trainingLabels = generateVectors(trainingData)
	#Get Features and Labels from Validation Data
	validationFeatures, validationLabels = generateVectors(validationData)

	#Initialize Alpha and Beta Matrices
	if initFlag != 1:
		alpha = np.zeros([hiddenUnits, len(trainingFeatures[0])])
		beta = np.zeros([10, hiddenUnits])
	else:
		alpha = np.random.uniform(-0.1, 0.1, (hiddenUnits, len(trainingFeatures[0])))
		beta = np.random.uniform(-0.1, 0.1, (10, hiddenUnits))

	#Add Zero Row in Alpha and Beta
	zeroAlpha = np.zeros([1, len(trainingFeatures[0])])
	alpha = np.concatenate((zeroAlpha, alpha), axis = 0)
	zeroBeta = np.zeros([10, 1])
	beta = np.concatenate((zeroBeta, beta), axis = 1)

	print(np.shape(alpha))
	print(alpha)
	print(np.shape(beta))
	print(beta)
	# print(trainingFeatures)
	# print("\n")
	# print(trainingLabels)


