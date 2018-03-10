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

	print(np.shape(trainingFeatures))
	print(np.shape(trainingLabels))
	# print(trainingFeatures)
	# print("\n")
	# print(trainingLabels)


