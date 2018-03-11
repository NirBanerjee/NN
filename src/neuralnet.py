from fileIO import readDataFile, writeToFile
from vectorizer import generateVectors
from utilities import forwardPropagation, backPropagation, getCrossEntropy, predictLabels, getError
import numpy as np
import sys

np.set_printoptions(precision=8, suppress=True)

#Main Method
if __name__ == '__main__':

	#Get all command line arguments
	trainingFile = sys.argv[1]
	validationFile = sys.argv[2]
	trainingLabelsFile = sys.argv[3]
	validationLabelsFile = sys.argv[4]
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
	labelsSet = set(trainingLabels)
	labelsSet = list(labelsSet)
	labelsSet = sorted(labelsSet)
	if initFlag != 1:
		alpha = np.zeros([hiddenUnits, len(trainingFeatures[0])])
		beta = np.zeros([len(labelsSet), hiddenUnits])
	else:
		alpha = np.random.uniform(-0.1, 0.1, (hiddenUnits, len(trainingFeatures[0])))
		beta = np.random.uniform(-0.1, 0.1, (10, hiddenUnits))

	#Add Zero Row in Alpha and Beta
	zeroAlpha = np.zeros([hiddenUnits, 1])
	alpha = np.concatenate((zeroAlpha, alpha), axis = 1)
	zeroBeta = np.zeros([10, 1])
	beta = np.concatenate((zeroBeta, beta), axis = 1)

	metricsWriter = []
	#Start Neural Network Training
	for i in range(numEpochs):
		record = 0
		for feature in trainingFeatures:

			#Add the bias term
			feature = np.insert(feature, 0, 1)
			feature = np.matrix(feature)

			#Forward Propagation
			yHat, vectorZ = forwardPropagation(feature, alpha, beta)
			#Backward Propagation
			gradAlpha, gradBeta = backPropagation(feature, trainingLabels[record], alpha, beta, yHat, vectorZ)

			#Update Alpha and Beta
			alpha = alpha - eta * gradAlpha
			beta = beta - eta * gradBeta

			#And Continue
			record = record + 1

		#Get Cross Entropy for Train Data
		crossEntropyTrain = getCrossEntropy(trainingFeatures, trainingLabels, alpha, beta)
		metricsWriter.append("epoch=" + str(i+1) + " cossentropy(train): " + str(crossEntropyTrain))
		#Get Cross Entropy for Validation Data
		crossEntropyValidate = getCrossEntropy(validationFeatures, validationLabels, alpha, beta)
		metricsWriter.append("epoch=" + str(i+1) + " cossentropy(validation): " + str(crossEntropyValidate))

	#Predict Training Labels
	trainLabelsList = predictLabels(trainingFeatures, alpha, beta)
	trainError = getError(trainLabelsList, trainingLabels)
	metricsWriter.append("error(train): " + str(trainError))

	#Predict Validation Labels
	validationLabelsList = predictLabels(validationFeatures, alpha, beta)
	validationError = getError(validationLabelsList, validationLabels)
	metricsWriter.append("error(validation): " + str(validationError))	

	#Write All Data To File
	writeToFile(trainingLabelsFile, trainLabelsList)
	writeToFile(validationLabelsFile, validationLabelsList)
	writeToFile(metricsFile, metricsWriter)