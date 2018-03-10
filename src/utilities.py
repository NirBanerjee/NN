import numpy as np 
import math

def calculateSigmoid(vector):
	return 1 /(1 + np.exp(-vector))

def softmax(vector):
	return np.exp(vector) / np.exp(vector).sum()

def forwardPropagation(feature, alpha, beta):

	#Calculate the Vector Z for second Layer
	vectorA = np.dot(alpha, feature.T)
	vectorZ = calculateSigmoid(vectorA)

	#Add the bias term for the second layer
	vectorZ = np.vstack([[1], vectorZ])

	#Calculate the prediction yHat
	vectorB = np.dot(beta, vectorZ)
	yHat = softmax(vectorB)
	
	return yHat, vectorZ

def backPropagation(feature, label, alpha, beta, yHat, vectorZ):

	#Calculating Gradient Beta
	gradB = yHat
	labelVal = int(label)
	gradB[labelVal][0] = gradB[labelVal][0] - 1
	gradBeta = np.dot(gradB, vectorZ.T)

	#Calculating Gradient Alpha
	gradZ = np.dot(beta.T, gradB)
	gradA = np.multiply(vectorZ, (1 - vectorZ))
	gradA = np.multiply(gradZ, gradA)
	gradA = np.delete(gradA, 0, axis = 0)
	gradAlpha = np.dot(gradA, feature)

	return gradAlpha,gradBeta

def getCrossEntropy(featureList, labelList, alpha, beta):
	sumEntropy = 0
	record = 0
	for feature in featureList:

		#Add the bias term
		feature = np.insert(feature, 0, 1)
		feature = np.matrix(feature)

		#Forward Propagation
		yHat, vectorZ = forwardPropagation(feature, alpha, beta)

		#Calculate Entropy
		labelVal = int(labelList[record])
		sumEntropy = sumEntropy + math.log(yHat[labelVal][0])
		record = record + 1

	return -sumEntropy/record

def predictLabels(featureList, alpha, beta):
	predictedLabels = []
	for feature in featureList:

		#Add the bias term
		feature = np.insert(feature, 0, 1)
		feature = np.matrix(feature)

		#Forward Propagation
		yHat, vectorZ = forwardPropagation(feature, alpha, beta)

		#Get Label
		yHatArr = yHat.flatten()
		predictedLabel = np.argmax(yHatArr)
		predictedLabels.append(predictedLabel)

	return predictedLabels

def getError(predictedLabels, actualLabels):
	errorCount = 0
	record = 0
	for label in predictedLabels:
		if label != actualLabels[record]:
			errorCount = errorCount + 1
		record = record + 1

	return errorCount/record