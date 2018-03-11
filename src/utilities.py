import numpy as np 
import math

def calculateSigmoid(vector):
	return 1 /(1 + np.exp(-vector))

def softmax(vector):
	return np.exp(vector)

def forwardPropagation(feature, alpha, beta):

	#Calculate the Vector Z for second Layer
	vectorA = np.dot(alpha, feature.T)
	vectorZ = calculateSigmoid(vectorA)

	#Add the bias term for the second layer
	vectorZ = np.vstack([[1], vectorZ])

	#Calculate the prediction yHat
	vectorB = np.dot(beta, vectorZ)
	yHat = softmax(vectorB)
	yHat = yHat/sum(yHat)
	
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
	record = 0
	sumEntropy = 0
	#featureListTranspose = featureList.T
	zMatrix = np.dot(alpha, featureList.T)
	zMatrix = sigmoid(zMatrix)
	biasArray = np.matrix(np.ones(len(featureList)))
	zMatrix = np.concatenate((biasArray, zMatrix), axis = 0)
	yHatMatrix = softmax(np.dot(beta, zMatrix))
	yHatMatrix = np.log(yHatMatrix)
	for column in yHatMatrix.T:
		columnArray = np.array(column).flatten()
		sumEntropy = sumEntropy + columnArray[labelList[record]]
		record = record + 1
	return -sumEntropy/record

def predictLabels(featureList, alpha, beta):
	predictedLabels = []
	#featureListTranspose = featureList.T
	zMatrix = np.dot(alpha, featureList.T)
	zMatrix = sigmoid(zMatrix)
	biasArray = np.matrix(np.ones(len(featureList)))
	zMatrix = np.concatenate((biasArray, zMatrix), axis = 0)
	yHatMatrix = softmax(np.dot(beta, zMatrix))
	for column in yHatMatrix.T:
		predictedLabels.append(np.argmax(column))
	return predictedLabels


def getError(predictedLabels, actualLabels):
	errorCount = 0
	record = 0
	for label in predictedLabels:
		if label != actualLabels[record]:
			errorCount = errorCount + 1
		record = record + 1

	return errorCount/record