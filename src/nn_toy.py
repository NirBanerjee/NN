import numpy as np
import sys
import math
np.set_printoptions(precision=8, suppress=True)

def sigmoid(vector):
	return 1 /(1 + np.exp(-vector))

def softmax(vector):
	return np.exp(vector)

if __name__ == '__main__':
	inputFeature = [1,1,0,0,1,1]
	labels =[1]

	inputFeature = np.asarray(inputFeature)
	outputLabel = np.asarray(labels)

	alpha = np.matrix([[1, 1, 2, -3, 0, 1, -3], [1, 3, 1, 2, 1, 0, 2], [1, 2, 2, 2, 2, 2, 1], [1, 1, 0, 2, 1, -2, 2]])
	beta = np.matrix([[1, 1, 2, -2, 1], [1, 1, -1, 1, 2], [1, 3, 1, -1, 1]])

	#Add Bias Term
	inputFeature = np.insert(inputFeature, 0, 1)
	inputFeature = np.matrix(inputFeature)

	#Compute VectorA
	vectorA = np.dot(alpha, inputFeature.T)
	print("========VectorA========")
	print(vectorA)
	print("\n")
	vectorZ = sigmoid(vectorA)
	print("========VectorZ========")
	print(vectorZ)
	print("\n")
	vectorZ = np.vstack([[1], vectorZ])
	vectorB = np.dot(beta, vectorZ)
	print("========vectorB========")
	print(vectorB)
	print("\n")
	yHat = softmax(vectorB)
	yHat = yHat/sum(yHat)
	print("========yHat========")
	print(yHat)
	print("\n")

	#Backprop
	loss = math.log(yHat[1][0])
	loss = -loss
	print("========Loss========")
	print(loss)
	print("\n")
	gradB = yHat
	gradB[1][0] = gradB[1][0] - 1
	gradBeta = np.dot(gradB, vectorZ.T)
	print("========GradBeta========")
	print(gradBeta)
	print("\n")
	gradZ = np.dot(beta.T, gradB)
	print("========GradZ========")
	print(gradZ)
	print("\n")
	gradA = np.multiply(vectorZ, (1 - vectorZ))
	gradA = np.multiply(gradZ, gradA)
	gradA = np.delete(gradA, 0, axis = 0)
	print("========GradA========")
	print(gradA)
	print("\n")
	gradAlpha = np.dot(gradA, inputFeature)
	print("========GradAlpha========")
	print(gradAlpha)
	print("\n")

	alpha = alpha - gradAlpha
	beta = beta - gradBeta

	print("========Updated Alpha========")
	print(alpha)
	print("\n")

	print("========Updated Beta========")
	print(beta)
	print("\n")

	#Predict Again
	vectorA = np.dot(alpha, inputFeature.T)
	vectorZ = sigmoid(vectorA)
	vectorZ = np.vstack([[1], vectorZ])
	vectorB = np.dot(beta, vectorZ)
	yHat = softmax(vectorB)
	yHat = yHat/sum(yHat)
	print("========Updated Class========")
	print(yHat)
	print("\n")


