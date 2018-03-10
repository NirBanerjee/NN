import numpy as np 

def generateVectors(dataFrame):

	labelList = []
	featureList = []
	for row in dataFrame:
		row = row[:-1]
		elements = row.split(",")
		elements = [float(element) for element in elements]
		labelList.append(elements[0])
		featureList.append(elements[1:])

	return np.array(featureList), np.array(labelList)