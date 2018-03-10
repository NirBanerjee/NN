import csv

def readDataFile(fileName):
	dataFrame = []
	with open(fileName, 'r') as csvFile:
		for row in csvFile:
			dataFrame.append(row)

	return dataFrame