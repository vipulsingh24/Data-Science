import csv
import random
import math

def loadCsv(filename):
    lines = csv.reader(open(filename, 'r'))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset

def splitDataset(dataset, trainprop=0.20):
    trainSize = int(len(dataset) * trainprop)
    trainSet = []
    testSet = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(testSet))
        trainSet.append(testSet.pop(index))
    return [trainSet, testSet]

def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated

def mean(numbers):
    return sum(numbers) / float(len(numbers))

def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)

def summarize(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	del summaries[-1]
	return summaries

def summarizeByClass(dataset):
	separated = separateByClass(dataset)
	summaries = {}
	for classValue, instances in separated.items():
		summaries[classValue] = summarize(instances)
	return summaries

def calculateProb(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

def calculateClassProb(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            x = inputVector[i]
            probabilities[classValue] *= calculateProb(x,mean, stdev)
    return probabilities

def predict(summaries, inputVector):
    probabilities = calculateClassProb(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel

def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0

if __name__ == '__main__':
    filename = 'pima-indians-diabetes.data.csv'
    trainprop = 0.80
    dataset = loadCsv(filename)
    trainingSet, testSet = splitDataset(dataset, trainprop)
    print('Split {0} rows into train={1} and test={2} rows'.format(len(dataset), len(trainingSet), len(testSet)))
    # prepare model
    summaries = summarizeByClass(trainingSet)
    # test model
    predictions = getPredictions(summaries, testSet)
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: {0}%'.format(accuracy))


# dataset = [[1,20,0], [2,21,1], [3,22,0]]
# summary = summarize(dataset)
# print('Attribute summaries: {0}'.format(summary))

# dataset = [[1,2,3], [3,4,5], [4,5,6], [4,5,6], [40,50,60]]
# trainprop = 0.80
# train, test = splitDataset(dataset, trainprop=trainprop)
# print('Split {0} rows into train with {1} and test with {2}'.format(len(dataset), train, test))

# filename = 'pima-indians-diabetes.data.csv'
# dataset = loadCsv(filename)
# print('Loaded data file {0} with {1} rows'.format(filename, len(dataset)))
