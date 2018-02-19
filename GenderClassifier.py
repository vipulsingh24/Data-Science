import numpy as np
from sklearn import tree
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# Data
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47],
     [175, 64, 39], [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]
# Label
Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

# Classifiers
clf_tree = tree.DecisionTreeClassifier()
clf_svm = SVC()
clf_perceptron = Perceptron()
clf_KNN = KNeighborsClassifier()

# Training the models
clf_tree.fit(X, Y)
clf_svm.fit(X, Y)
clf_perceptron.fit(X, Y)
clf_KNN.fit(X, Y)

# Testing the data
pred_tree = clf_tree.predict(X)
pred_svm = clf_svm.predict(X)
pred_perceptron = clf_perceptron.predict(X)
pred_KNN = clf_KNN.predict(X)

# Checking the accuracy
acc_tree = accuracy_score(Y, pred_tree) * 100
print('Accuracy for Decision Tree: {}'.format(acc_tree))

acc_svm = accuracy_score(Y, pred_svm) * 100
print('Accuracy for SVM: {}'.format(acc_svm))

acc_perceptron = accuracy_score(Y, pred_perceptron) * 100
print('Accuracy for Perceptron: {}'.format(acc_perceptron))

acc_KNN = accuracy_score(Y, pred_KNN) * 100
print('Accuracy for K-Nearest Neighbour: {}'.format(acc_KNN))

# Best Classifier
index = np.argmax([acc_tree, acc_svm, acc_perceptron, acc_KNN])
classifier = {0: 'Decision Tree', 1: 'SVM', 2: 'Perceptron', 3: 'KNearest Neighbour'}
print('Best gender classifier is {}' .format(classifier[index]))

