import os
import sys
import pandas as pd
import numpy as np
import random as rnd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

src_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(src_dir)
os.chdir('reports')


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
combine = [train_df, test_df]

# train_df.info()
# print('_' * 40)
# test_df.info()

# train_df.describe()
# train_df.describe(include=['0'])

'''
# Analyze by pivoting features
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
'''

'''
# Analyze by visualizing data
# 1. Correlating Numerical Features
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)

# 2. Correlating numerical and ordinal features
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()

# 3. Correlating categorical features
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()

# 4.Correalting categorical and numerical features
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()
'''

# Wrangle Data
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)

# Creating new feature from existing feature
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)

# pd.crosstab(train_df['Title'], train_df['Sex'])

# Replace many titles with a more common name or classify them as Rare
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace(['Mlle', 'Ms'], 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

# train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

# Convert the categorical titles to ordinal
title_mapping = {"Mr": 1, "Miss": 2, "Mrs":3, "Master":4, "Rare":5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

# Now drop the Name Feature
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis = 1)
combine = [train_df, test_df]

# Converting a categorical feature
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)

# An empty array to contain guessed Age values based on Pclass x Gender combinations
guess_ages = np.zeros((2, 3))

for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()
            age_guess = guess_df.mean()
            # Convert random age float to nearest .5 age
            guess_ages[i, j] = int( age_guess/0.5 + 0.5 ) * 0.5

    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex==i) & (dataset.Pclass == j+1), 'Age' ] = guess_ages[i, j]
    dataset['Age'] = dataset['Age'].astype(int)

train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
# train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

# Replace Age with ordinals based on these bands
for dataset in combine:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] == 0
    dataset.loc[ (dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 64), 'Age'] = 5

# Removing the AgeBand feature.
train_df = train_df.drop(['AgeBand'], axis = 1)
combine = [train_df, test_df]

# Create new feature combining existing features
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

# train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)

for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

# train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()

# Dropping Parch, SibSp, and FamilySize features in favor of IsAlone.
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

# create an artificial feature combining Pclass and Age
for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

# Completing a categorical feature
'''
Embarked feature takes S, Q, C values based on port of embarkation. 
Our training dataset has two missing values. 
We simply fill these with the most common occurance.
'''
freq_port = train_df.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

# train_df[['Emabarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# convert the EmbarkedFill feature by creating a new numeric Port feature
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

# fare_mode = test_df['Fare'].dropna().median()
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].dropna().median())

train_df['FareBand'] = pd.cut(train_df['Fare'], 4)
# train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)

# Convert the Fare feature to ordinal values based on the FareBand
for dataset in combine:
    dataset.loc[ (dataset['Fare'] <= 7.91), 'Fare'] = 0
    dataset.loc[ (dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[(dataset['Fare'] > 31), 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]

# Now we are ready to train a model and predict the required solution.
'''
Logistic Regression
KNN or k-Nearest Neighbors
Support Vector Machines
Naive Bayes classifier
Decision Tree
Random Forrest
Perceptron
Artificial neural network
RVM or Relevance Vector Machine
'''

X_train = train_df.drop('Survived', axis=1)
Y_train = train_df['Survived']
X_test = test_df.drop('PassengerId', axis=1).copy()

# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred_log_reg = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df['Correlation'] = pd.Series(logreg.coef_[0])

# coeff_df.sort_values(by='Correlation', ascending=False)

# Support Vector Machines
svc =SVC()
svc.fit(X_train, Y_train)
Y_pred_svc = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

# KNN, k-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)
Y_pred_knn = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred_nb = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train)*100, 2)

# Perceptron
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred_perc = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train)*100, 2)

# Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred_lsvc = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train)*100, 2)

# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred_sgd = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train)*100, 2)

# Decission Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred_dt = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train)*100, 2)

# Random Forest
random_forest = RandomForestClassifier()
random_forest.fit(X_train, Y_train)
Y_pred_rf = random_forest.predict(X_test)
acc_random_forest = round(random_forest.score(X_train, Y_train)*100, 2)

# Model Evaluation
models = pd.DataFrame({'Model':['Support Vector Machines', 'KNN', 'Logistic Regression',
                                'Random Forest', 'Naive Bayes', 'Perceptron',
                                'Stochastic Gradient Descent', 'Linear SVC', 'Decision Tree'],
                       'Score':[acc_svc, acc_knn, acc_log, acc_random_forest,
                                acc_gaussian, acc_perceptron, acc_sgd, acc_linear_svc,
                                acc_decision_tree]})
models.sort_values(by='Score',ascending=False)

print(models)

# Output csv file's

# Support Vector Machines
submission_svc = pd.DataFrame({'PassengerId': test_df['PassengerId'],'Survived': Y_pred_svc})
if(os.path.exists('titanic_pred_svc.csv')):
    pass
else:
    submission_svc.to_csv('titanic_pred_svc.csv', index=False)

# Decision Tree
submission_decision_tree = pd.DataFrame({'PassengerId': test_df['PassengerId'],'Survived': Y_pred_dt})
if(os.path.exists('titanic_pred_decision_tree.csv')):
    pass
else:
    submission_decision_tree.to_csv('titanic_pred_decision_tree.csv', index=False)

# Random Forest
submission_random_forest = pd.DataFrame({'PassengerId': test_df['PassengerId'],'Survived': Y_pred_rf})
if(os.path.exists('titanic_pred_random_forest.csv')):
    pass
else:
    submission_random_forest.to_csv('titanic_pred_random_forest.csv', index=False)

# KNN
submission_knn = pd.DataFrame({'PassengerId': test_df['PassengerId'],'Survived': Y_pred_knn})
if(os.path.exists('titanic_pred_knn.csv')):
    pass
else:
    submission_knn.to_csv('titanic_pred_knn.csv', index=False)

# Naive Bayes
submission_nb = pd.DataFrame({'PassengerId': test_df['PassengerId'],'Survived': Y_pred_nb})
if(os.path.exists('titanic_pred_naive_bayes.csv')):
    pass
else:
    submission_nb.to_csv('titanic_pred_naive_bayes.csv', index=False)

# Stochastic Gradient Descent
submission_sgd = pd.DataFrame({'PassengerId': test_df['PassengerId'],'Survived': Y_pred_sgd})
if(os.path.exists('titanic_pred_gradient_descent.csv')):
    pass
else:
    submission_sgd.to_csv('titanic_pred_gradient_descent.csv', index=False)

# Linear SVC
submission_lsvc = pd.DataFrame({'PassengerId': test_df['PassengerId'],'Survived': Y_pred_lsvc})
if(os.path.exists('titanic_pred_linear_svc.csv')):
    pass
else:
    submission_lsvc.to_csv('titanic_pred_linear_svc.csv', index=False)

# Logistic Regression
submission_log_reg = pd.DataFrame({'PassengerId': test_df['PassengerId'],'Survived': Y_pred_log_reg})
if(os.path.exists('titanic_pred_logistic_reg.csv')):
    pass
else:
    submission_log_reg.to_csv('titanic_pred_logistic_reg.csv', index=False)

# Perceptron
submission_perc = pd.DataFrame({'PassengerId': test_df['PassengerId'],'Survived': Y_pred_perc})
if(os.path.exists('titanic_pred_perceptron.csv')):
    pass
else:
    submission_perc.to_csv('titanic_pred_perceptron.csv', index=False)