import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import math
#
# We read two sources, a training set and a testing set on which our algorithm will work. We use pandas to read from the
# csv source
unclean_train = pd.read_csv('../Data/results/copy_of_the_training_data.csv')
unclean_test = pd.read_csv('../Data/results/copy_of_the_testing_data.csv')
train = []
test = []
#
# We convert the data set into numpy arrays
unclean_train = np.array(unclean_train)
unclean_test = np.array(unclean_test)
#
# Convert sex feature to binary value [0 = male, 1 = female]
for row in unclean_train:
    if row[4] == 'male':
        row[4] = 0
    elif row[4] == 'female':
        row[4] = 1
    else:
        print('Invalid Data format for sex [training set]')
        exit(0)
for row in unclean_test:
    if row[4] == 'male':
        row[4] = 0
    elif row[4] == 'female':
        row[4] = 1
    else:
        print('Invalid Data format for sex [testing set]')
        exit(0)
#
# Clean data and training set for Nan values based on the picked filters
for row in unclean_train:
    if not math.isnan(row[4]) and not math.isnan(row[5]):
        train.append(row)
for row in unclean_test:
    if not math.isnan(row[4]) and not math.isnan(row[5]):
        test.append(row)
#
# We convert the data set into numpy arrays
n_train = np.array(train)
n_test = np.array(test)
#
# Initialize data banks
x_features = []
y_labels = []
#
# For every row in the training set, we extract the sex [0 = male, 1 = female] and age and store them in the x_
# features data bank. We do the same for the survivability of the data element and store it in the y_labels data bank
for row in n_train:
    feature = [row[4], row[5]] # Sex, Age
    x_features.append(feature)
    y_labels.append(row[1]) # Survivability
#
# Convert python lists to numpy arrays
x_nfeatures = np.array(x_features)
y_nlabels = np.array(y_labels)
#
# We use a Naive Bayes classifier
classifier = GaussianNB()
classifier.fit(x_nfeatures, y_nlabels)
#
# Testing the classifier on the test set
x_features = []
y_labels = []
for row in n_test:
    feature = [row[4], row[5]]  # Sex, Age
    x_features.append(feature)
    y_labels.append(row[1])
#
predicted = classifier.predict(x_features)
#
print(predicted)
print('Accuracy score of: ' + str(accuracy_score(y_labels[:len(predicted)], predicted)))