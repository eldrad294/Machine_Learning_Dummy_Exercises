import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import math
#
# We read two sources, a training set and a testing set on which our algorithm will work. We use pandas to read from the
# csv source
train = pd.read_csv('../Data/results/copy_of_the_training_data.csv')
test = pd.read_csv('../Data/results/copy_of_the_testing_data.csv')
#
# We convert the data set into numpy arrays
n_train = np.array(train)
n_test = np.array(test)
#
# Initialize data banks
x_features = []
y_labels = []
#
# For every row in the training set, we extract the sex [0 = male, 1 = female] and age and store them in the x_features data bank. We do the
# same for the survivability of the data element and store it in the y_labels data bank
for row in n_train:
    #
    # Check for NaN values, and skip them
    temp_val = float(row[5])
    if math.isnan(temp_val):
        continue
    feature = [0 if row[4] == "male" else 1, row[5]] # Sex, Age
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
for row in n_test:
    #
    # Check for NaN values, and skip them
    temp_val = float(row[5])
    if math.isnan(temp_val):
        continue
    feature = [0 if row[4] == "male" else 1, row[5]]  # Sex, Age
    x_features.append(feature)
#
predicted = classifier.predict(x_features)
#
print(predicted)
