import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.metrics import accuracy_score
#
# Import training set into panda structures
df = pd.read_csv('../data/results/copy_of_the_training_data.csv')
df = df[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Survived']].dropna(axis=0, how='any').replace(['male','female'],[0,1]).replace(['C','S','Q'],[1,2,3])
df_label = df[['Survived']].values
df = df[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']].values
#df /= df.max()
df = preprocessing.normalize(df)
#
# df_Pclass = df[['Pclass']].dropna(axis=0, how='any').values
# df_Sex = df[['Sex']].dropna(axis=0, how='any').replace(['male','female'],[0,1]).values
# df_Age = df[['Age']].dropna(axis=0, how='any').values
# df_SibSp = df[['SibSp']].dropna(axis=0, how='any').values
# df_Parch = df[['Parch']].dropna(axis=0, how='any').values
# df_Fare = df[['Fare']].dropna(axis=0, how='any').values
# df_Embarked = df[['Embarked']].dropna(axis=0, how='any').replace(['C','S','Q'],[1,2,3]).values
#
# plt.figure()
# plt.xlabel('Age'), plt.ylabel('Sibsp'), plt.axis([0,85,-0.5,9]), [(plt.plot(dp[0],dp[1],'ro')) for dp in df_age_sibsp], plt.show()
# plt.xlabel('Age'), plt.ylabel('Parch'), plt.axis([0,85,-0.5,1.5]), [(plt.plot(dp[0],dp[1],'ro')) for dp in df_age_parch], plt.show()
# plt.xlabel('Fare'), plt.ylabel('Embarked'), plt.axis([0,300,0,4]), [(plt.plot(dp[0],dp[1],'ro')) for dp in df_fare_embarked], plt.show()
# plt.xlabel('Fare'), plt.ylabel('Pclass'), plt.axis([0,300,0,4]), [(plt.plot(dp[0],dp[1],'ro')) for dp in df_fare_pclass], plt.show()
# plt.xlabel('Survived'), plt.ylabel('Age'), plt.axis([-1,2,0,85]), [(plt.plot(dp[0],dp[1],'ro')) for dp in df_suvived_age], plt.show()
# plt.xlabel('Survived'), plt.ylabel('Pclass'), plt.axis([-1,2,0,4]), [(plt.plot(dp[0],dp[1],'ro')) for dp in df_survived_pclass], plt.show()
# plt.xlabel('Survived'), plt.ylabel('SibSp'), plt.axis([-1,2,-0.5,9]), [(plt.plot(dp[0],dp[1],'ro')) for dp in df_survived_sibsp], plt.show()
#
weights = [10,10,10,10,10,10,10,10]
labels = []
[(labels.append(int(x))) for x in df_label]
features = []
[(features.append(list(x))) for x in df]
alpha = .01
#
def wtx(features):
    total = weights[0] * 1
    for i, feature in enumerate(features):
        total += feature * weights[i+1]
    return total
#
sigmoid = lambda z : 1 / (1 + math.exp(-z))
cost_function = lambda y,yhat : -y*math.log(yhat) - ((1-y)*math.log(1-yhat))
train = lambda w,y,yhat : w - alpha * cost_function(y,yhat)
determine = lambda p : 1 if p >= .5 else 0
#
# Train the model by adjusting weights
for i, row in enumerate(features):
    yhat = sigmoid(1*weights[0])
    weights[0] = train(weights[0], labels[i], yhat)
    for j, feature in enumerate(row):
        yhat = sigmoid(feature*weights[j+1])
        weights[j+1] = train(weights[j+1], labels[i], yhat)
#
logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(features, labels)
#
#######################################################################
#
df = pd.read_csv('../data/results/Test.csv')
df = df[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked', 'Survived']].dropna(axis=0, how='any').replace(['male','female'],[0,1]).replace(['C','S','Q'],[1,2,3])
df_label_test = df_label = df[['Survived']].values
df = df[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']].values
#df /= df.max()
df = preprocessing.normalize(df)
#
features = []
expected_labels = []
predicted_labels = []
[(expected_labels.append(int(x))) for x in df_label_test]
[(features.append(list(x))) for x in df]
#
for row in features:
    yhat = sigmoid(wtx(row))
    predicted_labels.append(determine(yhat))
#
def accuracy(expected, predicted):
    correct = 0
    for i in range(len(expected)):
        if expected[i] == predicted[i]:
            correct += 1
    return (correct / len(expected)) * 100
#
print('My Accuracy: ' + str(accuracy(expected=expected_labels,predicted=predicted_labels)) + '%')
#
pred = logreg.predict(df)
#
print('Lib Accuracy:' + str(accuracy_score(predicted_labels, pred)*100) + '%')






