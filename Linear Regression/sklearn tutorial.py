from sklearn.naive_bayes import GaussianNB
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1,1], [2,1], [3,2]])
Y = np.array([1,1,1,2,2,2])
clf = GaussianNB()
clf.fit(X,Y)
pred = clf.predict([[2, 1]])
print(pred)
print(accuracy_score(Y, pred))
#0.009
#0.099