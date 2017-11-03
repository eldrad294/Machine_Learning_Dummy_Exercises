from sklearn.cluster import KMeans
import numpy as np
X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])
kmeans = KMeans(n_clusters=4, random_state=1).fit(X)
print(kmeans.predict([[0,0], [1, 1], [2, 2], [3, 3], [4,4], [5,5]]))