from scipy.spatial.distance import pdist
import numpy as np
from sklearn.cluster import KMeans
X = np.array([[1, 2], [4, 5], [1,3], [9,10]], np.int32)

print(X)

Y = pdist(X, 'euclidean')

print(Y)

kmeans = KMeans(n_clusters=1, random_state=0).fit(X)

print("center:")
print(kmeans.cluster_centers_)

print("metrics from center:")
dist = kmeans.transform(X)
print(dist)