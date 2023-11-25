from numpy import unique
from numpy import where
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import RandomizedSearchCV  # Number of trees in random forest
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# initialize the data set we'll work with
X,Y = make_classification(
    n_samples=500,
    n_features=10,
    n_informative=10,
    n_redundant=0,
    n_clusters_per_class=3,
    flip_y= 0.0001,
    class_sep=3,

)

# define the model

kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X)
                for k in range(1, 10)]
silhouette_scores = [silhouette_score(X, model.labels_)
                     for model in kmeans_per_k[1:]]

print(silhouette_scores)

plt.figure(figsize=(8, 3))
plt.plot(range(2, 10), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score", fontsize=14)
plt.axis([1.8, 8.5, 0.0, 0.8])
#plt.save_fig("silhouette_score_vs_k_plot")
plt.show()

kmeans = KMeans(n_clusters=20)

# assign each data point to a cluster
result = kmeans.fit_predict(X)
print(kmeans.cluster_centers_)

# get all of the unique clusters
clusters = unique(result)

# plot the clusters
for cluster in clusters:
    # get data points that fall in this cluster
    index = where(result == cluster)
    # make the plot
    plt.scatter(X[index, 0], X[index, 1])

# show the DBSCAN plot
plt.title("Clustering predictions – D1 +2 ")
plt.show()

for cluster in clusters:
    # get data points that fall in this cluster
    index = where(result == cluster)
    # make the plot
    plt.scatter(X[index, 1], X[index, 2])

# show the DBSCAN plot
plt.title("Clustering predictions – D2+3 ")
plt.show()

plt.scatter(X[:,0], X[:,1], c=Y)
plt.title("True labels – D1+2")
plt.show()

plt.scatter(X[:,1], X[:,2], c=Y)
plt.title("True labels – D2+3")
plt.show()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)


lr_model = LogisticRegression()
lr_model = LogisticRegressionCV(cv=3, solver='liblinear', multi_class="auto")
lr_model.fit(X_train, Y_train)
test_predictions = lr_model.predict(X_test)
print("Log Reg results:")
print(classification_report(Y_test, test_predictions))

rf_model = RandomForestClassifier()
rf_model.fit(X_train, Y_train)
test_predictions = rf_model.predict(X_test)
print("Random Forest results:")
print(classification_report(Y_test, test_predictions))



n_estimators = [int(x) for x in np.linspace(start=10, stop=1000, num=100)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]  # Create the random grid
random_grid = {'n_estimators': n_estimators,
          #     'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


print(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune

# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator=rf_model, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                               random_state=42, n_jobs=-1)  # Fit the random search model
rf_random.fit(X_train, Y_train)
test_predictions = rf_random.best_estimator_.predict(X_test)

print(rf_random.best_params_)
print(accuracy_score(Y_test, test_predictions))
