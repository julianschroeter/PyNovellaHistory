system = "wcph113"

import pandas as pd
import numpy as np
import os
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from preprocessing.presetting import global_corpus_representation_directory
from classification.custom_classification import resample_boostrapped_LR

metadata_filepath = metadata_filepath= os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")
df = pd.read_csv(metadata_filepath, index_col=0)

df = df[["Gattungslabel_ED_normalisiert", "Nachname", "Gender", "Medientyp_ED", "Kanon_Status", "Jahr_ED"]]


values_dict = {"Gattungslabel_ED_normalisiert": ["N", "E"]}
df = df[df.isin(values_dict).any(1)]
df = df.dropna()

labels = df["Gattungslabel_ED_normalisiert"]


data =  df[["Nachname", "Gender", "Medientyp_ED", "Kanon_Status", "Jahr_ED"]]

columns_list = ["Nachname", "Gender", "Medientyp_ED", "Kanon_Status", "Jahr_ED"]

df_dummies = pd.get_dummies(data, columns=columns_list)

X = df_dummies.values
Y = labels.values

train_size = 0.80
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, train_size=train_size, random_state=seed)

print(len(Y_train), len(Y_validation))

model = LogisticRegression(solver="liblinear", multi_class="ovr")
model.fit(X_train, Y_train)

predictions = model.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


print("same process based on bootstrapped resampling with equal sample size:")

df_dummies = pd.concat([df_dummies, labels], axis=1)

print(df_dummies)


acc, f1 = resample_boostrapped_LR(n=1000, df=df_dummies, genre_category="Gattungslabel_ED_normalisiert",genre_labels=["N", "E"], train_size=0.8)

print("accuracy score results (all results, mean, std):")
print(acc)
print(np.mean(acc))
print(np.std(acc))

print("f1 score results (all results, mean, std):")
print(f1)
print(np.mean(f1))
print(np.std(f1))

model = RandomForestClassifier()
model.fit(X_train, Y_train)

predictions = model.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


print("same process based on bootstrapped resampling with equal sample size:")

df_dummies = pd.concat([df_dummies, labels], axis=1)

print(df_dummies)

from sklearn.model_selection import RandomizedSearchCV  # Number of trees in random forest

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
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


print(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune

# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                               random_state=42, n_jobs=-1)  # Fit the random search model
rf_random.fit(X_train, Y_train)
test_predictions = rf_random.best_estimator_.predict(X_validation)

print(rf_random.best_params_)
print(accuracy_score(Y_validation, test_predictions))
