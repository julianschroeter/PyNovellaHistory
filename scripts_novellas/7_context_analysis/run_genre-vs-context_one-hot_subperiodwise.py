system = "my_xps" #  "wcph113"

import pandas as pd
import numpy as np
import os
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from preprocessing.presetting import global_corpus_representation_directory
from preprocessing.metadata_transformation import years_to_periods
from classification.custom_classification import resample_boostrapped_LR

metadata_filepath = metadata_filepath= os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")
df = pd.read_csv(metadata_filepath, index_col=0)

df = df[["Gattungslabel_ED_normalisiert", "Nachname", "Gender", "Medientyp_ED", "Kanon_Status", "Jahr_ED"]]

values_dict = {"Gattungslabel_ED_normalisiert": ["N", "E"]}
df = df[df.isin(values_dict).any(axis=1)]
df = df.dropna()


data =  df[[ "Jahr_ED", "Gattungslabel_ED_normalisiert","Medientyp_ED", "Nachname"]] # "Gender", ,"Kanon_Status"
data = years_to_periods(input_df=data, category_name="Jahr_ED", start_year=1790, end_year=1950, epoch_length=30,
                      new_periods_column_name="periods")


columns_list = ["Medientyp_ED", "Jahr_ED", "Nachname"] #,"Gender", , "Kanon_Status"

train_size = 0.80
seed = 7

periods = list(set(data.periods.values.tolist()))
print(periods)
periods = [x for x in periods if x != 0]
periods.sort()
for period in periods:
    print("period is: ", period)
    period_data = data[data["periods"] == period]
    labels = period_data["Gattungslabel_ED_normalisiert"]
    period_data = period_data.drop(columns=["periods", "Gattungslabel_ED_normalisiert", "Jahr_ED"])

    df_dummies = pd.get_dummies(period_data, columns=columns_list)


    X = df_dummies.values
    Y = labels.values
    print("sample size: ",len(X), len(Y))

    n = 10
    lr_acc_scores = []
    for i in range(n):
        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, train_size=train_size)

        model = LogisticRegression(solver="liblinear", multi_class="ovr")
        model.fit(X_train, Y_train)
        predictions = model.predict(X_validation)
        lr_acc_scores.append(accuracy_score(Y_validation, predictions))
    print(len(Y_train), len(Y_validation))
    print("mean of accuracy scores for LR:")
    print(np.mean(np.array(lr_acc_scores)))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))

    rf_acc_scores = []
    for i in range(n):
        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, train_size=train_size)
        model = RandomForestClassifier()
        model.fit(X_train, Y_train)
        predictions = model.predict(X_validation)
        rf_acc_scores.append(accuracy_score(Y_validation, predictions))
    print("mean of accuracy scores for RF:")
    print(np.mean(np.array(rf_acc_scores)))
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
