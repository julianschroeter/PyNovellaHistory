system = "wcph113"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')

# from my own modules:
from preprocessing.presetting import global_corpus_representation_directory, local_temp_directory
from preprocessing.corpus import DocFeatureMatrix
from preprocessing.metadata_transformation import standardize_meta_data_medium, full_genre_labels, years_to_periods

# standard libraries
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.metrics import r2_score

import seaborn as sns

medium_cat = "Medientyp_ED"
genre_cat = "Gattungslabel_ED_normalisiert"


infile_df_path = os.path.join(local_temp_directory(system), "novella_corpus_length_matrix.csv")
metadata_df_path = os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")

matrix_obj = DocFeatureMatrix(data_matrix_filepath=infile_df_path, metadata_csv_filepath=metadata_df_path)
matrix_obj = matrix_obj.reduce_to(["token_count"])
matrix_obj = matrix_obj.add_metadata(["Gattungslabel_ED_normalisiert", "Jahr_ED", "Medientyp_ED"])

cat_labels = ["N", "E", "0E", "XE", "R", "M"]
matrix_obj = matrix_obj.reduce_to_categories("Gattungslabel_ED_normalisiert", cat_labels)

df = matrix_obj.data_matrix_df

df = df[df.isin({medium_cat:["Familienblatt", "Anthologie", "Taschenbuch", "Buch"]}).any(1)]
scaler = MinMaxScaler()
df.iloc[:, :1] = scaler.fit_transform(df.iloc[:, :1].to_numpy())

data = df.loc[:, ("token_count",genre_cat,medium_cat, "Jahr_ED")]

epoch_lengths = [20, 30, 50, 60]
genre_selections = ["all_genres", "MLP", "N-vs-MLP", "MLP+R", "MLP+M", "N-vs-E"]
media_selections = ["all_media", "TB-Ant-RS-FB", "TB-Ant-RS-FB-Journal", "TB-Journal-Book"]

triples = []
for i in epoch_lengths:
    for j in genre_selections:
        for k in media_selections:
            triple = (i, j, k)
            triples.append(triple)

best_model_score = 0
all_model_params = []
for triple in triples:
    i,j,k = triple
    df = data.copy()

    df = years_to_periods(df, category_name="Jahr_ED", start_year=1790, end_year=1941,
                     epoch_length=i, new_periods_column_name="periods")
    if j == "all_genres":
        cat_labels = ["N", "E", "0E", "XE", "R", "M"]
        df = df[df.isin({"Gattungslabel_ED_normalisiert": cat_labels}).any(1)]
    elif j == "MLP":
        cat_labels = ["N", "E", "0E", "XE"]
        df = df[df.isin({"Gattungslabel_ED_normalisiert": cat_labels}).any(1)]
        replace_dict = {"Gattungslabel_ED_normalisiert": {"N": "N", "E": "E", "0E": "MLP",
                                                        "XE": "MLP"}}
        df = full_genre_labels(df, replace_dict=replace_dict)
    elif j == "N-vs-MLP":
        cat_labels = ["N", "E", "0E", "XE"]
        df = df[df.isin({"Gattungslabel_ED_normalisiert": cat_labels}).any(1)]
        replace_dict = {"Gattungslabel_ED_normalisiert": {"N": "N", "E": "MLP", "0E": "MLP",
                                                          "XE": "MLP"}}
    elif j == "MLP+R":
        cat_labels = ["N", "E", "0E", "XE", "R"]
        df = df[df.isin({"Gattungslabel_ED_normalisiert": cat_labels}).any(1)]
        replace_dict = {"Gattungslabel_ED_normalisiert": {"N": "N", "E": "E", "0E": "MLP",
                                                          "XE": "MLP"}}
        df = full_genre_labels(df, replace_dict=replace_dict)
    elif j == "MLP+M":
        cat_labels = ["N", "E", "0E", "XE", "M"]
        df = df[df.isin({"Gattungslabel_ED_normalisiert": cat_labels}).any(1)]
        replace_dict = {"Gattungslabel_ED_normalisiert": {"N": "N", "E": "E", "0E": "MLP",
                                                          "XE": "MLP"}}
        df = full_genre_labels(df, replace_dict=replace_dict)
    elif j == "N-vs-E":
        cat_labels = ["N", "E"]
        df = df[df.isin({"Gattungslabel_ED_normalisiert": cat_labels}).any(1)]

    if k == "all_media":
        cat_labels = ["Taschenbuch", "Zeitschrift", "Zeitung", "Rundschau", "Anthologie", "Familienblatt", "Sammlung", "Werke","Buch"]
        df = df[df.isin({"Medientyp_ED": cat_labels}).any(1)]
        replace_dict = {"Medientyp_ED": {"Zeitung": "Journal", "Zeitschrift": "Journal",
                                         "Werke": "Buch"}}
        df = full_genre_labels(df, replace_dict=replace_dict)

    elif k == "TB-Ant-RS-FB":
        cat_labels = ["Taschenbuch", "Rundschau", "Anthologie", "Familienblatt"]
        df = df[df.isin({"Medientyp_ED": cat_labels}).any(1)]
    elif k == "TB-Ant-RS-FB-Journal":
        cat_labels = ["Taschenbuch", "Rundschau", "Anthologie", "Familienblatt", "Zeitung", "Zeitschrift"]
        df = df[df.isin({"Medientyp_ED": cat_labels}).any(1)]
        replace_dict = {"Medientyp_ED": {"Zeitung": "Journal", "Zeitschrift": "Journal"}}
        df = full_genre_labels(df, replace_dict=replace_dict)

    elif k == "TB-Journal-Book":
        cat_labels = ["Taschenbuch", "Rundschau", "Anthologie", "Familienblatt", "Zeitung", "Zeitschrift", "Buch", "Werke"]
        df = df[df.isin({"Medientyp_ED": cat_labels}).any(1)]
        replace_dict = {"Medientyp_ED": {"Zeitung": "Journal", "Zeitschrift": "Journal",
                                         "Rundschau": "Journal", "Familienblatt":"Journal",
                                         "Werke":"Buch"}}
        df = full_genre_labels(df, replace_dict=replace_dict)

    variables = ["Gattungslabel_ED_normalisiert", "Medientyp_ED", "periods", "token_count"]
    predicted_var = variables[3]
    predictor_var = (variables[0], variables[1], variables[2])
    Y = df[predicted_var].values
    X = df.loc[:, predictor_var]

    X = pd.get_dummies(data=X, drop_first=False)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=15)

    model = LinearRegression()
    model.fit(X_train, y_train)
    coeff_parameter = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
    predictions = model.predict(X_test)



    # compare r2 for train and test sets

    oos_current_model_r2 = r2_score(y_test, model.predict(X_test))
    train_current_model_r2 = r2_train = r2_score(y_train, model.predict(X_train))

    if oos_current_model_r2 > best_model_score:
        best_model_params = triple
        best_model_score = oos_current_model_r2
        best_df = df

    all_model_params.append([i,j,k,oos_current_model_r2, train_current_model_r2])
from operator import itemgetter
sorted_params = sorted(all_model_params, key = itemgetter(3))

params_df = pd.DataFrame(sorted_params)
print(params_df)

print("best model: parameters: ", best_model_params, best_model_score)

data = best_df
# model without interaction between medum and genre:

# model with interaction between medum and genre:


print("Model including an interaction between genre and periods:")
lm = ols('token_count ~ C(Gattungslabel_ED_normalisiert, Sum)+C(Medientyp_ED, Sum) + C(periods, Sum) + C(Gattungslabel_ED_normalisiert, Sum)*C(periods, Sum)',
                 data=data).fit()
print("Model including an interaction between medium and periods:")
lm = ols('token_count ~ C(Gattungslabel_ED_normalisiert, Sum)+C(Medientyp_ED, Sum) + C(periods, Sum) + C(Medientyp_ED, Sum)*C(periods, Sum)',
                 data=data).fit()
print("Model without interaction:")
lm = ols('token_count ~ C(Gattungslabel_ED_normalisiert, Sum)+C(Medientyp_ED, Sum) + C(periods, Sum)',
                 data=data).fit()
print("Model including an interaction between genre and medium:")
lm = ols('token_count ~ C(Gattungslabel_ED_normalisiert, Sum)+C(Medientyp_ED, Sum) + C(periods, Sum) + C(Gattungslabel_ED_normalisiert, Sum)*C(Medientyp_ED, Sum)',
                 data=data).fit()


anova_table = sm.stats.anova_lm(lm, typ=2) # Type 2  Anova DataFrame
anova_ann = "Results of the Anova test:"
print(anova_ann)
print(anova_table)

deltas_df = anova_table.copy()
deltas_df["deltas"] = anova_table["sum_sq"] / anova_table["df"]
print(deltas_df)

lm_summary = lm.summary()
print(lm_summary)


variables = ["Gattungslabel_ED_normalisiert", "Medientyp_ED", "periods", "token_count"]
predicted_var = variables[3]
predictor_var = (variables[0], variables[1], variables[2])
Y = df[predicted_var].values
X = df.loc[:,predictor_var]

# independent variables to one-hot encoding:
X = pd.get_dummies(data=X, drop_first=False)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=42)
print(X_train.shape)
print(X_test.shape)

# do a scikit-learn linear regression
model = LinearRegression()
model.fit(X_train,y_train)
coeff_parameter = pd.DataFrame(model.coef_,X.columns,columns=['Coefficient'])
print(coeff_parameter)
predictions = model.predict(X_test)
sns.regplot(x= y_test, y= predictions)
sns.regplot(x = y_train, y= model.predict(X_train), color="red")
plt.show()

from sklearn.metrics import r2_score
r2_train = r2_score(y_train, model.predict(X_train))
r2_test = r2_score(y_test, model.predict(X_test))
# compare r2 for train and test sets
print("r2 train: ", r2_train, " | r2 test: ", r2_test)


X_train_Sm = sm.add_constant(X_train)
ls = sm.OLS(y_train,X_train_Sm).fit()
print("Predictor variables: ", predictor_var)
print("Predicted variables: ", predicted_var)
print(ls.summary())


