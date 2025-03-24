system = "my_xps"  # "wcph113" #
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')

from preprocessing.presetting import global_corpus_representation_directory, language_model_path, local_temp_directory
from preprocessing.corpus import DocFeatureMatrix
from preprocessing.metadata_transformation import years_to_periods, full_genre_labels
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
from scipy.special import expit
from sklearn.linear_model import LinearRegression, LogisticRegression

lang = "en" # "de"

my_model_de = language_model_path(system)

infile_df_path = os.path.join(local_temp_directory(system), "novella_corpus_length_matrix.csv")
metadata_df_path = os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")

matrix_obj = DocFeatureMatrix(data_matrix_filepath=infile_df_path, metadata_csv_filepath=metadata_df_path)
matrix_obj = matrix_obj.reduce_to(["token_count"])
matrix_obj = matrix_obj.add_metadata(["Gattungslabel_ED_normalisiert", "Jahr_ED", "Medientyp_ED"])
matrix_obj.data_matrix_df = years_to_periods(matrix_obj.data_matrix_df, category_name="Jahr_ED", start_year=1770, end_year=1970, epoch_length=20, new_periods_column_name="periods5a")

cat_labels = ["N", "E", "0E", "XE", "R"]
cat_labels = ["R", "M"]
matrix_obj = matrix_obj.reduce_to_categories("Gattungslabel_ED_normalisiert", cat_labels)

df = matrix_obj.data_matrix_df

replace_dict = {"Gattungslabel_ED_normalisiert": {"N": "Novelle", "E": "Erzählung", "0E": "kein Label",
                                    "R": "Roman", "M": "Märchen", "XE": "andere Label"}}

replace_dict = {"Gattungslabel_ED_normalisiert": {"N": 0, "E": 0, "0E": 0,
                                    "R": 1, "M": 0 , "XE": 0}}

df = full_genre_labels(df, replace_dict=replace_dict)

df = df.loc[:, ["token_count", "Gattungslabel_ED_normalisiert"]]

sample0 = df[df["Gattungslabel_ED_normalisiert"] == 0].sample(n=30).token_count.values
sample1 = df[df["Gattungslabel_ED_normalisiert"] == 1].sample(n=80).token_count.values


X = df.token_count.values.reshape(-1,1)
y = df.Gattungslabel_ED_normalisiert.values






# Fit the classifier
clf = LogisticRegression()
clf.fit(X, y)

# and plot the result
fig, axes = plt.subplots(1,2, figsize=(12,6))

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.3)
colors_list = ["orange" if x == 0 else "blue" for x in y_train.tolist()]

#plot logistic regression curve
sns.regplot(x=X_train, y=y_train, logistic=True, ci=None, scatter_kws={'color': colors_list}, line_kws={'color': 'red'}, ax=axes[0])
#axes[0].set_xlabel("Textlänge in Wort-Token")

axes[0].set_title("Training")
axes[0].set_xlim(0,250000)

X_test = np.linspace(0, 100000, 1000)
loss = expit(X_test * clf.coef_ + clf.intercept_).ravel()
if lang == "de":
    axes[1].plot(X_test, loss, label="Modell", color="red", linewidth=3)
else:
    axes[1].plot(X_test, loss, label="Model", color="red", linewidth=3)

sample0_loss = expit(sample0 * clf.coef_ + clf.intercept_).ravel()
axes[1].scatter(sample0, sample0_loss, label="Märchen", color="orange")
sample1_loss = expit(sample1 * clf.coef_ + clf.intercept_).ravel()
axes[1].scatter(sample1, sample1_loss, label="Romane", color="blue")
axes[1].set_xlim(0,60000)
if lang == "en":
    axes[1].set_title("Prediction")
else:
    axes[1].set_title("Klassifikation")
axes[1].axhline(y=0.5 + 0.03/2, c="black")
axes[1].axhline(y=0.5 - 0.03/2, c="black")
axes[1].legend()

print(loss)
print(clf.coef_, clf.intercept_)


if lang == "de":
    fig.supylabel("Wahrscheinlichkeit für die Vorhersage: Roman")
    fig.suptitle("Funktionsweise Logistischer Regression")
    fig.supxlabel("Textlänge in Wort-Token")
    fig.tight_layout()
    fig.savefig(os.path.join(local_temp_directory(system), "figures",
                             "Abb_Beispiel_Training-und-Klassifikation_Log_Reg_Länge.svg"))
    plt.show()
elif lang == "en":
    fig.supylabel("Predictive probability for Roman")
    fig.suptitle("The machinery of logistic regression")
    fig.supxlabel("Text length in word tokens")
    fig.tight_layout()
    fig.savefig(os.path.join(local_temp_directory(system), "figures",
                             "Figure_en_Training-and-Klassifikation_Log_Reg_Length.svg"))
    plt.show()


