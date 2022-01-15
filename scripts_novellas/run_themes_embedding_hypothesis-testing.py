from TopicModeling.Postprocessing import DocTopicMatrix
from preprocessing.presetting import language_model_path, vocab_lists_dicts_directory, global_corpus_representation_directory, load_stoplist, set_DistReading_directory, mallet_directory
from clustering.my_pca import PC_df
from preprocessing.metadata_transformation import years_to_periods
from preprocessing.corpus import DocFeatureMatrix
from preprocessing.SamplingMethods import equal_sample
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
from sklearn.metrics import accuracy_score, classification_report
from sklearn import tree

system_name = "wcph113" # "my_mac" # "wcph104"

data_matrix_filepath = os.path.join(global_corpus_representation_directory(system_name), "DocThemesMatrix.csv")
language_model = language_model_path(system_name)
metadata_csv_filepath = os.path.join(global_corpus_representation_directory(system_name=system_name), "Bibliographie.csv")
print(data_matrix_filepath)
print(metadata_csv_filepath)
df = pd.read_csv(os.path.join(global_corpus_representation_directory(system_name), "Bibliographie.csv"), index_col=0)

colors_list = load_stoplist(os.path.join(vocab_lists_dicts_directory(system_name), "my_colors.txt"))

textanalytic_metadata_filepath = os.path.join(global_corpus_representation_directory(system_name), "textanalytic_metadata.csv")

matrix = DocFeatureMatrix(data_matrix_filepath= data_matrix_filepath, data_matrix_df=None, metadata_df=None,
                                  metadata_csv_filepath = textanalytic_metadata_filepath, mallet=False)


matrix = matrix.reduce_to(["lieben", "lieben.1", "lieben.2", "lieben.3", "lieben.4", "streicheln", "einzigartig", "einzigartig.1", "perfekt", "perfekt.2"])

matrix.data_matrix_df['love_mean'] = matrix.data_matrix_df.mean(axis=1)



matrix = matrix.reduce_to(["love_mean"]) # = bestimmtes Thema
print(matrix.data_matrix_df)


matrix = matrix.add_metadata("ende")
matrix = matrix.add_metadata("region")
matrix = matrix.add_metadata("liebesspannung")
matrix = matrix.add_metadata("titel")
matrix = matrix.add_metadata("stadt_land")

matrix.data_matrix_df["liebesspannung"] = matrix.data_matrix_df["liebesspannung"].fillna("other", inplace=False)

matrix.data_matrix_df.replace({"region": {"Italien": "rom",
                                                        "Spanien": "rom",
                                                        "Frankreich": "rom", "Lateinamerika" : "rom",
                                          "Karibik" : "rom", "Chile" : "rom", "Portugal":"rom",
                                          "Deutschland" : "non_rom", "Österreich" : "non_rom",
                                          "Niederlande" : "non_rom", "Ungarn" : "non_rom", "Russlan" : "non_rom",
                                          "Polen" : "non_rom", "Schweden" : "non_rom", "Universum" : "non_rom", "Dänemark" :"non_rom",
                                          "Russland" : "non_rom", "Schweiz" : "non_rom", "Karibik" : "rom", "Alpen" : "non_rom",
                                          "Nordamerika" :"non_rom", "Meer":"non_rom"

                                          }}, inplace=True)

matrix.data_matrix_df.replace({"stadt_land": {"Bergdorf": "land",
                                                        "land": "land",
                                                        "Alpenregion": "land",
                                              "stadt" :"None", "stadt+land": "None"
                                          }}, inplace=True)

matrix.data_matrix_df.replace({"liebesspannung": {"gering": "nein",
                                                        "mittel": "nein",
                                                        "stark": "ja", "kaum": "nein"
                                          }}, inplace=True)


matrix.data_matrix_df["ende"] = matrix.data_matrix_df["ende"].fillna("other", inplace=False)
matrix.data_matrix_df["region"] = matrix.data_matrix_df["region"].fillna("other", inplace=False)



#matrix = matrix.reduce_to_categories("ende", ["schauer", "tragisch", "Liebesglück", "nein", "Erkenntnis",
 #                                                     "Entsagung", "tragisch (schwach)", "unbestimmt", "other"])


#matrix = matrix.reduce_to_categories("region", ["rom", "non_rom"])
#matrix = matrix.reduce_to_categories("liebesspannung", ["ja", "nein"])

df = matrix.data_matrix_df
scaler = StandardScaler()
df.iloc[:, :1] = scaler.fit_transform(df.iloc[:, :1].to_numpy())

matrix = matrix.eliminate(["stadt_land", "titel", "ende", "liebesspannung"])
final_df = matrix.data_matrix_df
print(final_df)

print(final_df.describe())

category = "region"# "liebesspannung" # "stadt_land" #
first_value ="rom" # "ja" #"land" #
second_value = "non_rom" #"nein" # "None" #≈

final_df.boxplot(by=category)
plt.show()



sample_df = final_df[final_df[category] == first_value]
counter_sample_df = final_df[final_df[category] == second_value]

class_df = equal_sample(sample_df, counter_sample_df)
class_df = class_df.sample(frac=1)

sample_df = sample_df.drop([category], axis=1)
counter_sample_df = counter_sample_df.drop([category], axis=1)



#keine_liebe_df = final_df[final_df["liebesspannung"] == 'ja']
#keine_liebe_df = keine_liebe_df.drop(["liebesspannung"], axis=1)
whole_sample = final_df.drop([category], axis=1)
pop_mean = whole_sample.mean()
print(sample_df)



from scipy import stats
F, p = stats.ttest_ind(sample_df, counter_sample_df)
print("F, p: ", F, p)

F, p = stats. ttest_1samp(sample_df, pop_mean)
print("F, p: ", F, p)

grouped_df = final_df.groupby([category]).mean()

print(grouped_df)

grouped_df = final_df.groupby([category]).std()
print("grouped std: ")
print(grouped_df)

grouped_df = final_df.groupby([category]).count()
print("grouped std: ")
print(grouped_df)

print("popmean: ", pop_mean)

final_df = final_df
matrix.data_matrix_df.replace({"ende": {"tragisch": "tragisch",
                                                        "schauer": "tragisch",
                                                        "Liebesglück": "positiv",
                                                       "nein": "positiv",
                                                       "Erkenntnis": "positiv",
                                                        "tragisch (schwach)" : "tragisch",
                                                        "unbestimmt" : "positiv",
                                                        "Entsagung" : "positiv"

                                                       }}, inplace=True)


new_df = matrix.data_matrix_df


# classification

lr_model = LogisticRegressionCV()
dt_clf = DecisionTreeClassifier(max_leaf_nodes=2)

array = class_df.to_numpy()
X = array[:, 0:(array.shape[1]-1)]
Y = array[:, array.shape[1]-1]


X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=0.2,random_state=2)
print(Y_train)
lr_model.fit(X_train, Y_train)
dt_clf.fit(X_train, Y_train)
predictions = lr_model.predict(X_validation)
print("Accuracy score: ", accuracy_score(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
print("coef:" , lr_model.coef_)
print("Schwellenwert: " , 0.5 / lr_model.coef_)


X_new = np.linspace(-1, 1, 1000).reshape(-1, 1)

y_proba = lr_model.predict_proba(X_new)


plt.plot(X_new, y_proba[:, 1], "g-", linewidth=2, label="Iris-Virginica")
plt.plot(X_new, y_proba[:, 0], "b--", linewidth=2, label="Not Iris-Virginica")
plt.show()


y_proba = lr_model.predict_proba(X_new)

print(X_new[y_proba[:,1]])
decision_boundary = X_new[y_proba[:, 1] >= 0.5][0]

plt.figure(figsize=(8, 3))
#plt.plot(X[y==0], y[y==0], "bs")
#plt.plot(X[y==1], y[y==1], "g^")
plt.plot([decision_boundary, decision_boundary], [-1, 2], "k:", linewidth=2)
plt.plot(X_new, y_proba[:, 1], "g-", linewidth=2, label="Iris-Virginica")
plt.plot(X_new, y_proba[:, 0], "b--", linewidth=2, label="Not Iris-Virginica")
plt.text(decision_boundary+0.02, 0.15, "Decision  boundary", fontsize=14, color="k", ha="center")
plt.arrow(decision_boundary, 0.08, -0.3, 0, head_width=0.05, head_length=0.1, fc='b', ec='b')
plt.arrow(decision_boundary, 0.92, 0.3, 0, head_width=0.05, head_length=0.1, fc='g', ec='g')
plt.xlabel("Petal width (cm)", fontsize=14)
plt.ylabel("Probability", fontsize=14)
plt.legend(loc="center left", fontsize=14)
plt.axis([0, 3, -0.02, 1.02])
#save_fig("logistic_regression_plot")
plt.show()

tree.plot_tree(dt_clf)
plt.show()