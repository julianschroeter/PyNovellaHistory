from topicmodeling.Postprocessing import DocTopicMatrix
from preprocessing.presetting import language_model_path, vocab_lists_dicts_directory, local_temp_directory, global_corpus_representation_directory, load_stoplist, set_DistReading_directory, mallet_directory
from preprocessing.corpus_alt import DocFeatureMatrix
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
from sklearn.metrics import accuracy_score, classification_report
from sklearn import tree

system_name = "wcph113" # "my_mac" # "wcph104"
data_matrix_filepath = os.path.join(local_temp_directory(system_name), "output_composition_100topics.txt")
language_model = language_model_path(system_name)
metadata_csv_filepath = os.path.join(global_corpus_representation_directory(system_name=system_name), "Bibliographie.csv")

df = pd.read_csv(os.path.join(global_corpus_representation_directory(system_name), "Bibliographie.csv"), index_col=0)
print(df)

colors_list = load_stoplist(os.path.join(vocab_lists_dicts_directory(system_name), "my_colors.txt"))
textanalytic_metadata_filepath = os.path.join(global_corpus_representation_directory(system_name), "textanalytic_metadata.csv")
topic_doc_matrix = DocTopicMatrix(data_matrix_filepath= data_matrix_filepath, data_matrix_df=None, metadata_df=None,
                                  metadata_csv_filepath = textanalytic_metadata_filepath, mallet=True)

topic_doc_matrix = topic_doc_matrix.adjust_doc_chunk_multiindex()
topic_doc_matrix.data_matrix_df.to_csv(os.path.join(global_corpus_representation_directory(system_name), "test-doc-topics.csv"))
df_all_segments = topic_doc_matrix.data_matrix_df
matrix = topic_doc_matrix.mean_doclevel()
doc_topic_df = matrix.data_matrix_df
doc_topic_df = doc_topic_df.head(-7)


matrix = DocFeatureMatrix(data_matrix_filepath=None, data_matrix_df = doc_topic_df,
                         metadata_df=None,metadata_csv_filepath=textanalytic_metadata_filepath,
                        mallet=False)


matrix = matrix.reduce_to([34]) # = bestimmtes topic



matrix = matrix.add_metadata("ende")
matrix = matrix.add_metadata("region")
matrix = matrix.add_metadata("liebesspannung")
matrix = matrix.add_metadata("titel")
matrix = matrix.add_metadata("stadt_land")
matrix.data_matrix_df.replace({"region": {"Italien": "rom",
                                                        "Spanien": "rom",
                                                        "Frankreich": "rom", "Lateinamerika" : "rom",
                                          "Karibik" : "rom", "Chile" : "rom",
                                          "Deutschland" : "non_rom", "Österreich" : "non_rom",
                                          "Niederlande" : "non_rom", "Ungarn" : "non_rom", "Russlan" : "non_rom",
                                          "Polen" : "non_rom", "Schweden" : "non_rom"
                                          }}, inplace=True)

matrix.data_matrix_df.replace({"stadt_land": {"Bergdorf": "land",
                                                        "land": "land",
                                                        "Alpenregion": "land",
                                              "stadt" :"None", "stadt+land": "None"
                                          }}, inplace=True)

matrix.data_matrix_df.replace({"liebesspannung": {"gering": "nein",
                                                        "mittel": "ja",
                                                        "stark": "ja", "kaum": "nein"
                                          }}, inplace=True)


matrix.data_matrix_df["ende"] = matrix.data_matrix_df["ende"].fillna("other", inplace=False)
matrix.data_matrix_df["region"] = matrix.data_matrix_df["region"].fillna("other", inplace=False)
matrix.data_matrix_df["liebesspannung"] = matrix.data_matrix_df["liebesspannung"].fillna("other", inplace=False)


#matrix = matrix.reduce_to_categories("ende", ["schauer", "tragisch", "Liebesglück", "nein", "Erkenntnis",
 #                                                     "Entsagung", "tragisch (schwach)", "unbestimmt", "other"])


#matrix = matrix.reduce_to_categories("region", ["rom", "non_rom"])
#matrix = matrix.reduce_to_categories("liebesspannung", ["ja", "nein"])

df = matrix.data_matrix_df
matrix = matrix.eliminate(["region", "titel", "ende", "stadt_land"])
final_df = matrix.data_matrix_df
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
final_df.iloc[:, :1] = scaler.fit_transform(final_df.iloc[:, :1].to_numpy())
print(final_df)

print(final_df.describe())


category = "liebesspannung" # "stadt_land" # "region"

final_df.boxplot(by=category)
plt.show()



first_value = "ja" # "land" # "rom"
second_value = "nein" # "None" #"non_rom"

final_df.boxplot(by=category)
plt.show()

from preprocessing.sampling import equal_sample

sample_df = final_df[final_df[category] == first_value]
counter_sample_df = final_df[final_df[category] == second_value]
equal_sample(sample_df, counter_sample_df)
class_df = equal_sample(sample_df, counter_sample_df)
class_df = class_df.sample(frac=1)

sample_df = sample_df.drop([category], axis=1)
counter_sample_df = counter_sample_df.drop([category], axis=1)

whole_sample = final_df.drop([category], axis=1)
pop_mean = whole_sample.mean()



from scipy import stats
F, p = stats.ttest_ind(sample_df, counter_sample_df)
print("T-Test ind two samples: F, p: ", F, p)

#F, p = stats. ttest_1samp(sample_df, counter_sample_df)
#print("Test sample versus population F, p: ", F, p)

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


lr_model = LogisticRegressionCV(solver='lbfgs', multi_class='ovr')
dt_clf = DecisionTreeClassifier(max_leaf_nodes=2)

array = class_df.to_numpy()
X = array[:, 0:(array.shape[1]-1)]
Y = array[:, array.shape[1]-1]


X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=0.2,random_state=2)

lr_model.fit(X_train, Y_train)
dt_clf.fit(X_train, Y_train)
predictions = lr_model.predict(X_validation)
print("Accuracy score: ", accuracy_score(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
print("coef:" , lr_model.coef_)

#tree.plot_tree(dt_clf)
#plt.show()