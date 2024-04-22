import statistics

from preprocessing.presetting import (language_model_path, local_temp_directory,
                                      global_corpus_representation_directory, load_stoplist, set_DistReading_directory, mallet_directory)
from preprocessing.corpus import DocFeatureMatrix
from preprocessing.sampling import equal_sample
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collections import Counter
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches

system_name = "my_xps" #  "wcph113" # "my_mac" # "wcph104" "my_xps" #

data_matrix_filepath = os.path.join(global_corpus_representation_directory(system_name), "DocThemesMatrix.csv")
language_model = language_model_path(system_name)
metadata_csv_filepath = os.path.join(global_corpus_representation_directory(system_name=system_name), "Bibliographie.csv")


textanalytic_metadata_filepath = os.path.join(global_corpus_representation_directory(system_name), "textanalytic_metadata.csv")

matrix = DocFeatureMatrix(data_matrix_filepath= data_matrix_filepath, data_matrix_df=None, metadata_df=None,
                                  metadata_csv_filepath = textanalytic_metadata_filepath, mallet=False)

matrix = matrix.reduce_to(["Marseille"])


matrix = matrix.add_metadata("region")

matrix.data_matrix_df.replace({"region": {"Italien": "rom",
                                                        "Spanien": "rom",
                                                        "Frankreich": "rom", "Lateinamerika" : "lat_am",
                                          "Karibik" : "lat_am", "Chile" : "lat_am", "Portugal":"rom",
                                          "Deutschland" : "non_rom", "Österreich" : "non_rom",
                                          "Niederlande" : "non_rom", "Ungarn" : "non_rom", "Russlan" : "non_rom",
                                          "Polen" : "non_rom", "Schweden" : "non_rom", "Universum" : "non_rom", "Dänemark" :"non_rom",
                                          "Russland" : "non_rom", "Schweiz" : "non_rom", "Karibik" : "lat_am", "Alpen" : "non_rom",
                                          "Nordamerika" :"non_rom", "Meer":"non_rom", "unbestimmt": "other"

                                          }}, inplace=True)

matrix.data_matrix_df["region"] = matrix.data_matrix_df["region"].fillna("other", inplace=False)

df = matrix.data_matrix_df

df = df.rename(columns={"Marseille":"SettingShare"})
df_Setting_Share = df.copy()

df_whole_corpus = df
df = df.query("region != '''other'''")
df = df.query("region != '''lat_am'''")

scaler = StandardScaler()
scaler = MinMaxScaler()
df.iloc[:, :1] = scaler.fit_transform(df.iloc[:, :1].to_numpy())

matrix = matrix.eliminate(["stadt_land", "titel", "ende", "liebesspannung"])
final_df = df

category = "region"# "liebesspannung" # "stadt_land" #
first_value ="rom" # "ja" #"land" #
second_value = "non_rom" #"nein" # "None" #≈


sample_df = final_df[final_df[category] == first_value]
counter_sample_df = final_df[final_df[category] == second_value]

test_sample_df = sample_df.drop([category], axis=1)
test_counter_sample_df = counter_sample_df.drop([category], axis=1)

whole_sample = final_df.drop([category], axis=1)
pop_mean = whole_sample.mean()
print("Sample orders for method 1: ")
print(sample_df.sort_values(by="SettingShare", ascending=False))
print(counter_sample_df.sort_values(by="SettingShare", ascending=False))

from scipy import stats
F, p = stats.ttest_ind(test_sample_df, test_counter_sample_df)
print("F, p: ", F, p)

F, p = stats. ttest_1samp(test_sample_df, pop_mean)
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

list_of_boundarys = []
n = 1
for i in range(n):
    class_df = equal_sample(sample_df, counter_sample_df)
    class_df = class_df.sample(frac=1)
    array = class_df.to_numpy()
    X = array[:, 0:(array.shape[1]-1)]
    Y = array[:, array.shape[1]-1]
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=0.2,random_state=2)

    lr_model.fit(X_train, Y_train)

    predictions = lr_model.predict(X_validation)
    print("Accuracy score: ", accuracy_score(Y_validation, predictions))
    print("cv score: ", lr_model.score(X, Y))
    print(classification_report(Y_validation, predictions))
    print("coef:" , lr_model.coef_)
    boundary = 0.5 / lr_model.coef_
    print("Schwellenwert: " , boundary )
    list_of_boundarys.append(boundary)

    lengths = np.arange(-300, 300)
    lengths = np.asarray([x / 100 for x in lengths])
    predictions = lr_model.predict(lengths.reshape(-1, 1))
    probabs = lr_model.predict_proba(lengths.reshape(-1, 1))
    probabs = [x[1] for x in probabs]
    pred_colors = ["red" if x == "rom" else "green" for x in predictions]

    prob_df = pd.DataFrame(data={"lengths": lengths, "probabs": probabs,
                                 "predictions": predictions})
    x_boundary = prob_df[prob_df["probabs"] >= 0.5].iloc[0,0]

    plt.scatter(lengths, probabs, c=pred_colors)
    plt.title("Entscheidungsgrenze für die Klassifikation (Val)")
    plt.ylabel("Vorhersagewahrscheinlichkeit")
    plt.xlabel("Romanisches Setting (NamedEntShare)")
    plt.vlines(x_boundary,0, 0.5, label="Decision boundary")
    plt.hlines(0.5, -3, x_boundary, label= "Decision Boundary: ")
    plt.text(x_boundary,0, "Entscheidungsgrenze: "+ str(x_boundary), ha='left', va='center')
    plt.legend
    plt.show()

final_df.boxplot(by=category)
plt.axhline(y = x_boundary)
plt.show()


# repeat the same validation procedure for the NamedEntShare operationalization
data_matrix_filepath = os.path.join(global_corpus_representation_directory(system_name), "toponym_share_Matrix.csv")
language_model = language_model_path(system_name)
metadata_csv_filepath = os.path.join(global_corpus_representation_directory(system_name=system_name), "Bibliographie.csv")
print(data_matrix_filepath)
print(metadata_csv_filepath)


textanalytic_metadata_filepath = os.path.join(global_corpus_representation_directory(system_name), "textanalytic_metadata.csv")

matrix = DocFeatureMatrix(data_matrix_filepath= data_matrix_filepath, data_matrix_df=None, metadata_df=None,
                                  metadata_csv_filepath = textanalytic_metadata_filepath, mallet=False)


print(matrix.data_matrix_df)

matrix = matrix.reduce_to(["rom_top"])
print(matrix.data_matrix_df)


matrix = matrix.add_metadata("region")

matrix.data_matrix_df.replace({"region": {"Italien": "rom",
                                                        "Spanien": "rom",
                                                        "Frankreich": "rom", "Lateinamerika" : "lat_am",
                                          "Karibik" : "lat_am", "Chile" : "lat_am", "Portugal":"rom",
                                          "Deutschland" : "non_rom", "Österreich" : "non_rom",
                                          "Niederlande" : "non_rom", "Ungarn" : "non_rom", "Russlan" : "non_rom",
                                          "Polen" : "non_rom", "Schweden" : "non_rom", "Universum" : "non_rom", "Dänemark" :"non_rom",
                                          "Russland" : "non_rom", "Schweiz" : "non_rom", "Karibik" : "lat_am", "Alpen" : "non_rom",
                                          "Nordamerika" :"non_rom", "Meer":"non_rom", "unbestimmt" : "other"

                                          }}, inplace=True)

matrix.data_matrix_df["region"] = matrix.data_matrix_df["region"].fillna("other", inplace=False)

df = matrix.data_matrix_df

df = df.query("region != '''other'''")
df = df.query("region != '''lat_am'''")

df = df.rename(columns={"rom_top":"NamedEntShare"})

scaler = StandardScaler()
df.iloc[:, :1] = scaler.fit_transform(df.iloc[:, :1].to_numpy())

matrix = matrix.eliminate(["stadt_land", "titel", "ende", "liebesspannung"])
final_df = df


print(final_df.describe())

category = "region"# "liebesspannung" # "stadt_land" #
first_value ="rom" # "ja" #"land" #
second_value = "non_rom" #"nein" # "None" #≈



sample_df = final_df[final_df[category] == first_value]
counter_sample_df = final_df[final_df[category] == second_value]

print(sample_df.sort_values(by="NamedEntShare", ascending=False))
print(counter_sample_df.sort_values(by="NamedEntShare", ascending=False))

test_sample_df = sample_df.drop([category], axis=1)
test_counter_sample_df = counter_sample_df.drop([category], axis=1)

whole_sample = final_df.drop([category], axis=1)
pop_mean = whole_sample.mean()
print(sample_df)



F, p = stats.ttest_ind(test_sample_df, test_counter_sample_df)
print("F, p: ", F, p)

F, p = stats. ttest_1samp(test_sample_df, pop_mean)
print("F, p: ", F, p)

grouped_df = final_df.groupby([category]).mean()

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

list_of_boundarys = []
for i in range(n):

    class_df = equal_sample(sample_df, counter_sample_df)
    class_df = class_df.sample(frac=1)
    print(class_df)


    # classification

    lr_model = LogisticRegressionCV()

    array = class_df.to_numpy()
    X = array[:, 0:(array.shape[1]-1)]
    Y = array[:, array.shape[1]-1]


    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=0.2,random_state=2)

    lr_model.fit(X_train, Y_train)

    predictions = lr_model.predict(X_validation)
    print("Accuracy score on separate validation set: ", accuracy_score(Y_validation, predictions))
    print("cv score: ", lr_model.score(X, Y))
    print(classification_report(Y_validation, predictions))
    print("coef:" , lr_model.coef_)
    boundary = 0.5 / lr_model.coef_
    print("Schwellenwert: " , boundary)
    list_of_boundarys.append(boundary)

    lengths = np.arange(-300, 300)
    lengths = np.asarray([x / 100 for x in lengths])
    predictions = lr_model.predict(lengths.reshape(-1, 1))
    probabs = lr_model.predict_proba(lengths.reshape(-1, 1))
    probabs = [x[1] for x in probabs]
    pred_colors = ["magenta" if x == "rom" else "grey" for x in predictions]

    prob_df = pd.DataFrame(data={"lengths": lengths, "probabs": probabs,
                                 "predictions": predictions})
    x_boundary = prob_df[prob_df["probabs"] >= 0.5].iloc[0, 0]

    plt.scatter(lengths, probabs, c=pred_colors)
    plt.title("Entscheidungsgrenze für die Klassifikation: \n Romanisch/Nicht-Rom.")
    plt.ylabel("Vorhersagewahrscheinlichkeit")
    plt.xlabel("Romanisches Setting (NamedEntShare)")
    plt.vlines(x_boundary, 0, 0.5, label="Decision boundary", color="grey")
    plt.hlines(0.5, -3, x_boundary, label="Decision Boundary: ", color="grey")
    plt.text(x_boundary, 0, "Entscheidungsgrenze: " + str(x_boundary).replace(".",","), ha='left', va='center')

    predictions = ["Romanisch" if "rom" else "Nicht-Rom." for element in predictions]
    mpatches_list = []
    zipped_dict = {"Romanisch": "magenta", "Nicht-Rom.": "grey"}
    for key, value in zipped_dict.items():
        patch = mpatches.Patch(color=value, label=key)
        mpatches_list.append(patch)

    plt.legend(handles=mpatches_list)
    plt.savefig(os.path.join(local_temp_directory(system_name), "figures",
                             "sigmoid_Entscheidungsgrenze_RomSetting_NEShare.svg"))
    plt.show()

df = final_df.rename(columns={"region":"Romanisches Setting"})
#df.iloc[:, :1] = scaler.fit_transform(df.iloc[:, :1].to_numpy())
category ="Romanisches Setting"
rom_data = df[df["Romanisches Setting"] == "rom"]["NamedEntShare"].values.tolist()
non_rom_data = df[df["Romanisches Setting"] == "non_rom"]["NamedEntShare"].values.tolist()
print(non_rom_data)

fig, axes = plt.subplots(1,2)
axes[0].boxplot([rom_data, non_rom_data],  medianprops=dict(color="grey"))
#plt.xlim([0,3])
axes[0].set_xticks([1,2], ["romanisch", "nicht-romanisch"])
axes[0].axhline(y= x_boundary, color="grey")
#axes[0].set_xlabel("Textgruppen nach Annotation")
axes[0].set_ylabel("NamedEntShare")
axes[0].set_title("NamedEntShare")
#plt.tight_layout()
#fig.savefig(os.path.join(local_temp_directory(system_name), "figures", "Boxplot_Entscheidungsgrenze_RomSetting_NEShare.svg") )
#plt.show()

df = df_whole_corpus.drop(columns=["region"])

matrix_obj = DocFeatureMatrix(data_matrix_filepath=None, data_matrix_df=df, metadata_csv_filepath=metadata_csv_filepath)
matrix_obj = matrix_obj.add_metadata(("Gattungslabel_ED_normalisiert"))
matrix_N = matrix_obj.reduce_to_categories("Gattungslabel_ED_normalisiert", ["N"])
matrix_E = matrix_obj.reduce_to_categories("Gattungslabel_ED_normalisiert", ["E"])
matrix_mid_length = matrix_obj.reduce_to_categories("Gattungslabel_ED_normalisiert", ["0E", "XE"])

whole_array = df_whole_corpus.to_numpy()
X = whole_array[:, 0:(whole_array.shape[1]-1)]
X = scaler.fit_transform(X)

print("predictions on whole corpus: ")


counts = Counter(lr_model.predict(X))
print(counts)


print("within Novellen: ")
df = matrix_N.data_matrix_df
whole_array = df.to_numpy()
X = whole_array[:, 0:(whole_array.shape[1]-1)]
X = scaler.fit_transform(X)
counts = Counter(lr_model.predict(X))
print(counts)


print("within Erzählungen: ")
df = matrix_E.data_matrix_df
whole_array = df.to_numpy()
X = whole_array[:, 0:(whole_array.shape[1]-1)]
X = scaler.fit_transform(X)
counts = Counter(lr_model.predict(X))
print(counts)

print("within other mid-length prose fiction: ")
df = matrix_mid_length.data_matrix_df
whole_array = df.to_numpy()
X = whole_array[:, 0:(whole_array.shape[1]-1)]
X = scaler.fit_transform(X)
counts = Counter(lr_model.predict(X))
print(counts)

print(final_df)

df = df_Setting_Share.copy()
df.iloc[:, :1] = scaler.fit_transform(df.iloc[:, :1].to_numpy())
df = df.rename(columns={"region":"Romanisches Setting"})
rom_data = df[df["Romanisches Setting"] == "rom"]["SettingShare"].values.tolist()
non_rom_data = df[df["Romanisches Setting"] == "non_rom"]["SettingShare"].values.tolist()
print("rom_data: ", rom_data)
print(non_rom_data)

boxplot_colors = dict(boxes='dimgray', whiskers='gray', medians='gray', caps='gray')
c = "grey"
#fig, ax = plt.subplots()
axes[1].boxplot([rom_data, non_rom_data],
            medianprops=dict(color=c))
#plt.xlim([0,3])
axes[1].set_xticks([1,2], ["romanisch", "nicht-romanisch"])
#ax.axhline(y= x_boundary)
fig.supxlabel("Textgruppen nach Annotation")
axes[1].set_ylabel("SettingShare(Romanisch)")
axes[1].set_title("SettingShare")
fig.suptitle("Boxplots mit Entscheidungsgrenze")
plt.tight_layout()
fig.savefig(os.path.join(local_temp_directory(system_name), "figures", "Boxplot_Entscheidungsgrenze_RomSetting_NE-and-SettingShare.svg") )
plt.show()