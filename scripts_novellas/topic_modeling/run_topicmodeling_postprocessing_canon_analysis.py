from topicmodeling.Postprocessing import DocTopicMatrix
from preprocessing.presetting import language_model_path, vocab_lists_dicts_directory, local_temp_directory, global_corpus_representation_directory, load_stoplist, set_DistReading_directory, mallet_directory
from preprocessing.corpus import DocFeatureMatrix
from preprocessing.metadata_transformation import full_genre_labels
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
system = "my_xps" # "wcph113" # "my_mac" # "wcph104"
data_matrix_filepath = os.path.join(local_temp_directory(system), "output_composition_100topics1.txt")
language_model = language_model_path(system)
metadata_csv_filepath = os.path.join(global_corpus_representation_directory(system_name=system), "Bibliographie.csv")

Metadata_df = pd.read_csv(os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv"), index_col=0)
print(Metadata_df)

topic_keys_df = pd.read_csv(os.path.join(local_temp_directory(system), "output_keys_100topics1.txt"), sep="\t",
                            names=["Topic_ID", "p", "Topic_Keys"])
print(topic_keys_df)


colors_list = load_stoplist(os.path.join(vocab_lists_dicts_directory(system), "my_colors.txt"))
textanalytic_metadata_filepath = os.path.join(global_corpus_representation_directory(system), "textanalytic_metadata.csv")
topic_doc_matrix = DocTopicMatrix(data_matrix_filepath= data_matrix_filepath, data_matrix_df=None, metadata_df=None,
                                  metadata_csv_filepath = textanalytic_metadata_filepath, mallet=True)

topic_doc_matrix = topic_doc_matrix.adjust_doc_chunk_multiindex()
topic_doc_matrix.data_matrix_df.to_csv(os.path.join(global_corpus_representation_directory(system), "test-doc-topics.csv"))
df_all_segments = topic_doc_matrix.data_matrix_df
matrix = topic_doc_matrix.mean_doclevel()
doc_topic_df = matrix.data_matrix_df
doc_topic_df = doc_topic_df.head(-7)


year_cat_name = "Jahr_ED"

matrix = DocFeatureMatrix(data_matrix_filepath=None, data_matrix_df = doc_topic_df,
                         metadata_df=None,metadata_csv_filepath=metadata_csv_filepath,
                        mallet=False)

i = 0
list_of_topics = [5,13,22,36,45, 57, 58, 59,65,66, 67, 72,73,82,84,90,94,95]
list_of_topics = [58,59,65,66]

fig, axes = plt.subplots(nrows=len(list_of_topics), ncols=2, figsize=(16, 23))
for topic_id in list_of_topics:
    matrix_red = matrix.reduce_to([topic_id]) # = bestimmtes topic
    matrix_red = matrix_red.add_metadata([year_cat_name, "Kanon_Status", "Titel", "Nachname", "Gender"])


    df = matrix_red.data_matrix_df

    scaler = StandardScaler()
    scaler = MinMaxScaler()
    df.iloc[:, :1] = scaler.fit_transform(df.iloc[:, :1].to_numpy())

    df.rename(columns={"Kanon_Status":"Canonicity"}, inplace=True)

    category = "Canonicity" # "stadt_land" # "region"


    replace_dict = {category: {0: "low", 1: "low", 2: "high",
                                                      3:"high"}}
    df  = full_genre_labels(df, replace_dict=replace_dict)

    replace_dict = {"Gender": {"f": "female", "unbekannt":"unknown", "m": "male"}}
    df = full_genre_labels(df, replace_dict=replace_dict)

    y_variable = topic_id
    sns.lineplot(data=df, x=year_cat_name, y=y_variable, hue=category,
                 palette={"low":"grey", "high": "purple"}, ax=axes[i,0])

    axes[i,0].set_ylabel("Topic: " + str(y_variable))


    sns.lineplot(data=df, x=year_cat_name, y=y_variable, hue="Gender",
                 palette={"female": "red", "male": "blue", "unknown": "grey"}, ax=axes[i,1])


    axes[i, 1].set_xlabel("")
    axes[i,0].set_xlabel("Keys: " + str(topic_keys_df.loc[topic_id, "Topic_Keys"]))
    axes[i,0].xaxis.set_label_coords(0.99, -0.05)
    axes[i,1].set_ylabel("")
    i += 1
fig.suptitle("Distribution of Topics relative to Gender and Canonicity\n")
fig.supxlabel("\n Year of First Publication")
fig.tight_layout()
fig.savefig(os.path.join(local_temp_directory(system), "figures",
                         str(y_variable) + "_set_of_lovetopics_alt" + "canon_lineplot_gender_timeline.svg"))
fig.show()


