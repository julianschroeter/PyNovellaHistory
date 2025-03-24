from copy import copy
import random
import pandas as pd



# meta date category names are hard coded here and should be changed for individual purposes
name_cat = "Nachname"
periods_cat = "Jahr_ED" # "date" # set do corpus "Jahr_ED"
genre_cat = "Gattungslabel_ED_normalisiert" #"in_Deutscher_Novellenschatz" # "Kanon_Status" #  "genre" #"Gattungslabel_ED_normalisiert"

def equal_sample(df_group1, df_group2, minor_frac=1.0):
    """generate a sample with two groups of equal size,
    determine the smaller sample, draw a set of n instances with
    n == size of the smaller sample and reduce both subsamples to minor_frac (<= 1.0, default = 1.0)"""
    if df_group1.shape[0] <= df_group2.shape[0]:
        df_group1 = df_group1.sample(frac=minor_frac)
        df_group2 = df_group2.sample(n=df_group1.shape[0])
        df_group2 = df_group2.sample(frac=minor_frac)
        sample = pd.concat([df_group1, df_group2])
    else:
        df_group2 = df_group2.sample(frac=minor_frac)
        df_group1 = df_group1.sample(n=df_group2.shape[0])
        df_group1 = df_group1.sample(frac=minor_frac)
        sample = pd.concat([df_group2, df_group1])
    # change order of rows, so that the groups are merged (otherwise accuracy scoring will always yield the same results)
    sample = sample.sample(frac=1)
    return sample


def sample_n_from_cat(df, cat_name="Nachname", n=1):
    df_copy = copy(df)
    df_copy = df_copy.groupby(cat_name).sample(n)
    df_copy = df_copy.drop(columns=[cat_name])
    return df_copy

def split_to2samples(input_df, metadata_category, label_list):
    df = input_df
    df_1 = df[df[metadata_category] == label_list[0]]
   # df_1 = df_1.drop([metadata_category], axis=1)
    df_2 = df[df[metadata_category] == label_list[1]]
   # df_2 = df_2.drop([metadata_category], axis=1)
    return df_1, df_2

def principled_sampling(input_df_1, input_df_2, select_one_per_period=True, select_one_per_author=True):
    """
    takes two data frames and returns a test, val split based on a principled sampling (selecting one text per author in each group, and one text
    per year
    returns: training sample, validation sample (which is the untrained proportion of the overall data
    """
    if select_one_per_period == True:
        df_1 = sample_n_from_cat(input_df_1, cat_name=periods_cat)
        df_2 = sample_n_from_cat(input_df_2, cat_name=periods_cat)
    else:
        df_1 = input_df_1.drop(columns=periods_cat)
        df_2 = input_df_2.drop(columns=periods_cat)

    if select_one_per_author == True:
        df_1 = sample_n_from_cat(input_df_1, cat_name=name_cat)
        df_2 = sample_n_from_cat(input_df_2, cat_name=name_cat)
    else:
        df_1 = df_1.drop(columns=[name_cat])
        df_2 = df_2.drop(columns=[name_cat])


    train_sample = equal_sample(df_1, df_2)
    train_sample = train_sample.drop(columns=[periods_cat])
    train_sample = train_sample.sample(frac=0.8)

    all_df = pd.concat([input_df_1, input_df_2])
    all_df = all_df.drop(columns=[name_cat, periods_cat])
    test_sample = all_df.drop(index=train_sample.index.values)

    grouped = test_sample.groupby(genre_cat)
    dfs_list = [grouped.get_group(df) for df in grouped.groups]
    test_df_1 = dfs_list[0]
    test_df_2 = dfs_list[1]


    test_sample = equal_sample(test_df_1, test_df_2)

    return train_sample, test_sample

def select_consecutive_rows(df, n):
    """
    returns a new data frame with n consecutive and randomly chosen rows from df
    """
    if df.shape[0] > n:
        k = random.randint(0, df.shape[0]-n)
        return df[k:k+n]
    else:
        return df


def consecutive_rows_to_one(df,n):
    new_df = select_consecutive_rows(df,n)
    new_df = new_df.groupby(["doc_id"])["sent_string"].apply(" ".join).reset_index()
    return new_df

def select_from_corpus_df(df,n, list_of_ids=None, id_cat="doc_id", join_sent=True):
    """
    returns a new data frame with n consecutive and randomly chosen rows for all rows that match the id from list_of_ids
    """
    list_of_dfs = []
    if not list_of_ids:
        list_of_ids = pd.unique(df.doc_id).tolist()

    for id in list_of_ids:
        text_df = df[df[id_cat] == id]
        if join_sent == True:
            selected_df = consecutive_rows_to_one(text_df, n)
        else:
            selected_df = select_consecutive_rows(df,n)
        list_of_dfs.append(selected_df)
    new_df = pd.concat(list_of_dfs)
    return new_df

