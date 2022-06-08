from copy import copy

import pandas as pd

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
    df_1 = df_1.drop([metadata_category], axis=1)
    df_2 = df[df[metadata_category] == label_list[1]]
    df_2 = df_2.drop([metadata_category], axis=1)
    return df_1, df_2