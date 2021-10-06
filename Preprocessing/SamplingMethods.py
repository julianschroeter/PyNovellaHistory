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


