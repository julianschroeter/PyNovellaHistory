import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform, cdist
from scipy import stats
from sklearn.model_selection import train_test_split

from preprocessing.sampling import sample_n_from_cat
from preprocessing.util import from_n_dict_entries, first_n_dict_entries

name_cat = "Nachname"
periods_cat = "periods"
genre_cat = "Medientyp_ED" # nur für Gruppierung nach Medienformaten
genre_cat = "in_Deutscher_Novellenschatz" # "Kanon_Status" #
genre_cat = "Gattungslabel_ED_normalisiert"
class GroupDistances():
    """
    calculates the ingroup metrics and stores them in a distance_matrix and a list of metrics as attributes.
    """
    def __init__(self, input_df, metric, select_one_per_author, select_one_per_period, sample_size_df):
        frac_sample = None
        if sample_size_df is None:
            frac_sample = 1
        df = input_df.drop(columns=genre_cat)
        if select_one_per_period == True:
            df = sample_n_from_cat(df, cat_name="periods")
            if select_one_per_author == True:
                df = sample_n_from_cat(df)
            elif select_one_per_author == False:
                df = df.drop(columns = name_cat)
        elif select_one_per_period == False:
            df = df.drop(columns=periods_cat)
            if select_one_per_author == True:
                df = sample_n_from_cat(df)
            elif select_one_per_author == False:
                df = df.drop(columns= name_cat)

        df = df.sample(sample_size_df, frac=frac_sample)

        self.distances = pdist(df, metric=metric)
        v = squareform(self.distances)
        self.dist_matr_df = pd.DataFrame(v, index=df.index, columns=df.index)
        #self.dist_matr_df = self.dist_matr_df.replace({0: pd.NA})
        self.dist_matr_df["mean_dist"] = self.dist_matr_df.mean(axis=1)
        self.sample_length = len(df)


    def group_mean(self):
        return self.distances.mean()
    def group_std(self):
        return self.distances.std()


class InterGroupDistances(GroupDistances):
    """

    """
    def __init__(self, input_df_1, input_df_2, metric, select_one_per_author, select_one_per_period,
                 smaller_sample_size, sample_size_df_1, sample_size_df_2):

        frac_sample_1, frac_sample_2 = None, None
        if sample_size_df_1 is None:
            frac_sample_1 = 1
        if sample_size_df_2 is None:
            frac_sample_2 = 1

        if select_one_per_period == True:
            df_1 = sample_n_from_cat(input_df_1, cat_name=periods_cat)
            df_2 = sample_n_from_cat(input_df_2, cat_name=periods_cat)
            if select_one_per_author == True:
                both_dfs = pd.concat([df_1, df_2]).sample(frac=1.0)
                red_both_df = sample_n_from_cat(both_dfs)
                grouped = red_both_df.groupby(genre_cat)
                dfs_list = [grouped.get_group(df) for df in grouped.groups]
                df_1 = dfs_list[0].drop(columns=genre_cat)
                df_2 = dfs_list[1].drop(columns=genre_cat)
            elif select_one_per_author == False:
                df_1 = df_1.drop(columns=[genre_cat, name_cat])
                df_2 = df_2.drop(columns=[genre_cat, name_cat])
        elif select_one_per_period == False:
            df_1 = input_df_1.drop(columns=periods_cat)
            df_2 = input_df_2.drop(columns=periods_cat)
            if select_one_per_author == True:
                both_dfs = pd.concat([df_1, df_2]).sample(frac=1.0)
                red_both_df = sample_n_from_cat(both_dfs)
                grouped = red_both_df.groupby(genre_cat)
                dfs_list = [grouped.get_group(df) for df in grouped.groups]
                print(dfs_list)
                df_1 = dfs_list[0].drop(columns=genre_cat)
                df_2 = dfs_list[1].drop(columns=genre_cat)

            elif select_one_per_author == False:
                df_1 = df_1.drop(columns=[name_cat, genre_cat])
                df_2 = df_2.drop(columns=[name_cat, genre_cat])

        else:
            df_1 = input_df_1.drop(columns=[name_cat, genre_cat, periods_cat])
            df_2 = input_df_2(columns=[name_cat, genre_cat, periods_cat])

        df_1 = df_1.sample(sample_size_df_1, frac=frac_sample_1)
        df_2 = df_2.sample(sample_size_df_2, frac=frac_sample_2)

        if smaller_sample_size == True:
            len_sample_1 = len(df_1)
            len_sample_2 = len(df_2)
            final_sample_size = min(len_sample_1, len_sample_2)
            df_1 = df_1.sample(n=final_sample_size)
            df_2 = df_2.sample(n=final_sample_size)

        self.distances = cdist(df_1, df_2, metric=metric)
        self.dist_matr_df = pd.DataFrame(self.distances, index=df_1.index, columns=df_2.index)
        self.dist_matr_df = self.dist_matr_df.replace({0: pd.NA})
        self.dist_matr_df["mean_dist"] = self.dist_matr_df.mean(axis=1)
        self.dist_matr_df.loc["mean_dist"] = self.dist_matr_df.mean(axis=0)
        self.distances = np.array([e for sublist in self.distances.tolist() for e in sublist])
        self.sample1_length = len(df_1)
        self.sample2_length = len(df_2)



    #def group_mean(self):
     #   return self.metrics.mean()
    #def group_std(self):
     #   return self.metrics.std()


class IterateDistanceCalc:
    def __init__(self, n, input_df_1, input_df_2=None, metric="cosine",
                 select_one_author=False, select_one_per_period=False, smaller_sample_size=False,
                 sample_size_df_1=None, sample_size_df_2=None):


        means, stds = [], []
        distances = []
        list_sample_size_1, list_sample_size_2 = [], []
        mins_dict, max_dict = {}, {}
        for i in range(n):
            if input_df_2 is None:
                dist_obj = GroupDistances(input_df=input_df_1, metric=metric, select_one_per_author=select_one_author, select_one_per_period=select_one_per_period,
                                          sample_size_df=sample_size_df_1)
                list_sample_size_1.append(dist_obj.sample_length)
            elif input_df_2 is not None:
                dist_obj = InterGroupDistances(input_df_1=input_df_1, input_df_2=input_df_2, metric= metric,
                                               select_one_per_author=select_one_author, select_one_per_period=select_one_per_period,
                                               smaller_sample_size=smaller_sample_size, sample_size_df_1=sample_size_df_1, sample_size_df_2=sample_size_df_2)
                list_sample_size_1.append(dist_obj.sample1_length)
                list_sample_size_2.append(dist_obj.sample2_length)
            mean = dist_obj.group_mean()
            std = dist_obj.group_std()
            means.append(mean)
            stds.append(std)

            distances.extend(dist_obj.distances)

            if dist_obj.dist_matr_df["mean_dist"].idxmin() in mins_dict.keys():
                mins_dict[str(dist_obj.dist_matr_df["mean_dist"].idxmin())].append(dist_obj.dist_matr_df["mean_dist"].min())
            else:
                mins_dict[str(dist_obj.dist_matr_df["mean_dist"].idxmin())] = []
                mins_dict[str(dist_obj.dist_matr_df["mean_dist"].idxmin())].append(dist_obj.dist_matr_df["mean_dist"].min())

            if dist_obj.dist_matr_df["mean_dist"].idxmax() in max_dict.keys():
                max_dict[str(dist_obj.dist_matr_df["mean_dist"].idxmax())].append(dist_obj.dist_matr_df["mean_dist"].max())
            else:
                max_dict[str(dist_obj.dist_matr_df["mean_dist"].idxmax())] = []
                max_dict[str(dist_obj.dist_matr_df["mean_dist"].idxmax())].append(dist_obj.dist_matr_df["mean_dist"].max())

        new_min_dict, new_max_dict = {}, {}
        for key, values in mins_dict.items():
            params = [len(values) / n, np.array(values).mean()]
            new_min_dict[key] = params

        for key, values in max_dict.items():
            params = [len(values) / n, np.array(values).mean()]
            new_max_dict[key] = params

        min_dist = sorted(new_min_dict.items(), key=lambda x: x[1][0], reverse=True)
        max_dist = sorted(new_max_dict.items(), key= lambda x: x[1][0], reverse=True)

        self.min_dist = min_dist
        self.max_dist = max_dist
        self.mean = np.array(means).mean()
        self.stdev = np.array(stds).mean()
        self.distances = distances
        self.sample_size_1 = np.array(list_sample_size_1).mean()
        self.sample_size_2 = np.array(list_sample_size_2).mean()

def results_1group_dist(n, input_df, metric, select_one_author, select_one_per_period, sample_size_df=None):
    d = {}
    iter_obj = IterateDistanceCalc(n=n, input_df_1=input_df, metric=metric,
                                   select_one_author=select_one_author, select_one_per_period=select_one_per_period,
                                   sample_size_df_1=sample_size_df)
    d["mean"] = iter_obj.mean
    d["std"] = iter_obj.stdev
    d["sample1_size"] = iter_obj.sample_size_1
    d["sample2_size"] = iter_obj.sample_size_2
    d["min_dist"] = iter_obj.min_dist
    d["max_dist"] = iter_obj.max_dist
    return d


def results_2groups_dist(n, input_df_1, input_df_2,metric, select_one_author, select_one_per_period, smaller_sample_size=False,
                         sample_size_df_1=None, sample_size_df_2=None):
    d = {}
    iter_obj = IterateDistanceCalc(n=n, input_df_1=input_df_1,input_df_2=input_df_2, metric=metric,
                                   select_one_author=select_one_author, select_one_per_period=select_one_per_period,
                                   smaller_sample_size=smaller_sample_size, sample_size_df_1=sample_size_df_1, sample_size_df_2=sample_size_df_2)
    d["mean"] = iter_obj.mean
    d["std"] = iter_obj.stdev
    d["sample1_size"] = iter_obj.sample_size_1
    d["sample2_size"] = iter_obj.sample_size_2
    d["instance_of_df1_with_min_dist_to_df2_group"] = iter_obj.min_dist
    d["instance_of_df1_with_max_dist_to_df2_group"] = iter_obj.max_dist
    return d


class DistResults():
    """
    generates an object that has a results dataframe (with mean and std for the distance analyses for text groups) and a dictionary of the items with minimal and maximal distance
    including the parameters of proportion of being the minimal or maximal distance item over all iterations and the average distance to the group.
    """
    def __init__(self, n, input_df_1, input_df_2=None, metric="cosine", label1="", label2="", select_one_author=True, select_one_per_period=False,
                 smaller_sample_size=False, sample_size_df_1=None, sample_size_df_2=None):
        if input_df_2 is None:
            groups_str = ""
            row_name = groups_str + label1
            results_d = results_1group_dist(n, input_df_1, metric=metric, select_one_author=select_one_author, select_one_per_period=select_one_per_period,
                                            sample_size_df=sample_size_df_1)
        else:
            groups_str = "D(inter)_"
            row_name = groups_str + label1 + "_" + label2
            results_d = results_2groups_dist(n, input_df_2, metric=metric,
                                             select_one_author=select_one_author, select_one_per_period=select_one_per_period,
                                             smaller_sample_size=smaller_sample_size, sample_size_df_1=sample_size_df_1, sample_size_df_2=sample_size_df_2)

        red_dict = first_n_dict_entries(4, results_d)
        min_max_dict = from_n_dict_entries(4, results_d)
        self.min_max_results_dict = {}
        self.min_max_results_dict[row_name] = min_max_dict
        self.results_df = pd.DataFrame(red_dict, index=[row_name], columns=["mean", "std", "sample1_size", "sample2_size"])
        self.results_df.loc[row_name, "group_1"] = label1
        self.results_df.loc[row_name, "group_2"] = label2
        if label2 is None:
            self.results_df.loc[row_name, "inter_bool"] = False
        else:
            self.results_df.loc[row_name, "inter_bool"] = True

    def add_result(self, n, input_df_1, input_df_2=None, metric="cosine", label1=None, label2=None,
                   select_one_author=False, select_one_per_period=False, smaller_sample_size=False,
                   sample_size_df_1=None, sample_size_df_2=None):
        if input_df_2 is None:
            groups_str = ""
            row_name = groups_str + label1
            results_d = results_1group_dist(n, input_df_1, metric=metric,
                                            select_one_author=select_one_author, select_one_per_period=select_one_per_period,
                                            sample_size_df=sample_size_df_1)
        else:
            groups_str = "D(inter)_"
            row_name = groups_str + label1 + "_vs_" + label2
            results_d = results_2groups_dist(n, input_df_1, input_df_2, metric=metric,
                                             select_one_author=select_one_author, select_one_per_period=select_one_per_period,
                                             smaller_sample_size=smaller_sample_size, sample_size_df_1=sample_size_df_1, sample_size_df_2=sample_size_df_2)

        red_dict = first_n_dict_entries(4, results_d)

        min_max_dict = from_n_dict_entries(4, results_d)
        self.min_max_results_dict[row_name] = min_max_dict
        #self.results_df = pd.DataFrame(red_dict, index=[row_name], columns=["mean", "std", "sample1_size", "sample2_size"])
        self.results_df.loc[row_name, "mean"] = results_d["mean"]
        self.results_df.loc[row_name, "std"] = results_d["std"]
        self.results_df.loc[row_name, "sample1_size"] = results_d["sample1_size"]
        self.results_df.loc[row_name, "sample2_size"] = results_d["sample2_size"]
        self.results_df.loc[row_name, "group_1"] = label1
        self.results_df.loc[row_name, "group_2"] = label2
        if label2 is None:
            self.results_df.loc[row_name, "inter_bool"] = False
        else:
            self.results_df.loc[row_name, "inter_bool"] = True


    def calculate_differences_of_distances(self):
        inter_df = self.results_df[self.results_df["inter_bool"] == True]
        intra_df = self.results_df[self.results_df["inter_bool"] == False]
        inter_df.loc[:, "group_1_dist"] = inter_df["group_1"].apply(lambda x: intra_df.loc[x, "mean"])
        inter_df.loc[:, "group_2_dist"] = inter_df["group_2"].apply(lambda x: intra_df.loc[x, "mean"])
        inter_df.loc[:, "max_in_dist"] = inter_df[["group_1_dist", "group_2_dist"]].max(axis=1)

        inter_df.loc[:, "diff_of_dist"] = inter_df["mean"] - inter_df["max_in_dist"]

        self.results_df = pd.concat([intra_df, inter_df])
        self.results_df.loc["averages"] = self.results_df.mean(axis=0, numeric_only=True)

    def calculate_ratio_of_distances(self):
        inter_df = self.results_df[self.results_df["inter_bool"] == True]
        intra_df = self.results_df[self.results_df["inter_bool"] == False]
        inter_df.loc[:, "group_1_dist"] = inter_df["group_1"].apply(lambda x: intra_df.loc[x, "mean"])
        inter_df.loc[:, "group_2_dist"] = inter_df["group_2"].apply(lambda x: intra_df.loc[x, "mean"])
        inter_df.loc[:, "max_in_dist"] = inter_df[["group_1_dist", "group_2_dist"]].max(axis=1)

        inter_df.loc[:, "ratio_of_dist"] = inter_df["mean"] / inter_df["max_in_dist"]

        self.results_df = pd.concat([intra_df, inter_df])
        self.results_df.loc["averages"] = self.results_df.mean(axis=0, numeric_only=True)


def pairwise_self_counter_av_distances(input_df_1, input_df_2, metric, select_one_author, select_one_per_period, smaller_sample_size):
    """
    returns or each document in rows a DataFrame with the average metrics to all groups members for every text in the first column, and the average
    distance to all members of the comparison group (e.g. another genre) in the second column. The row indexes are the document ids
    """

    df_1 = GroupDistances(input_df_1, metric, select_one_author, select_one_per_period).dist_matr_df[["mean_dist"]]
    df_2 = InterGroupDistances(input_df_1, input_df_2, metric, select_one_author, select_one_per_period, smaller_sample_size).dist_matr_df[["mean_dist"]]
    df = pd.concat([df_1, df_2], axis=1)
    df.columns = ["in_group_mean", "out_group_mean"]
    df = df.drop("mean_dist")
    return df

def rel_ttest_in_out_groups(input_df_1, input_df_2, metric, select_one_author=True, select_one_per_period=False, smaller_sample_size=False, alternative= "less", compare_to_first=True):
    """
    directly returns F and p value for calculating relative ttest on pairwise calculating in- and out-group metrics for each document
    """

    pairs_df = pairwise_self_counter_av_distances(input_df_1, input_df_2, metric, select_one_author, select_one_per_period, smaller_sample_size, compare_to_first)
    F, p = stats.ttest_rel(pairs_df["in_group_mean"], pairs_df["out_group_mean"], alternative=alternative)
    return F, p


def iterate_rel_ttest(n, input_df_1, input_df_2, metric, select_one_author=True, alternative="less", compare_to_first=True):
    """

    """
    F_p_list = []
    for i in range(n):
        F_p_list.append(rel_ttest_in_out_groups(input_df_1, input_df_2, metric, select_one_author=select_one_author, alternative=alternative, compare_to_first=compare_to_first))
    return F_p_list


def random_iterate_rel_ttest(n, input_df_1, input_df_2, metric, select_one_author=True, random_sample_size=150, alternative="two-sided"):
    """

    """
    F_p_list = []
    for i in range(n):
        F_p_list.append(rel_ttest_in_out_groups(input_df_1.sample(random_sample_size), input_df_2.sample(random_sample_size), metric, select_one_author=select_one_author, alternative = alternative))
    return F_p_list


def iterate_inter_tests(n, input_df_1, input_df_2, metric, select_one_author=True, select_one_per_period=False,
                        test_function=stats.mannwhitneyu, alternative="less", smaller_sample_size=False,
                        sample_size_df_1=None, sample_size_df_2=None):
    p_list_to_group1, p_list_to_group2, F_list_to_group_1, F_list_to_group2 = [], [], [], []
    for i in range(n):

        if smaller_sample_size == True:
            len_sample_1 = len(sample_n_from_cat(input_df_1))
            len_sample_2 = len(sample_n_from_cat(input_df_2))
            final_sample_size = min(len_sample_1, len_sample_2)
            final_sample_frac=None
        else:
            final_sample_size = None
            final_sample_frac=1.0
        df_1_dists = GroupDistances(input_df=input_df_1.sample(n=final_sample_size, frac=final_sample_frac),
                                    metric=metric, select_one_per_author=select_one_author, select_one_per_period=select_one_per_period,
                                    sample_size_df=sample_size_df_1).distances
        df_2_dists = GroupDistances(input_df=input_df_2.sample(n=final_sample_size, frac=final_sample_frac),
                                    metric=metric, select_one_per_author=select_one_author, select_one_per_period=select_one_per_period,
                                    sample_size_df=sample_size_df_2).distances
        df_1_2_dists = InterGroupDistances(input_df_1.sample(n=final_sample_size, frac=final_sample_frac),
                                           input_df_2.sample(n=final_sample_size, frac=final_sample_frac),
                                           metric=metric, select_one_per_author=select_one_author, select_one_per_period=select_one_per_period,
                                           smaller_sample_size=smaller_sample_size,
                                           sample_size_df_1=sample_size_df_1, sample_size_df_2=sample_size_df_2).distances
        F,p = test_function(df_1_2_dists, df_1_dists, alternative=alternative)
        p_list_to_group1.append(p)
        F_list_to_group_1.append(F)
        F, p = test_function(df_1_2_dists, df_2_dists, alternative=alternative)
        p_list_to_group2.append(p)
        F_list_to_group2.append(F)
    return p_list_to_group1, F_list_to_group_1, p_list_to_group2, F_list_to_group2

def iterate_intra_tests(n, input_df_1, input_df_2, metric, select_one_author=True, select_one_per_period=False, test_function=stats.mannwhitneyu,
                        alternative="less", smaller_sample_size=False, sample_size_df_1=None, sample_size_df_2=None):
    p_list_to_group1, p_list_to_group2, F_list_to_group_1, F_list_to_group2 = [], [], [], []
    for i in range(n):

        if smaller_sample_size == True:
            len_sample_1 = len(sample_n_from_cat(input_df_1))
            len_sample_2 = len(sample_n_from_cat(input_df_2))
            final_sample_size = min(len_sample_1, len_sample_2)
            final_sample_frac = None
        else:
            final_sample_size = None
            final_sample_frac = 1.0
        df_1_dists = GroupDistances(input_df=input_df_1.sample(n=final_sample_size, frac=final_sample_frac),
                                    metric=metric, select_one_per_author=select_one_author, select_one_per_period=select_one_per_period,
                                    sample_size_df=sample_size_df_1).distances
        df_2_dists = GroupDistances(
                    input_df=input_df_2.sample(n=final_sample_size, frac=final_sample_frac),
                    select_one_per_author=select_one_author, select_one_per_period=select_one_per_period,
                    metric=metric, sample_size_df=sample_size_df_2).distances


        F, p = test_function(df_1_dists, df_2_dists, alternative=alternative)
        p_list_to_group1.append(p)
        F_list_to_group_1.append(F)
        F, p = test_function(df_1_dists, df_2_dists, alternative=alternative)
        p_list_to_group2.append(p)
        F_list_to_group2.append(F)
    return p_list_to_group1, F_list_to_group_1, p_list_to_group2, F_list_to_group2


def rd_iterate_intra_tests(n, input_df, metric, select_one_author=True, select_one_per_period=False,
                           test_function=stats.mannwhitneyu, alternative="less", sample_size_df=100):
    p_list_to_group1, p_list_to_group2, F_list_to_group_1, F_list_to_group2 = [], [], [], []
    for i in range(n):
        df_1, df_2 = train_test_split(input_df, train_size=0.5)

        df_1_dists = GroupDistances(input_df=df_1,
                                    metric=metric, select_one_per_author=select_one_author,
                                    select_one_per_period=select_one_per_period,
                                    sample_size_df=sample_size_df).distances
        df_2_dists = GroupDistances(
                    input_df=df_2,
                    select_one_per_author=select_one_author, select_one_per_period=select_one_per_period,
                    metric=metric, sample_size_df=sample_size_df).distances


        F, p = test_function(df_1_dists, df_2_dists, alternative=alternative)
        p_list_to_group1.append(p)
        F_list_to_group_1.append(F)
        F, p = test_function(df_1_dists, df_2_dists, alternative=alternative)
        p_list_to_group2.append(p)
        F_list_to_group2.append(F)
    return p_list_to_group1, F_list_to_group_1, p_list_to_group2, F_list_to_group2