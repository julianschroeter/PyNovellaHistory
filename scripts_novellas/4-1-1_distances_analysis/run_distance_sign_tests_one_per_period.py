system = "wcph113"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')

from metrics.distances import IterateDistanceCalc, iterate_inter_tests, iterate_intra_tests
from preprocessing.corpus import DTM
from preprocessing.presetting import global_corpus_raw_dtm_directory, global_corpus_representation_directory
from preprocessing.metadata_transformation import years_to_periods
from preprocessing.sampling import sample_n_from_cat
import os
from scipy import stats
import numpy as np
from sklearn.model_selection import train_test_split

filename = "scaled_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv"
filename = "scaled_no-stopwords_no-names_raw_dtm_lemmatized_l1__use_idf_False6000mfw.csv"
metadata_path = os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")
filepath = os.path.join(global_corpus_raw_dtm_directory(system), filename)

label_list = ["R", "M", "E", "N", "0E", "XE"]
genre_cat = "Gattungslabel_ED_normalisiert"
year_cat = "Jahr_ED"

metric = "cosine"

dtm_obj = DTM(data_matrix_filepath=filepath, metadata_csv_filepath=metadata_path)
dtm_obj = dtm_obj.add_metadata([genre_cat, year_cat, "Nachname"])
dtm_obj = dtm_obj.eliminate(["novelle","erzählung", "roman", "märchen", "fle", "be", "te", "ge", "de", "di"])

dtm_obj.data_matrix_df = years_to_periods(dtm_obj.data_matrix_df,category_name=year_cat, start_year=1790, end_year=1950, epoch_length=1,
                                          new_periods_column_name="periods")
dtm_obj.eliminate([year_cat])

label_list = ["N", "E", "M", "R"]
genre_dtm_obj = dtm_obj.reduce_to_categories(metadata_category=genre_cat, label_list=[label_list[0]])
#genre_dtm_obj.data_matrix_df = sample_n_from_cat(genre_dtm_obj.data_matrix_df, cat_name="periods", n=1)
#genre_dtm_obj = genre_dtm_obj.eliminate(["periods"])
genre_dtm_obj = genre_dtm_obj.eliminate([genre_cat, year_cat])
df_N = genre_dtm_obj.data_matrix_df

print("The following test results are based on sampling one text per year and then one text per author in each group:")


print("size of whole novellen sample: ", len(df_N))
print("Sample size Novellen (sample one instance per year and per author): ", len(sample_n_from_cat(sample_n_from_cat(df_N, cat_name= "periods"))))

genre_dtm_obj = dtm_obj.reduce_to_categories(metadata_category=genre_cat, label_list=[label_list[1]])
#genre_dtm_obj.data_matrix_df = sample_n_from_cat(genre_dtm_obj.data_matrix_df, cat_name="periods", n=1)
#genre_dtm_obj = genre_dtm_obj.eliminate(["periods"])
genre_dtm_obj = genre_dtm_obj.eliminate([genre_cat, year_cat])
df_E = genre_dtm_obj.data_matrix_df
print("Sample size Erzählungen (sample one instance per year and per author): ", len(sample_n_from_cat(sample_n_from_cat(df_E, cat_name="periods"))))


genre_dtm_obj = dtm_obj.reduce_to_categories(metadata_category=genre_cat, label_list=[label_list[2]])
#genre_dtm_obj.data_matrix_df = sample_n_from_cat(genre_dtm_obj.data_matrix_df, cat_name="periods", n=1)
genre_dtm_obj = genre_dtm_obj.eliminate([genre_cat, year_cat])
df_M = genre_dtm_obj.data_matrix_df
print("Sample size Märchen (sample one instance per year and per author): ", len(sample_n_from_cat(sample_n_from_cat(df_M, cat_name="periods"))))

genre_dtm_obj = dtm_obj.reduce_to_categories(metadata_category=genre_cat, label_list=[label_list[3]])
#genre_dtm_obj.data_matrix_df = sample_n_from_cat(genre_dtm_obj.data_matrix_df, cat_name="periods", n=1)
genre_dtm_obj = genre_dtm_obj.eliminate([genre_cat, year_cat])
df_R = genre_dtm_obj.data_matrix_df
print("Sample size Romane (sample one instance per year and per author): ", len(sample_n_from_cat(sample_n_from_cat(df_R, cat_name="periods"))))



rd_dtm_obj = dtm_obj.eliminate([year_cat, genre_cat])
df = rd_dtm_obj.data_matrix_df
rd_sample1, rd_sample2 = train_test_split(df, train_size=0.5)

print("len rd_sample1", len((sample_n_from_cat(rd_sample1))))
print("len rd_sample2", len((sample_n_from_cat(rd_sample2))))

mid_length_prose_rd_sample_obj = dtm_obj.eliminate([genre_cat, year_cat])
df_mid_length_rd = mid_length_prose_rd_sample_obj.data_matrix_df.sample(220)



n = 10

print("Number of iterations = ", str(n))

p_to_df1, F_to_df1, p_to_df2, F_to_df2 = iterate_inter_tests(n, df_N, df_E, metric=metric, alternative="greater", test_function=stats.mannwhitneyu, select_one_per_period=True)
print("average over n samples: N_E vs N sign. MannWhitneyU Test, test-statistic, p-value: ")
print(np.array(F_to_df1).mean(), np.array(p_to_df1).mean())
print(" average over n samples: N_E vs E sign. MannWhitneyU Test, test-statistic, p-value: ")
print(np.array(F_to_df2).mean(), np.array(p_to_df2).mean())

p_to_df1, F_to_df1, p_to_df2, F_to_df2 = iterate_inter_tests(n, df_R, df_M, metric=metric, alternative="greater", test_function=stats.mannwhitneyu, select_one_per_period=True)
print("average over n samples: R_M vs R sign. MannWhitneyU Test, test-statistic, p-value: ")
print(np.array(F_to_df1).mean(), np.array(p_to_df1).mean())
print(" average over n samples: R_M vs M sign. MannWhitneyU Test, test-statistic, p-value: ")
print(np.array(F_to_df2).mean(), np.array(p_to_df2).mean())

p_to_df1, F_to_df1, p_to_df2, F_to_df2 = iterate_inter_tests(n, rd_sample1, rd_sample2, metric=metric, alternative="greater", test_function=stats.mannwhitneyu, select_one_author=True, select_one_per_period=True)
print("average over n samples: rd vs rd1 sign. MannWhitneyU Test, test-statistic, p-value: ")
print(np.array(F_to_df1).mean(), np.array(p_to_df1).mean())
print(" average over n samples: rd vs rd2 sign. MannWhitneyU Test, test-statistic, p-value: ")
print(np.array(F_to_df2).mean(), np.array(p_to_df2).mean())

p_to_df1, F_to_df1, p_to_df2, F_to_df2 = iterate_intra_tests(n, df_N, df_mid_length_rd, metric=metric, alternative="less", test_function=stats.mannwhitneyu, select_one_per_period=True)
print("average over n samples: N vs. rd sample sign. MannWhitneyU Test, test-statistic, p-value: ")
print(np.array(F_to_df1).mean(), np.array(p_to_df1).mean())


p_to_df1, F_to_df1, p_to_df2, F_to_df2 = iterate_intra_tests(n, df_E, df_mid_length_rd, metric=metric, alternative="less", test_function=stats.mannwhitneyu, select_one_per_period=True)
print("average over n samples: E vs. rd sample sign. MannWhitneyU Test, test-statistic, p-value: ")
print(np.array(F_to_df1).mean(), np.array(p_to_df1).mean())

p_to_df1, F_to_df1, p_to_df2, F_to_df2 = iterate_intra_tests(n, df_M, rd_sample1.sample(100), metric=metric, alternative="less", test_function=stats.mannwhitneyu, select_one_per_period=True)
print("average over n samples: M vs. rd sample sign. MannWhitneyU Test, test-statistic, p-value: ")
print(np.array(F_to_df1).mean(), np.array(p_to_df1).mean())

p_to_df1, F_to_df1, p_to_df2, F_to_df2 = iterate_intra_tests(n, df_R, rd_sample1.sample(100), metric=metric, alternative="less", test_function=stats.mannwhitneyu, select_one_per_period=True)
print("average over n samples: R vs. rd sample sign. MannWhitneyU Test, test-statistic, p-value: ")
print(np.array(F_to_df1).mean(), np.array(p_to_df1).mean())