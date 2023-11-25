

#the same for periods:
new_dtm_obj = copy(dtm_obj)
label_list = ["N", "E", "0E", "XE", "M", "R"]
new_dtm_obj = new_dtm_obj.reduce_to_categories(genre_cat, label_list)
new_dtm_obj = new_dtm_obj.eliminate([genre_cat])
new_dtm_obj = new_dtm_obj.add_metadata(["Jahr_ED"])

new_dtm_obj.data_matrix_df = years_to_periods(input_df=new_dtm_obj.data_matrix_df, category_name="Jahr_ED",
                                          start_year=1750, end_year=1950, epoch_length=100,
                                          new_periods_column_name="periods")
new_dtm_obj = new_dtm_obj.eliminate(["Jahr_ED"])

periods_list= ["1790-1820", "1820-1850"]
periods_list = ["1850-1880", "1880-1910"]
periods_list = ["1750-1850", "1850-1950"]

before_df, after_df = split_to2samples(new_dtm_obj.data_matrix_df, metadata_category="periods", label_list=periods_list)

list_of_genre_dfs.append([str(periods_list[0])+ "_" + str(label_list[0]), before_df])
list_of_genre_dfs.append([str(periods_list[1])+"_"+ str(label_list[0]), after_df])


# again df has to be set up as DTM object because genre label has to be added as metadata from metadata table:
label_list = ["N", "E"]

before_obj = DTM(data_matrix_df=before_df, metadata_csv_filepath=metadata_path)
before_obj = before_obj.add_metadata((genre_cat))
df_N_before, df_E_before = split_to2samples(before_obj.data_matrix_df, genre_cat, label_list)
list_of_genre_dfs.append([str(periods_list[0]) +"_"+ str(label_list[0]), df_N_before])
list_of_genre_dfs.append([str(periods_list[0])+"_"+ str(label_list[1]), df_E_before])

label_list = ["M", "R"]
df_M_b, df_R_b = split_to2samples(before_obj.data_matrix_df, genre_cat, label_list)
#list_of_genre_dfs.append([[str(periods_list[0]), str(label_list[0])], df_M_b])
list_of_genre_dfs.append([str(periods_list[0])+"_"+str(label_list[1]), df_R_b])
#print(df_M_b)
print(df_R_b)


# generate a new dtm object "after 1850" for generating new subsamples for N and E:
label_list = ["N", "E"]

a_obj = DTM(data_matrix_df=after_df, metadata_csv_filepath=metadata_path)
a_obj = a_obj.add_metadata((genre_cat))
df_N_a, df_E_a = split_to2samples(a_obj.data_matrix_df, genre_cat, label_list)
list_of_genre_dfs.append([str(periods_list[1])+"_"+str(label_list[0]), df_N_a])
list_of_genre_dfs.append([str(periods_list[1])+"_"+str(label_list[1]), df_E_a])

label_list = ["M", "R"]
df_M_a, df_R_a = split_to2samples(a_obj.data_matrix_df, genre_cat, label_list)
#list_of_genre_dfs.append([[str(periods_list[1]), str(label_list[0])], df_M_a])
list_of_genre_dfs.append([str(periods_list[1])+ "_"+ str(label_list[1]), df_R_a])
#print(df_M_a)
print(df_R_a)