system = "wcph113"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')

# from my own modules:
from preprocessing.presetting import global_corpus_representation_directory, local_temp_directory
from preprocessing.corpus import DocFeatureMatrix
from preprocessing.metadata_transformation import standardize_meta_data_medium, full_genre_labels

# standard libraries
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns
import pymc as pm
from pymc import HalfCauchy, Model, Normal
import arviz as az
import bambi as bmb

media_cat_name = "Medium_ED"

scaler = StandardScaler()
scaler = MinMaxScaler()
columns_transl_dict = {"Gewaltverbrechen":"Gewaltverbrechen", "verlassen": "SentLexFear", "grässlich":"embedding_Angstempfinden",
                       "Klinge":"Kampf", "Oberleutnant": "Krieg", "rauschen":"UnbekannteEindr", "Dauerregen":"Sturm",
                       "zerstören": "Feuer", "entführen":"Entführung", "lieben": "Liebe", "Brustwarzen": "Erotik"}


dangers_list = ["Gewaltverbrechen", "Kampf", "Krieg", "Sturm", "Feuer", "Entführung"]
dangers_colors = ["cyan", "orange", "magenta", "blue", "pink", "purple"]
dangers_dict = dict(zip(dangers_list, dangers_colors[:len(dangers_list)]))

dangers_mpatches_list = []
for media, color in dangers_dict.items():
    patch = mpatches.Patch(color=color, label=media.capitalize())
    dangers_mpatches_list.append(patch)


infile_path = os.path.join(local_temp_directory(system), "MaxDangerFearCharacters_novellas.csv") # "All_Chunks_Danger_FearCharacters_novellas.csv" # all chunks

matrix = DocFeatureMatrix(data_matrix_filepath=infile_path)

df = matrix.data_matrix_df

df = df.rename(columns=columns_transl_dict)
df["doc_id"] = df.index

metadata_filepath = os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv" )
metadata_df = pd.read_csv(metadata_filepath, index_col=0)

df[media_cat_name] = df.apply(lambda x: metadata_df.loc[x["doc_id"], media_cat_name], axis=1)
df[media_cat_name] = df[media_cat_name].fillna("unknown")

df["genre"] = df.apply(lambda x: metadata_df.loc[x["doc_id"], "Gattungslabel_ED_normalisiert"], axis=1)
df["name"] = df.apply(lambda x: metadata_df.loc[x["doc_id"], "Nachname"], axis=1)
df_media_genre = standardize_meta_data_medium(df=df, medium_column_name=media_cat_name)

df = df_media_genre
df = df[df.isin({"genre":["N", "E", "0E", "XE"]}).any(1)]

replace_dict = {"genre": {"N": "N", "E": "E", "0E": "MLP",
                                    "R": "R", "M": "M", "XE": "MLP"}}
df = full_genre_labels(df, replace_dict=replace_dict)


df = df[df.isin({"medium_type":["Familienblatt", "Anthologie", "Rundschau", "Taschenbuch"]}).any(1)]

data = df.loc[:, ("max_value","genre", "medium_type")]

lm = ols('max_value ~ C(genre, Sum)+C(medium_type, Sum)',
                 data=data).fit()
anova_table = sm.stats.anova_lm(lm, typ=2) # Type 2  Anova DataFrame
anova_ann = "Results of the Anova test:"
print(anova_ann)
print(anova_table)


variables = ["genre", "medium_type", "max_value"]
predicted_var = variables[2]
predictor_var = (variables[0], variables[1])
Y = df[predicted_var].values
X = df.loc[:,predictor_var]

# independent variables to one-hot encoding:
X = pd.get_dummies(data=X, drop_first=False)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
print(X_train.shape)
print(X_test.shape)

# do a scikit-learn linear regression
model = LinearRegression()
model.fit(X_train,y_train)
coeff_parameter = pd.DataFrame(model.coef_,X.columns,columns=['Coefficient'])
print(coeff_parameter)
predictions = model.predict(X_test)
sns.regplot(x= y_test, y= predictions)
plt.show()



X_train_Sm = sm.add_constant(X_train)
ls = sm.OLS(y_train,X_train_Sm).fit()
print("Predictor variables: ", predictor_var)
print("Predicted variables: ", predicted_var)
print(ls.summary())


# generate a model with bambi:
model = bmb.Model('max_value ~ C(genre, Sum) + C(medium_type, Sum)', data)
# Fit the model using 1000 on each of 4 chains
results = model.fit(draws=3000, chains=4)

# Use ArviZ to plot the results
az.rcParams["plot.matplotlib.show"] = True
az.plot_trace(results)
plt.savefig(os.path.join(local_temp_directory(system), "bambi_traceplot_genre-medium-regression-on-suspense.png"))
plt.show()


# Key summary and diagnostic info on the model parameters
bambi_summary = az.summary(results)
ann_bambi = "These are the results for a Bayesian modeling based on bambi:"
print(ann_bambi)
print(bambi_summary)


#with pm.Model() as model:  # model specifications in PyMC3 are wrapped in a with-statement
    # Define priors
 #   sigma = pm.HalfCauchy("sigma", beta=10, testval=1.0)
#    intercept = Normal("Intercept", 0, sigma=20)
   # genre_E_coeff = Normal("genre_E_coeff", 0, sigma=20)
 ##   genre_M_coeff = Normal("genre_M_coeff", 0, sigma=20)
   # genre_N_coeff = Normal("genre_N_coeff", 0, sigma=20)
   # genre_MLP_coeff = Normal("genre_MLP_coeff", 0, sigma=20)
 #   media_Anth_coeff = Normal("media_Anth_coeff", 0, sigma=20)
  #  media_FB_coeff = Normal("media_FB_coeff", 0, sigma=20)
   # media_RS_coeff = Normal("media_RS_coeff", 0, sigma=20)
  #  media_TB_coeff = Normal("media_TB_coeff", 0, sigma=20)

    # Define likelihood
   # likelihood = Normal("y", mu=intercept + genre_E_coeff * X.genre_E +
    #            #                genre_M_coeff * X.genre_M +
     #                   genre_N_coeff * X.genre_N +
      #                          genre_MLP_coeff * X.genre_MLP +
       #                         media_Anth_coeff * X.medium_type_Anthologie +
        #                media_FB_coeff * X.medium_type_Familienblatt +
         #                       media_RS_coeff + X.medium_type_Rundschau +
          #                      media_TB_coeff * X.medium_type_Taschenbuch,
           #             sigma=sigma, observed=Y)
#
    # Inference!
 #   # draw 3000 posterior samples using NUTS sampling
  #  trace = pm.sample(1000, return_inferencedata=True)


#az.plot_trace(trace, figsize=(10, 7))
#plt.savefig(os.path.join(local_temp_directory(system), "pymc_traceplot_genre-medium-regression-on-suspense.png"))
#plt.show()
#pymc_summary = az.summary(trace)
#pymc_ann = "These are the results of the pymc modeling: "
#print(pymc_ann)
#print(pymc_summary)


results_str = "\n".join([anova_ann, anova_table.to_string(), ann_bambi, bambi_summary.to_string()])
with open(os.path.join(local_temp_directory(system), "genre_media_regression_on_suspense_results.txt"), "w", encoding="utf8") as file:
    file.write(results_str)