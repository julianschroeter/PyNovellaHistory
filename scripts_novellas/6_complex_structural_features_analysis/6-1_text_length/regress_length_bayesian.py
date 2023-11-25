system = "my_xps" # "wcph113"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')

# from my own modules:
from preprocessing.presetting import global_corpus_representation_directory, local_temp_directory
from preprocessing.corpus import DocFeatureMatrix
from preprocessing.metadata_transformation import standardize_meta_data_medium, full_genre_labels, years_to_periods

# standard libraries
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns
import pymc as pm
from pymc import HalfCauchy, Model, Normal
import arviz as az
import bambi as bmb

medium_cat = "Medientyp_ED"
genre_cat = "Gattungslabel_ED_normalisiert"


infile_df_path = os.path.join(local_temp_directory(system), "novella_corpus_length_matrix.csv")
metadata_df_path = os.path.join(global_corpus_representation_directory(system), "Bibliographie.csv")

matrix_obj = DocFeatureMatrix(data_matrix_filepath=infile_df_path, metadata_csv_filepath=metadata_df_path)
matrix_obj = matrix_obj.reduce_to(["token_count"])
matrix_obj = matrix_obj.add_metadata(["Gattungslabel_ED_normalisiert", "Jahr_ED", "Medientyp_ED"])
matrix_obj.data_matrix_df = years_to_periods(matrix_obj.data_matrix_df, category_name="Jahr_ED", start_year=1790, end_year=1941, epoch_length=20, new_periods_column_name="periods")

replace_dict = {"Medientyp_ED": {"Zeitschrift": "Journal", "Zeitung": "Journal",
                                 "Kalender": "Kalender", "Rundschau" : "Rundschau",
                                 "Zyklus" : "Anthologie", "Roman" : "Buch",
                                 "(unbekannt)" : "(unbekannt)",
                                    "Illustrierte": "Journal", "Sammlung": "Anthologie",
                                 "Nachlass": "Buch", "Jahrbuch":"Taschenbuch",
                                 "Monographie": "Buch", "Werke": "Buch"}}
matrix_obj.data_matrix_df = full_genre_labels(matrix_obj.data_matrix_df, replace_dict=replace_dict)


cat_labels = ["N", "E"]
cat_labels = ["N", "E", "0E", "XE"]
matrix_obj = matrix_obj.reduce_to_categories("Gattungslabel_ED_normalisiert", cat_labels)

scaler = MinMaxScaler()
matrix_obj.data_matrix_df.iloc[:, :1] = scaler.fit_transform(matrix_obj.data_matrix_df.iloc[:, :1].to_numpy())

periods20a = ["1800-1820", "1820-1840", "1840-1860", "1860-1880", "1880-1900"]
periods20a = ["1790-1810", "1810-1830", "1830-1850", "1850-1870", "1870-1890", "1890-1910"]
periods20a = ["1850-1870"]

for period in periods20a:
    print(period)
    matrix_obj_period = matrix_obj.reduce_to_categories("periods", [period]) #, "1815-1822" , "1825-1845"

    #df = matrix_obj_period.data_matrix_df

    df = matrix_obj.data_matrix_df
    df["periods"] = df["periods"].astype("str")
    replace_dict = {"Gattungslabel_ED_normalisiert": {"N": "0N", "E": "1E", "0E": "2MLP",
                                    "R": "R", "M": "M", "XE": "2MLP"}}
    df = full_genre_labels(df, replace_dict=replace_dict)
    df = df[df.isin({medium_cat:["Familienblatt", "Anthologie", "Taschenbuch", "Rundschau", "Journal"]}).any(axis=1)]


    data = df.loc[:, ("token_count",genre_cat,medium_cat, "periods")]


    print("Model without interaction covering media only:")
    lm = ols('token_count ~ C(Medientyp_ED, Sum)',
                 data=data).fit()
    anova_table = sm.stats.anova_lm(lm, typ=2)  # Type 2  Anova DataFrame
    anova_ann = "Results of the Anova test:"
    print(anova_ann)
    print(anova_table)
    deltas_df = anova_table
    deltas_df["adj_sum_sq"] = anova_table["sum_sq"] / anova_table["df"]
    print(deltas_df)
    lm_summary = lm.summary()
    print(lm_summary)

    print("Model without interaction covering genre only:")
    lm = ols('token_count ~ C(Gattungslabel_ED_normalisiert, Sum)',
                 data=data).fit()
    anova_table = sm.stats.anova_lm(lm, typ=2)  # Type 2  Anova DataFrame
    anova_ann = "Results of the Anova test:"
    print(anova_ann)
    print(anova_table)
    deltas_df = anova_table
    deltas_df["adj_sum_sq"] = anova_table["sum_sq"] / anova_table["df"]
    print(deltas_df)
    lm_summary = lm.summary()
    print(lm_summary)

    print("Model including an interaction between genre and medium:")
    lm = ols('token_count ~ C(Gattungslabel_ED_normalisiert, Sum)+C(Medientyp_ED, Sum) + C(Gattungslabel_ED_normalisiert, Sum)*C(Medientyp_ED, Sum)',
                 data=data).fit()
    anova_table = sm.stats.anova_lm(lm, typ=2)  # Type 2  Anova DataFrame
    anova_ann = "Results of the Anova test:"
    print(anova_ann)
    print(anova_table)
    deltas_df = anova_table
    deltas_df["adj_sum_sq"] = anova_table["sum_sq"] / anova_table["df"]
    print(deltas_df)
    lm_summary = lm.summary()
    print(lm_summary)

    print("Model without interaction:")
    lm = ols('token_count ~ C(Gattungslabel_ED_normalisiert, Sum)+C(Medientyp_ED, Sum)',
                 data=data).fit()
    anova_table = sm.stats.anova_lm(lm, typ=2) # Type 2  Anova DataFrame
    anova_ann = "Results of the Anova test:"
    print(anova_ann)
    print(anova_table)
    deltas_df = anova_table
    deltas_df["adj_sum_sq"] = anova_table["sum_sq"] / anova_table["df"]
    print(deltas_df)
    print("Delta Genre: ", deltas_df.iloc[0,4] / (deltas_df.iloc[0,4] + deltas_df.iloc[1,4]))
    print("Delta Medium: ", deltas_df.iloc[1,4] / (deltas_df.iloc[0,4] + deltas_df.iloc[1,4]))
    lm_summary = lm.summary()
    print(lm_summary)


variables = ["Gattungslabel_ED_normalisiert", "Medientyp_ED", "periods", "token_count"]
predicted_var = variables[3]
predictor_var = (variables[0], variables[1])
Y = df[predicted_var].values
X = df.loc[:,predictor_var]

# independent variables to one-hot encoding:
X = pd.get_dummies(data=X, drop_first=False).astype(int)
print(X)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=5)
print(X_train.shape)
print(X_test.shape)

# do a scikit-learn linear regression
model = LinearRegression()
model.fit(X_train,y_train)
coeff_parameter = pd.DataFrame(model.coef_,X.columns,columns=['Coefficient'])
print(coeff_parameter)
predictions = model.predict(X_test)
sns.regplot(x= y_test, y= predictions)
sns.regplot(x = y_train, y= model.predict(X_train), color="red")
plt.show()


r2_train = r2_score(y_train, model.predict(X_train))
r2_test = r2_score(y_test, model.predict(X_test))
# compare r2 for train and test sets
print("r2 train: ", r2_train, " | r2 test: ", r2_test)

print("train a linear regression (OLS) without specified model:")
X_train_Sm = sm.add_constant(X_train)
ls = sm.OLS(y_train,X_train_Sm).fit()
print("Predictor variables: ", predictor_var)
print("Predicted variables: ", predicted_var)
print(ls.summary())


print("train a GLM without specified model:")
X_train_Sm = sm.add_constant(X_train)
ls = sm.GLM(y_train,X_train_Sm).fit()
print("Predictor variables: ", predictor_var)
print("Predicted variables: ", predicted_var)
print(ls.summary())


# generate a model with bambi:


print("Bayesian Model with full interaction of all variables:")
model = bmb.Model('token_count ~ C(Gattungslabel_ED_normalisiert, Sum) + C(Medientyp_ED, Sum) + C(Gattungslabel_ED_normalisiert, Sum) * C(Medientyp_ED, Sum)'
                  , data)

print("Bayesian Model with interaction of variables:")
model = bmb.Model('token_count ~ C(Gattungslabel_ED_normalisiert, Sum) * C(Medientyp_ED, Sum)'
                  , data)

print("Bayesian Model without interaction of variables:")
model = bmb.Model('token_count ~ C(Gattungslabel_ED_normalisiert, Sum) + C(Medientyp_ED, Sum) + C(periods, Sum)'
                  , data)


# Fit the model using 1000 on each of 4 chains
results = model.fit(draws=1000, chains=4)

# Use ArviZ to plot the results
az.rcParams["plot.matplotlib.show"] = True

az.plot_density(results)
plt.show()


az.plot_posterior(results)
plt.show()

az.plot_trace(results, compact=True, legend=True, backend="matplotlib")
plt.tight_layout()
plt.savefig(os.path.join(local_temp_directory(system), "bambi_traceplot_length_on_genre-medium-regression.png"))

plt.show()



# Key summary and diagnostic info on the model parameters
bambi_summary = az.summary(results)
ann_bambi = "These are the results for a Bayesian modeling based on bambi:"
print(ann_bambi)
print(bambi_summary)


with pm.Model() as model:  # model specifications in PyMC3 are wrapped in a with-statement
   # Define priors
   sigma = pm.HalfCauchy("sigma", beta=10, testval=1.0)
   intercept = Normal("Intercept", 0, sigma=20)
   genre_E_coeff = Normal("genre_E_coeff", 0, sigma=20)

   genre_N_coeff = Normal("genre_N_coeff", 0, sigma=20)
   genre_MLP_coeff = Normal("genre_MLP_coeff", 0, sigma=20)
   media_Anth_coeff = Normal("media_Anth_coeff", 0, sigma=20)
   media_FB_coeff = Normal("media_FB_coeff", 0, sigma=20)
   media_RS_coeff = Normal("media_RS_coeff", 0, sigma=20)
   media_TB_coeff = Normal("media_TB_coeff", 0, sigma=20)


   # Define likelihood
   likelihood = Normal("Y", mu=intercept + genre_E_coeff * X["Gattungslabel_ED_normalisiert_E"] +
                        genre_N_coeff * X["Gattungslabel_ED_normalisiert_N"] +
                               genre_MLP_coeff * X["Gattungslabel_ED_normalisiert_MLP"] +
                                media_Anth_coeff * X["Medientyp_ED_Anthologie"] +
                                media_TB_coeff * X["Medientyp_ED_Taschenbuch"],
                        sigma=sigma, observed=Y)

   trace = pm.sample(4000, return_inferencedata=True)


az.plot_trace(trace, legend=True)
plt.tight_layout()
plt.savefig(os.path.join(local_temp_directory(system), "pymc_traceplot_regress_length_on_genre-medium.png"))

az.plot_posterior(trace)
plt.tight_layout()
plt.savefig(os.path.join(local_temp_directory(system), "pymc_plotposterior_regress_length_on_genre-medium.png"))
plt.show()


pymc_summary = az.summary(trace)
pymc_ann = "These are the results of the pymc modeling: "
print(pymc_ann)
print(pymc_summary)


results_str = "\n".join([anova_ann, anova_table.to_string(), "lm summary: ",lm_summary.as_text(), ann_bambi, bambi_summary.to_string()])
with open(os.path.join(local_temp_directory(system), "regress_length_on_genre_and_media_results.txt"), "w", encoding="utf8") as file:
    file.write(results_str)

bambi_summary.to_csv(os.path.join(local_temp_directory(system), "bamby_summary.csv"))