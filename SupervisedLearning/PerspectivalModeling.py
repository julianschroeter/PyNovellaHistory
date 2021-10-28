from Preprocessing.SamplingMethods import equal_sample

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.metrics import accuracy_score, f1_score

def sample_perspectival_sets(input_df, period_category="periods_100a", metadata_cat_fit_list=["1790-1850"],
                             metadata_cat_transfer_list=["1850-1950"], genre_category="Gattungslabel_ED", genre_labels=None,
                             equal_sample_size_fit=True,equal_sample_size_transfer=True,
                             minor_frac_fit=1.0, minor_frac_transfer=1.0):
    """
    generates a training or validation set that is not based on randomization but on metadata such as historical periods
    input_df is a Pandas Data Frame as a document term matrix with two meta data columns: one with dependent variable for the
    classification task (for example genre labels) and one with the meta data categories that determines the training and the validation
    set (for example: metadata_cat_train "texts before 1850" and metadata_cat_val "texts after 1850")
    genre_labels is a list of two genre labels, for example ["N", "E"]
    """
    fit_df = input_df[input_df.isin({period_category: metadata_cat_fit_list}).any(1)]
    fit_df.drop([period_category], axis=1, inplace=True)
    transfer_df = input_df[input_df.isin({period_category: metadata_cat_transfer_list}).any(1)]
    transfer_df.drop([period_category], axis=1, inplace=True)
    df_genre0_fit_df = fit_df[fit_df[genre_category] == genre_labels[0]]
    df_genre1_fit_df = fit_df[fit_df[genre_category] == genre_labels[1]]
    df_genre0_transfer_df = transfer_df[transfer_df[genre_category] == genre_labels[0]]
    df_genre1_transfer_df = transfer_df[transfer_df[genre_category] == genre_labels[1]]
    equal_fit_df = equal_sample(df_genre0_fit_df, df_genre1_fit_df, minor_frac=minor_frac_fit)
    equal_transfer_df = equal_sample(df_genre0_transfer_df, df_genre1_transfer_df, minor_frac=minor_frac_transfer)
    if equal_sample_size_fit == False:
        if equal_sample_size_transfer == False:
            return fit_df, transfer_df
        else:
            return fit_df, equal_transfer_df
    elif equal_sample_size_fit == True:
        if equal_sample_size_transfer == True:
            return equal_fit_df, equal_transfer_df
        elif equal_sample_size_transfer == False:
            return equal_fit_df, transfer_df


def split_features_labels(input_df):
    array = input_df.to_numpy()
    X = array[:, 0:(array.shape[1]-1)]
    Y = array[:, array.shape[1]-1]
    return X,Y

def perspectival_sets_split(model_fit_df, model_transfer_df):
    X_model_fit, Y_model_fit = split_features_labels(model_fit_df)
    X_model_transfer, Y_model_transfer = split_features_labels(model_transfer_df)
    return X_model_fit, X_model_transfer, Y_model_fit, Y_model_transfer


def LR_perspectival_resample(n, fit_val_size, input_df, period_category, metadata_cat_fit_list,
                             metadata_cat_transfer_list, genre_category, genre_labels,
                             equal_sample_size_fit,equal_sample_size_transfer,
                             minor_frac_fit, minor_frac_transfer):
    """

    """
    fit_accuracy_scores_list = []
    fit_f1scores_list = []
    transfer_accuracy_scores_list = []
    transfer_f1scores_list = []
    for i in range(n):
        fit_df, transfer_df = sample_perspectival_sets(input_df, period_category, metadata_cat_fit_list,
                             metadata_cat_transfer_list, genre_category, genre_labels,
                             equal_sample_size_fit,equal_sample_size_transfer,
                             minor_frac_fit, minor_frac_transfer)

        X_model_fit, X_model_transfer, Y_model_fit, Y_model_transfer = perspectival_sets_split(fit_df, transfer_df)
        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X_model_fit, Y_model_fit, test_size=fit_val_size,
                                                                                        random_state=42)
        lr_model = LogisticRegression(solver='lbfgs', multi_class='ovr')
        lr_model.fit(X_train, Y_train)
        fit_predictions = lr_model.predict(X_validation)
        transfer_predictions = lr_model.predict(X_model_transfer)
        fit_accuracy_scores_list.append(accuracy_score(Y_validation, fit_predictions))
        fit_f1scores_list.append(f1_score(Y_validation, fit_predictions, pos_label=genre_labels[0]))
        transfer_accuracy_scores_list.append(accuracy_score(Y_model_transfer, transfer_predictions))
        transfer_f1scores_list.append(f1_score(Y_model_transfer, transfer_predictions, pos_label=genre_labels[0]))
    return fit_accuracy_scores_list, fit_f1scores_list, transfer_accuracy_scores_list, transfer_f1scores_list