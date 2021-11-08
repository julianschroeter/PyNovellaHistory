import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler


def scatter(df, colors_list):
    array = df.to_numpy()
    data = array[:, 0:(array.shape[1]-1)]
    print(data)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)
    Y = array[:, array.shape[1] - 1]
    Y_list = Y.tolist()
    list_targetlabels = ", ".join(map(str, set(Y_list))).split(", ")

    zipped_dict = dict(zip(list_targetlabels, colors_list[:len(list_targetlabels)]))

    # wenn ein Label, z.B. "other" farblos (weiß) werden soll:
    # zipped_dict["other"] ="white"

    list_colors_target = [zipped_dict[item] for item in Y_list]

    #colors_str = ", ".join(map(str, df["target"].values))
    #colors_str = colors_str.translate(zipped_dict)
    # list_targetcolors = [zipped_dict[label] for label in list_target]
    plt.figure(figsize=(10, 10))
    plt.title("Scatter Plot")
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=list_colors_target, cmap='rainbow',
                alpha=0.8)
    plt.xlabel(str(df.columns.values.tolist()[0]))
    plt.ylabel(str(df.columns.values.tolist()[1]))
    # plt.xscale(value="log")
    # plt.yscale(value="log")
    # plt.xlim(-50000,500000)
    mpatches_list = []
    for key, value in zipped_dict.items():
        patch = mpatches.Patch(color=value, label=key)
        mpatches_list.append(patch)
    plt.legend(handles=mpatches_list)
    plt.show()