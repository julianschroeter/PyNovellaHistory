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


def rd_vectors_around_center(probabs):
    rd_angles = np.random.uniform(-np.pi,(np.pi),len(probabs))
    x_results = np.multiply(np.cos(rd_angles),probabs)
    y_results = np.multiply(np.sin(rd_angles),probabs)
    return x_results, y_results

def plot_prototype_concepts(probabs, labels, threshold, annotation=None,
                            lang="en", legend_dict=None, filepath=None):

    x_results, y_results = rd_vectors_around_center(probabs)
    lower_threshold_level = 0.5 - (threshold / 2)
    upper_threshold_level = 0.5 + (threshold / 2)
    fig, ax = plt.subplots()
    # fig = plt.figure(figsize=(5,5))
    ax.scatter(x_results, y_results, c=labels)

    ax.add_patch(plt.Circle((0, 0), 1, fill=False))
    ax.add_patch(plt.Circle((0, 0), lower_threshold_level, fill=False))
    ax.add_patch(plt.Circle((0, 0), upper_threshold_level, fill=False))

    rd_angle = np.random.uniform(-np.pi, np.pi, 1)
    if annotation:
        x_results = np.multiply(np.cos(rd_angle), annotation[1])
        y_results = np.multiply(np.sin(rd_angle), annotation[1])
        plt.annotate(annotation[0], (x_results, y_results), arrowprops=dict(facecolor='black', shrink=0.05))
    if lang == "de":
        plt.title("Prototypenkonzept für Genre-Paare")
        plt.xlabel(str("Grenzbereich der Unentscheidbarkeit: " + str(lower_threshold_level) + " – " + str(upper_threshold_level)))
        plt.ylabel("Nähe zum Prototypenzentrum (inv. Vorhersagewahrsch.)")

    if lang == "en":
        plt.title("Prototype Concept for Genre Pairs")
        plt.xlabel(str("Boundary of undecidabilty: "+str(lower_threshold_level)+ " – "+ str(upper_threshold_level)))
        plt.ylabel("closeness to center (inv. pred. prob.)")

    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

    if legend_dict:
        mpatches_list = []
        for key, value in legend_dict.items():
            patch = mpatches.Patch(color=value, label=key)
            mpatches_list.append(patch)
        plt.legend(handles=mpatches_list)

    if filepath:
        plt.savefig(filepath)

    plt.show()
