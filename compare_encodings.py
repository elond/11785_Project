import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
To generate plots:
    1 - update csv_path with the path to the model result csv file
    2 - call plot_all_models()
        Plots are optionally saved to png with a line commented out in plot_model_metrics
        Saving to png has overlapping titles and labels, so screenshoting the plot popup is probably better.
"""

csv_path = "./results_minus_refseqlstm.csv"
data = pd.read_csv(csv_path)


# Don't think I'll end up using this
def label_index(label):
    """
    gets column index of a label, too lazy to type out an enum
    """
    return list(data.columns).index(label)


# Don't think I'll end up using this
def only_differ_by_encoding(ones_hot, cat):
    """
    Return true if the two model rows differ only in encoding (size and dataset match)
    """
    # hella janky way to make sure architecture is the same, the csv doesn't have a column for this
    name_end = ones_hot[label_index("NetworkName")].index("_")
    return (
        ones_hot[label_index("NetworkName")][:name_end] == cat[label_index("NetworkName")][:name_end] and
        ones_hot[label_index("2nd Hidden Units")] == cat[label_index("1st Hidden Units")] and
        ones_hot[label_index("3rd Hidden Units")] == cat[label_index("2nd Hidden Units")] and
        ones_hot[label_index("Dataset")] == cat[label_index("Dataset")] and
        ones_hot[label_index("ML_Dataset")] == cat[label_index("ML_Dataset")]
    )


def get_row(dataset=None, architecture=None, encoding=None, h1=None, h2=None, h3=None, validation=None):
    """
    Selects a row from the csv matching all the parameters
    Parameters equal to None are not compared (similar to a '*')
    """
    val_str = "Validation" if validation else "Test"
    tmp = data.loc[(((data["Encoding"] == encoding) if encoding is not None else True) &
                    # (data["NetworkName"][:len(architecture)] == architecture) &
                    ((data["Dataset"] == dataset) if dataset is not None else True) &
                    ((data["1st Hidden Units"] == h1) if h1 is not None else True) &
                    ((data["2nd Hidden Units"] == h2) if h2 is not None else True) &
                    ((data["3rd Hidden Units"] == h3) if h3 is not None else True) &
                    ((data["ML_Dataset"] == val_str) if validation is not None else True))]

    # Janky way to extract correct architecture from the name, no way to slice the name string above
    if architecture is not None:
        for i, name in enumerate(tmp["NetworkName"]):
            if name[:len(architecture)] == architecture:
                return tmp.loc[tmp["NetworkName"] == name]
        
        return pd.DataFrame(None)

    return tmp


def str_to_array(array_str):
    """
    Converts a bad array string from csv into a good numpy array
    """

    tmp_str = ""
    for c in array_str:
        if c not in {"[", "]", "\n", "\r"}:
            tmp_str += c
    res = np.fromstring(tmp_str, sep=" ")
    return res


def plot_model_metrics(ones_hot_results, cat_results, train_set, validation, architecture):
    """
    Plots a set of ones_hot rows against set of cat rows for a common architecture.
    One plot for each statistic (subplot?), x axis is the model sizes, y axis is the metric being plotted
    Should have 12 plots per metric (Test/Val, RefSeq/GenBank, 3 architectures)
    """
    hidden_size_combos = np.array(np.meshgrid([16, 32, 64, 128, 256], [16, 32, 64, 128, 256])).T.reshape(-1, 2)
    metrics = ["Accuracy", "Recall", "Precision", "Specificity", "F1 score", "AUROC"]
    fig, axs = plt.subplots(len(metrics) // 2, 2)
    # print(hidden_size_combos)
    x_vals = range(len(cat_results))
    for i, metric in enumerate(metrics):
        print(f"Metric: {metric}")
        title = f"{architecture} Trained on {train_set}, Validation. Metric {metric}" if validation else f" {architecture} Trained on {train_set}, Test. Metric {metric}"
        if metric != "AUROC":
            ones_hot_metric_results = [np.mean(str_to_array(x.values[0, label_index(metric)])) for x in ones_hot_results]
            cat_metric_results = [np.mean(str_to_array(x.values[0, label_index(metric)])) for x in cat_results]
        else:
            ones_hot_metric_results = [x.values[0, label_index(metric)] for x in ones_hot_results]
            cat_metric_results = [x.values[0, label_index(metric)] for x in cat_results]

        axs[i % 3, i % 2].set_title(title)
        axs[i % 3, i % 2].plot(x_vals, ones_hot_metric_results, cat_metric_results, marker=".")
        axs[i % 3, i % 2].legend(["One-Hot", "CAT"])
        if i % 3 == 2:
            axs[i % 3, i % 2].set_xticks(x_vals)
            axs[i % 3, i % 2].set_xticklabels([str(size.T) for size in hidden_size_combos], Rotation=45)
        else:
            axs[i % 3, i % 2].get_xaxis().set_visible(False)

        # for j in range(len(cat_results)):
        #     axs[i % 3, i % 2].annotate(str(hidden_size_combos[j]), (x_vals[j], (cat_metric_results[j] + ones_hot_metric_results[j]) / 2))
    
    plt.show()

    # saves with overlapping labels. Might be better to set x axis labels instead
    # plt.savefig(f"{architecture}_{train_set}_val_results.png" if validation else f"{architecture}_{train_set}_test_results.png")


def plot_all_models():
    for dataset in ["RefSeq", "GenBank"]:
        for architecture in ["CNNLSTMNet", "LSTMNet", "CNNNet"]:
            # not reading these from the data because these have different values between encodings (21 and 0)
            for validation in [False, True]:
                ones_hot_rows, cat_rows = [], []
                for h1 in [16, 32, 64, 128, 256]:
                    for h2 in [16, 32, 64, 128, 256]:
                        ones_hot = get_row(
                            dataset, architecture, "One_hot", 21, h1, h2, validation)
                        cat = get_row(dataset, architecture,
                                      "CAT", h1, h2, 0, validation)
                        if ones_hot.empty:
                            print(
                                f"No data found for one_hot dataset={dataset}, net={architecture}, h1=21, h2={h1}, h3={h2}, val={validation}")
                            continue

                        if cat.empty:
                            print(
                                f"No data found for CAT dataset={dataset}, net={architecture}, h1={h1}, h2={h2}, h3=0, val={validation}")
                            continue

                        ones_hot_rows.append(ones_hot)
                        cat_rows.append(cat)

                plot_model_metrics(ones_hot_rows, cat_rows, dataset, validation, architecture)


# print(get_row("RefSeq", "LSTMNet", "CAT", 128, 256, 0, None))
plot_all_models()
