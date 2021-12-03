from numpy.core.numeric import ones
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csv_path = "./RefSeq_results.csv"
data = pd.read_csv(csv_path)
# print(data)
print(data.columns)
# print(data.values)


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

    return tmp


def str_to_array(array_str):
    """
    Converts a bad array string from csv into a good numpy array
    """
    tmp_str = ""
    for c in array_str:
        if c not in {"[", "]", "\n"}:
            tmp_str += c

    res = np.fromstring(tmp_str, sep=" ")
    return res


def plot_model_bargraph(ones_hot, cat):
    """
    Plots a one's hot row against a cat row.
    Bar graph of all the statistics
    """
    print(cat)
    print(np.mean(str_to_array(cat.values[0, 12])))
    plt.bar(range(1), cat["AUROC"])

    plt.show()  # should save to a file instead
    pass


def plot_model_(ones_hot, cat):
    """
    Plots a set of ones_hot rows against set of cat rows.
    One statistic across some parameter sweep
    Maybe not necessary
    """
    pass


def plot_all_models(bargraph=True):
    for dataset in set(data["Dataset"]):
        for architecture in ["CNNNet", "LSTMNet", "CNNLSTMNet"]:
            # not reading these from the data because these have different values between encodings (21 and 0)
            for h1 in [16, 32, 64, 128, 256]:
                for h2 in [16, 32, 64, 128, 256]:
                    for validation in [True, False]:
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

                        plot_model_bargraph(ones_hot, cat)


print(get_row("RefSeq", "CNNLSTMNet", "CAT", 16, 16, 0, True))
# plot_all_models(bargraph=True)
