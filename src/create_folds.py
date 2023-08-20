import pandas as pd
from utils import get_path_df
from pathlib import Path


def label_path_df(path_df, label_df):
    """
    Annotate the path_df with aspect labels based on associations in label_df.

    Parameters:
    - path_df (pd.DataFrame): The dataframe containing 'EntryID' and associated paths.
    - label_df (pd.DataFrame): The dataframe containing 'EntryID' and associated 'aspect' values.

    Returns:
    - pd.DataFrame: The annotated path_df with added columns for each aspect ('BPO', 'CCO', 'MFO').
                     Each new column has boolean values (0 or 1) indicating the presence of the associated aspect.
    """
    # Create a dictionary to map EntryID to its associated aspects
    entryid_to_aspects = label_df.groupby("EntryID")["aspect"].apply(set).to_dict()

    # Define a function to check if an EntryID is associated with a given aspect
    def has_aspect(entryid, aspect):
        return 1 if aspect in entryid_to_aspects.get(entryid, set()) else 0

    aspects = ["BPO", "CCO", "MFO"]
    for aspect in aspects:
        # Add the new columns to paths_df for each subont
        path_df[aspect] = path_df["EntryID"].apply(lambda x: has_aspect(x, aspect))

    return path_df


if __name__ == "__main__":
    PROCESSED_DATA_DIR = Path("data/processed")
    path_df = get_path_df(data_dir=PROCESSED_DATA_DIR, processed=True)
    path_df.to_csv("data/paths.csv", index=False)
    label_df = pd.read_csv("data/terms.tsv", sep="\t")

    labeled_path_df = label_path_df(path_df, label_df)

    labeled_path_df.to_csv("data/labeled_paths.csv")
    print(path_df.head())
