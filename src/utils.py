import pandas as pd
from pathlib import Path
import re


def log_error_to_file(error_message, log_file="errors.log"):
    """Append error messages to a log file."""
    with open(log_file, "a") as file:
        file.write(error_message + "\n")


def extract_pdb_info(s, processed=False):
    if processed:
        pattern = r"^(?P<EntryID>[^-]+)\.pkl$"
    else:
        pattern = r"^(?P<database>[^-]+)-(?P<EntryID>[^-]+)-(?P<misc>.+)\.pdb$"
    match = re.match(pattern, s)
    if match:
        return match.groupdict()
    else:
        return None


def get_path_df(data_dir: Path, processed_dir: Path = None, **kwargs) -> pd.DataFrame:
    """
    Create a dataframe containing paths to the PDB files in the specified directory.

    This function traverses through the `data_dir` to find PDB files and returns
    a dataframe with details about each file. If `processed_dir` is provided,
    the function will skip PDB files for which corresponding pickle files already
    exist in the `processed_dir`.

    Parameters:
    - data_dir (Path): The directory containing datasets with PDB files.
    - processed_dir (Path, optional): The directory containing processed pickle files.
                                      If provided, PDB files with corresponding pickle
                                      files in this directory will be skipped.

    Returns:
    - DataFrame: A pandas dataframe with columns ["database", "EntryID", "misc", "path", "set"],
                 where "path" is the full path to the PDB file, and "set" is the dataset name.
    """

    path_list = []
    for dataset in data_dir.iterdir():
        if dataset.is_dir():
            for file in dataset.iterdir():
                d = extract_pdb_info(file.name, **kwargs)

                # If processed_dir is provided, check for existing pickle files and skip if found
                if processed_dir:
                    pickle_path = processed_dir / dataset.name / (d["EntryID"] + ".pkl")
                    if pickle_path.exists():
                        continue

                d["path"] = str(file)
                d["set"] = dataset.name
                path_list.append(d)

    return pd.DataFrame(path_list)
