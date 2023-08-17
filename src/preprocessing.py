from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pickle
from tqdm.auto import tqdm
from torchdrug.data import Protein
import pandas as pd
import re


def extract_pdb_info(s):
    pattern = r"^(?P<database>[^-]+)-(?P<entry_id>[^-]+)-(?P<misc>.+)\.pdb$"
    match = re.match(pattern, s)
    if match:
        return match.groupdict()
    else:
        return None


def get_path_df(data_dir):
    path_list = []
    for dataset in data_dir.iterdir():
        if dataset.is_dir():
            for file in dataset.iterdir():
                d = extract_pdb_info(file.name)
                d["path"] = str(file)
                d["set"] = dataset.name
                path_list.append(d)
    return pd.DataFrame(path_list)


def pickle_protein(protein, path):
    path.parent.mkdir(exist_ok=True)
    with open(f"{path}.pkl", "wb") as file:
        pickle.dump(protein, file)


def log_error_to_file(error_message, log_file="errors.log"):
    """Append error messages to a log file."""
    with open(log_file, "a") as file:
        file.write(error_message + "\n")


class PreProcessingDataset(Dataset):
    def __init__(self, path_df, pdb_dir="data", transform=None):
        self.path_df = path_df.reset_index()
        self.pdb_dir = Path(pdb_dir)
        self.transform = transform

    def __getitem__(self, idx):
        pdb_path = self.path_df.at[idx, "path"]
        label = self.path_df.at[idx, "entry_id"]
        dataset = self.path_df.at[idx, "set"]
        try:
            protein = Protein.from_pdb(pdb_path)
        except Exception as e:
            error_message = f"Error processing entry {label} in dataset {dataset} at path {pdb_path}: {e}"
            log_error_to_file(error_message)
            protein = None
        return protein, label, dataset

    def __len__(self):
        return len(self.path_df)


if __name__ == "__main__":
    RAW_DATA_DIR = Path("data/raw")
    PROCESSED_DATA_DIR = Path("data/processed")

    df = get_path_df(RAW_DATA_DIR)
    df.to_csv(RAW_DATA_DIR / "paths.csv", index=False)

    df = pd.read_csv(RAW_DATA_DIR / "paths.csv")

    sample_df = df[df["set"] == "sample"].reset_index(drop=True)

    preprocessing_dataset = PreProcessingDataset(path_df=sample_df)

    for i, (protein, entry_id, dataset) in enumerate(
        tqdm(preprocessing_dataset, colour="green")
    ):
        if protein:
            pickle_protein(protein, PROCESSED_DATA_DIR / dataset / entry_id)

        # This is less than ideal, we should be using a dataloader,
        # dataloaders are designed to handle arrays, not torchdrug.proteins
        # torchdrug.protein objects are graphs, so we need to find a different collate
        # function for our dataloaders, thankfully torchdrug already has their own implementation
        # of the dataloader class, the question now is how to save them to hdf5 files
        if i == len(preprocessing_dataset) - 1:
            break
