from pathlib import Path
from torch.utils.data import Dataset
from torchdrug import data
import pandas as pd
import re


def extract_pdb_info(s):
    pattern = r"^(?P<database>[^-]+)-(?P<entry_id>[^-]+)-(?P<misc>.+)\.pdb$"
    match = re.match(pattern, s)
    if match:
        return match.groupdict()
    else:
        return None


def get_path_df(data_dir=Path("data")):
    path_list = []
    for dataset in data_dir.iterdir():
        if dataset.is_dir():
            for file in dataset.iterdir():
                d = extract_pdb_info(file.name)
                d["path"] = str(file)
                d["set"] = dataset.name
                path_list.append(d)
    return pd.DataFrame(path_list)


class PreProcessingDataset(Dataset):
    def __init__(self, path_df, pdb_dir="data", transform=None):
        self.path_df = path_df.reset_index()
        self.pdb_dir = Path(pdb_dir)
        self.transform = transform

    def __getitem__(self, idx):
        pdb_path = self.path_df.at[idx, "path"]
        return data.Protein.from_pdb(pdb_path)

    def __len__(self):
        len(self.path_df)


if __name__ == "__main__":
    df = get_path_df()
    df.to_csv("data/paths.csv", index=False)

    df = pd.read_csv("data/paths.csv")

    # use sample data
    sample_df = df[df["set"] == "sample"].reset_index(drop=True)

    preprocessing_dataset = PreProcessingDataset(path_df=sample_df)
    print(preprocessing_dataset[2])
    print(preprocessing_dataset[5])
