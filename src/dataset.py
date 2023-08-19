from torch.utils.data import Dataset
import torch
from utils import get_path_df
from pathlib import Path
import pickle
import pandas as pd


class GeneOntology(Dataset):
    def __init__(
        self, path_df, label_df, dataset_dir, transform=None, subontology="BPO"
    ):
        self.path_df = path_df.reset_index(drop=True)
        label_df = label_df[label_df["aspect"] == subontology].reset_index(drop=True)
        all_go_terms = label_df["term"].unique()

        # Group by EntryID using pandas groupby and aggregate terms into lists
        self.grouped_terms = label_df.groupby("EntryID")["term"].apply(list).to_dict()

        self.go_term_to_idx = self.go_term_index_mapping(all_go_terms)
        self.pdb_dir = Path(dataset_dir)
        self.transform = transform
        self.subont = subontology

    def __getitem__(self, idx):
        id = self.path_df.at[idx, "entry_id"]
        path = self.path_df.at[idx, "path"]
        go_terms = self.grouped_terms.get(id, [])
        print(f"{id}: {go_terms}")

        item = {
            "graph": self.read_pickle(path),
            "id": id,
            "dataset": self.path_df.at[idx, "set"],
            "label": self.protein_labels_to_tensor(
                go_terms_associated=go_terms, go_term_to_idx=self.go_term_to_idx
            ),
        }

        if self.transform:
            item = self.transform(item)
        return item

    def __len__(self):
        return len(self.path_df)

    def read_pickle(self, path):
        with open(path, "rb") as file:
            return pickle.load(file)

    def go_term_index_mapping(self, go_terms_list):
        """
        Create a dictionary that maps each GO term to a unique index.
        """
        return {go_term: idx for idx, go_term in enumerate(go_terms_list)}

    def protein_labels_to_tensor(self, go_terms_associated, go_term_to_idx):
        """
        Convert a list of GO terms associated with a protein to a binary tensor.

        Args:
        - go_terms_associated (list of str): List of GO terms associated with the protein.
        - go_term_to_idx (dict): Mapping from GO term to its index in the tensor.

        Returns:
        - tensor (torch.Tensor): Binary tensor representation of the protein's GO terms.
        """
        tensor = torch.zeros(len(go_term_to_idx))
        for go_term in go_terms_associated:
            if (
                go_term in go_term_to_idx
            ):  # Ensure the GO term exists in the mapping (it should!)
                tensor[go_term_to_idx[go_term]] = 1
        return tensor


if __name__ == "__main__":
    PROCESSED_DATA_DIR = Path("data/processed")
    path_df = get_path_df(data_dir=PROCESSED_DATA_DIR, processed=True)
    path_df = path_df[path_df["set"] == "train"]
    label_df = pd.read_csv("data/terms.tsv", sep="\t")
    go = GeneOntology(
        path_df=path_df, label_df=label_df, dataset_dir=PROCESSED_DATA_DIR
    )

    print(go[0])
    print()
    print(go[10])
    # # Example usage
    # # For demonstration, let's assume the following list of all GO terms
    # all_go_terms = ["GO:0001", "GO:0002", "GO:0003", "GO:0004"]
    # go_term_to_idx = go.create_go_term_index_mapping(all_go_terms)

    # # Let's say a protein has the following GO terms associated
    # protein_go_terms = ["GO:0001", "GO:0003"]

    # # Convert to tensor
    # protein_tensor = go.protein_labels_to_tensor(protein_go_terms, go_term_to_idx)
    # protein_tensor
