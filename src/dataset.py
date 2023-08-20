from torch.utils.data import Dataset
import torch
from pathlib import Path
from torch.utils import data as torch_data
import pandas as pd
import pickle
from torchdrug import utils


class GeneOntology(Dataset):
    def __init__(
        self,
        path_df,
        label_df,
        transform=None,
        subontology="BPO",
        split=(0.9, 0.05),
        sample=False,
    ):
        self.subontology = subontology
        # Filter Dataset by subontology
        self.label_df = label_df[label_df["aspect"] == subontology].reset_index(
            drop=True
        )
        self.path_df = path_df[path_df[subontology] == 1].reset_index(drop=True)

        is_sample = self.path_df["set"] == "sample"
        if sample:
            # only use the sample dataset
            self.path_df = self.path_df[is_sample].reset_index(drop=True)
        else:
            self.path_df = self.path_df[~is_sample].reset_index(drop=True)

        # Get all go labels for subontology
        self.tasks = self.label_df["term"].unique()

        # Group by EntryID using pandas groupby and aggregate terms into lists
        self.grouped_terms = label_df.groupby("EntryID")["term"].apply(list).to_dict()

        self.go_term_to_idx = self.go_term_index_mapping(self.tasks)
        self.targets = self.go_term_to_idx
        self.transform = transform
        self.subont = subontology

        self.num_samples = self.compute_split_sizes(split)

    def compute_split_sizes(self, split=(0.9, 0.05)):
        """
        Compute the number of samples for each split (train, test, valid).

        Args:
        - split (tuple or list): Proportions for train and test splits. Validation is inferred.

        Returns:
        - list[int]: List containing the number of samples for train, test, and valid splits.
        """
        total_samples = len(self)
        train_samples = int(split[0] * total_samples)
        test_samples = int(split[1] * total_samples)
        valid_samples = (
            total_samples - train_samples - test_samples
        )  # Remaining for valid

        return train_samples, test_samples, valid_samples

    def __getitem__(self, idx):
        id = self.path_df.at[idx, "EntryID"]
        path = self.path_df.at[idx, "path"]
        go_terms = self.grouped_terms.get(id, [])
        item = {
            "graph": self.read_pickle(path),
            "targets": self.protein_labels_to_sparse_tensor(
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

    def protein_labels_to_sparse_tensor(self, go_terms_associated, go_term_to_idx):
        """
        Convert a list of GO terms associated with a protein to a binary sparse tensor.

        Args:
        - go_terms_associated (list of str): List of GO terms associated with the protein.
        - go_term_to_idx (dict): Mapping from GO term to its index in the tensor.

        Returns:
        - tensor (torch.sparse_coo_tensor): Binary sparse tensor representation of the protein's GO terms.
        """
        # Get the indices of associated GO terms
        indices = [
            go_term_to_idx[go_term]
            for go_term in go_terms_associated
            if go_term in go_term_to_idx
        ]

        # Convert indices to a 2D tensor
        indices_tensor = torch.tensor(indices).unsqueeze(0)

        # Create the sparse tensor
        values = torch.ones(len(indices))  # Non-zero values are 1s
        tensor = utils.sparse_coo_tensor(
            indices=indices_tensor,
            values=values,
            size=[len(self.tasks)],
        )

        return tensor.to_dense()

    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            split = torch_data.Subset(self, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        return splits


if __name__ == "__main__":
    PROCESSED_DATA_DIR = Path("data/processed")
    # path_df = get_path_df(data_dir=PROCESSED_DATA_DIR, processed=True)

    labeled_path_df = pd.read_csv("data/labeled_paths.csv")
    label_df = pd.read_csv("data/terms.tsv", sep="\t")

    bp_go = GeneOntology(path_df=labeled_path_df, label_df=label_df, subontology="BPO")
    mf_go = GeneOntology(path_df=labeled_path_df, label_df=label_df, subontology="MFO")

    cc_go = GeneOntology(
        path_df=labeled_path_df, label_df=label_df, subontology="CCO", sample=True
    )

    train, test, valid = cc_go.split()

    sets = [train, test, valid]
    for s in sets:
        print(s[0]["graph"].node_feature.shape)
        print(s[0]["graph"].edge_feature.shape)
        print(len(s))

    print()
    print(cc_go[0])
    print(len(cc_go))
