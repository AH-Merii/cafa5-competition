from torch.utils.data import Dataset
import torch
from pathlib import Path
import pandas as pd
import pickle
from torchdrug import utils
from sklearn.model_selection import train_test_split


def split_data(df, train_ratio=0.7, test_ratio=0.2, seed=None):
    """
    Split the data into train, test, and validation sets based on EntryID.

    Parameters:
    - df: DataFrame containing the data.
    - train_ratio: Ratio of data to be used for training.
    - test_ratio: Ratio of data to be used for testing.
    - seed: Random seed for reproducibility.

    Returns:
    - train_df, test_df, valid_df: Train, test, and validation DataFrames.
    """

    # Determine validation ratio
    valid_ratio = 1 - train_ratio - test_ratio

    # Get unique EntryIDs
    entry_ids = df["EntryID"].unique()

    # Split EntryIDs for train, test, and validation
    train_entry_ids, temp_entry_ids = train_test_split(
        entry_ids, test_size=(test_ratio + valid_ratio), random_state=seed
    )
    test_entry_ids, valid_entry_ids = train_test_split(
        temp_entry_ids,
        test_size=valid_ratio / (test_ratio + valid_ratio),
        random_state=seed,
    )

    # Filter the main dataframe based on EntryIDs to get train, test, and validation dataframes
    train_df = df[df["EntryID"].isin(train_entry_ids)].reset_index(drop=True)
    test_df = df[df["EntryID"].isin(test_entry_ids)].reset_index(drop=True)
    valid_df = df[df["EntryID"].isin(valid_entry_ids)].reset_index(drop=True)

    return train_df, test_df, valid_df


def split_dataset(dataset, train_ratio=0.7, test_ratio=0.2, seed=None):
    """
    Split a GeneOntology dataset into train, test, and validation datasets.

    Parameters:
    - dataset (GeneOntology): The GeneOntology dataset instance.
    - train_ratio (float): Ratio of data to be used for training.
    - test_ratio (float): Ratio of data to be used for testing.
    - seed (int, optional): Random seed for reproducibility.

    Returns:
    - train_dataset, test_dataset, valid_dataset: Train, test, and validation GeneOntology datasets.
    """

    # Extract path_df from the provided dataset
    path_df = dataset.path_df

    # Split path_df using the split_data function
    train_df, test_df, valid_df = split_data(path_df, train_ratio, test_ratio, seed)

    # Create new GeneOntology dataset instances for each split
    train_dataset = GeneOntology(
        train_df,
        dataset.label_df,
        transform=dataset.transform,
        subontology=dataset.subontology,
    )
    test_dataset = GeneOntology(
        test_df,
        dataset.label_df,
        transform=dataset.transform,
        subontology=dataset.subontology,
    )
    valid_dataset = GeneOntology(
        valid_df,
        dataset.label_df,
        transform=dataset.transform,
        subontology=dataset.subontology,
    )

    return train_dataset, test_dataset, valid_dataset


class GeneOntology(Dataset):
    def __init__(self, path_df, label_df, transform=None, subontology="BPO"):
        self.subontology = subontology
        # Filter Dataset by subontology
        self.label_df = label_df[label_df["aspect"] == subontology].reset_index(
            drop=True
        )
        self.path_df = path_df[path_df[subontology] == 1].reset_index(drop=True)

        # Get all go labels for subontology
        self.tasks = self.label_df["term"].unique()

        # Group by EntryID using pandas groupby and aggregate terms into lists
        self.grouped_terms = label_df.groupby("EntryID")["term"].apply(list).to_dict()

        self.go_term_to_idx = self.go_term_index_mapping(self.tasks)
        self.targets = self.go_term_to_idx
        self.transform = transform
        self.subont = subontology

    def __getitem__(self, idx):
        id = self.path_df.at[idx, "EntryID"]
        path = self.path_df.at[idx, "path"]
        go_terms = self.grouped_terms.get(id, [])
        item = {
            "graph": self.read_pickle(path)[0][0],
            # "id": id,
            # "aspect": self.subontology,
            # "dataset": self.path_df.at[idx, "set"],
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

    def protein_labels_to_tensor(self, go_terms_associated, go_term_to_idx):
        """
        Convert a list of GO terms associated with a protein to a binary tensor.

        Args:
        - go_terms_associated (list of str): List of GO terms associated with the protein.
        - go_term_to_idx (dict): Mapping from GO term to its index in the tensor.

        Returns:
        - tensor (torch.Tensor): Binary tensor representation of the protein's GO terms.
        """
        tensor = torch.zeros(len(self.tasks))
        for go_term in go_terms_associated:
            if (
                go_term in go_term_to_idx
            ):  # Ensure the GO term exists in the mapping (it should!)
                tensor[go_term_to_idx[go_term]] = 1
        return tensor

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


if __name__ == "__main__":
    PROCESSED_DATA_DIR = Path("data/processed")
    # path_df = get_path_df(data_dir=PROCESSED_DATA_DIR, processed=True)

    labeled_path_df = pd.read_csv("data/labeled_paths.csv")
    label_df = pd.read_csv("data/terms.tsv", sep="\t")

    bp_go = GeneOntology(
        path_df=labeled_path_df,
        label_df=label_df,
        subontology="BPO",
    )
    mf_go = GeneOntology(
        path_df=labeled_path_df,
        label_df=label_df,
        subontology="MFO",
    )
    cc_go = GeneOntology(
        path_df=labeled_path_df,
        label_df=label_df,
        subontology="CCO",
    )
    # print(bp_go[0])
    # print()
    # print(cc_go[10000])
    # print()
    # print(mf_go[10000])
    # print()

    # print(f"Number of Data Points(MFO): {len(mf_go)}")
    # print(f"Number of Data Points(BPO): {len(bp_go)}")
    # print(f"Number of Data Points(CCO): {len(cc_go)}")
    # print()

    train_mf, test_mf, valid_mf = split_dataset(mf_go)

    mfs = [train_mf, test_mf, valid_mf]
    for mf in mfs:
        attrs = [attr for attr in dir(mf[0]["graph"]) if not attr.startswith("_")]
        print(attrs)
        print(mf[0]["graph"].node_feature.shape)
        print(mf[0]["graph"].edge_feature.shape)
        print(len(mf))
