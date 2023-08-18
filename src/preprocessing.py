from pathlib import Path
import time
from torch.utils.data import Dataset
from torchdrug.data import DataLoader, Protein
import pickle
from tqdm.auto import tqdm
from torchdrug import transforms
from torchdrug.layers import GraphConstruction, geometry
import pandas as pd
import re
import warnings
from rdkit import RDLogger


RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")


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


def log_error_to_file(error_message, log_file="errors.log"):
    """Append error messages to a log file."""
    with open(log_file, "a") as file:
        file.write(error_message + "\n")


class PreProcessingDataset(Dataset):
    def __init__(
        self,
        path_df,
        pdb_dir,
        transform=None,
        atom_feature="default",
        bond_feature="default",
        residue_feature="default",
    ):
        self.path_df = path_df.reset_index()
        self.pdb_dir = Path(pdb_dir)
        self.transform = transform
        self.atom_feature = atom_feature
        self.bond_feature = bond_feature
        self.residue_feature = residue_feature

    def __getitem__(self, idx):
        pdb_path = self.path_df.at[idx, "path"]
        id = self.path_df.at[idx, "entry_id"]
        dataset = self.path_df.at[idx, "set"]
        try:
            protein = Protein.from_pdb(
                pdb_path,
                atom_feature=self.atom_feature,
                bond_feature=self.bond_feature,
                residue_feature=self.residue_feature,
            )
        except Exception as e:
            error_message = f"Error processing entry {id} in dataset {dataset} at path {pdb_path}: {e}"
            log_error_to_file(error_message)
            protein = Protein(atom_type=[], bond_type=[], residue_type=[])

        item = {"graph": protein, "id": id, "dataset": dataset}
        if self.transform:
            item = self.transform(item)

        return item

    def __len__(self):
        return len(self.path_df)


class PreProcessor:
    def __init__(
        self,
        dataset: Dataset,
        batch_size,
        num_workers,
        output_dir,
        verbose=False,
    ):
        self.verbose = verbose
        self.output_dir = output_dir
        self.loader = DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers
        )

    def pickle_item(self, protein, path):
        path.parent.mkdir(exist_ok=True)
        with open(f"{path}.pkl", "wb") as file:
            pickle.dump((protein, path.name), file)

    def process_data(self, graph_transform=False, graph_key="graph"):
        if self.verbose:
            self.loader = tqdm(self.loader, colour="green")

        for i, batch in enumerate(self.loader):
            if graph_transform:
                batch[graph_key] = graph_transform(batch[graph_key])

            for protein, id, subset in zip(
                batch["graph"], batch["id"], batch["dataset"]
            ):
                self.pickle_item(
                    (protein, id, subset), Path(self.output_dir) / subset / id
                )

                # item["graph"], item["id"], item["dataset"]
                # if protein:
                #     pickle_protein(protein, PROCESSED_DATA_DIR / dataset / entry_id)


if __name__ == "__main__":
    start_time = time.time()
    RAW_DATA_DIR = Path("data/raw")
    PROCESSED_DATA_DIR = Path("data/processed")

    df = get_path_df(RAW_DATA_DIR)
    df.to_csv(RAW_DATA_DIR / "paths.csv", index=False)

    df = pd.read_csv(RAW_DATA_DIR / "paths.csv")

    df.reset_index(drop=True)

    # Dataset Transforms
    truncate_transform = transforms.TruncateProtein(max_length=1500, random=False)
    protein_view_transform = transforms.ProteinView(view="residue")
    transform = transforms.Compose([truncate_transform, protein_view_transform])

    preprocessing_dataset = PreProcessingDataset(
        path_df=df,
        transform=transform,
        pdb_dir=RAW_DATA_DIR,
        atom_feature=["default", "position"],
        bond_feature=None,
        residue_feature=["default", "symbol"],
    )

    # Batch Graph Transforms
    graph_construction_model = GraphConstruction(
        node_layers=[geometry.AlphaCarbonNode()],
        edge_layers=[
            geometry.SpatialEdge(radius=10.0, min_distance=5),
            geometry.KNNEdge(k=10, min_distance=5),
            geometry.SequentialEdge(max_distance=2),
        ],
    )

    preprocessor = PreProcessor(
        preprocessing_dataset,
        batch_size=64,
        num_workers=16,
        output_dir=PROCESSED_DATA_DIR,
        verbose=True,
    )

    preprocessor.process_data(graph_transform=graph_construction_model)

    end_time = time.time()
    print(f"\nTOTAL TIME: {end_time - start_time}")
