from pathlib import Path
import time
from torch.utils.data import Dataset
from torchdrug.data import DataLoader, Protein
import pickle
from tqdm.auto import tqdm
from torchdrug import transforms
from torchdrug.layers import GraphConstruction, geometry
import pandas as pd
import warnings
from rdkit import RDLogger
import torch
from torchdrug import data
from collections.abc import Mapping, Sequence
from utils import get_path_df, log_error_to_file


RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")


def graph_collate(batch):
    """
    Convert any list of same nested container into a container of tensors.

    For instances of :class:`data.Graph <torchdrug.data.Graph>`, they are collated
    by :meth:`data.Graph.pack <torchdrug.data.Graph.pack>`.

    Parameters:
        batch (list): list of samples with the same nested container
    """

    # Filter out None values
    batch = [item for item in batch if item is not None]
    try:
        elem = batch[0]
        if isinstance(elem, torch.Tensor):
            out = None
            if torch.utils.data.get_worker_info() is not None:
                numel = sum([x.numel() for x in batch])
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            return torch.stack(batch, 0, out=out)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, (str, bytes)):
            return batch
        elif isinstance(elem, data.Graph):
            return elem.pack(batch)
        elif isinstance(elem, Mapping):
            return {key: graph_collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, Sequence):
            it = iter(batch)
            elem_size = len(next(it))
            if not all(len(elem) == elem_size for elem in it):
                raise RuntimeError(
                    "Each element in list of batch should be of equal size"
                )
            return [graph_collate(samples) for samples in zip(*batch)]

        raise TypeError("Can't collate data with type `%s`" % type(elem))

    except Exception as e:
        print(f"Error during collation: {e}")
        # Print details of each item in the batch
        for item in batch:
            print(item)
        # Reraise the exception to ensure it's caught upstream
        raise e


class PreProcessingDataset(Dataset):
    def __init__(
        self,
        path_df,
        pdb_dir,
        transform=None,
        atom_feature="pretrain",
        bond_feature="pretrain",
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
        id = self.path_df.at[idx, "EntryID"]
        dataset = self.path_df.at[idx, "set"]
        try:
            protein = Protein.from_pdb(
                pdb_path,
                atom_feature=self.atom_feature,
                bond_feature=self.bond_feature,
                residue_feature=self.residue_feature,
            )
            item = {"graph": protein, "id": id, "dataset": dataset}
            if self.transform:
                item = self.transform(item)
            return item
        except Exception as e:
            error_message = f"Error processing entry {id} in dataset {dataset} at path {pdb_path}: {e}"
            log_error_to_file(error_message)
            return None

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
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=graph_collate,
        )

    def pickle_item(self, protein, path):
        path.parent.mkdir(exist_ok=True)
        with open(f"{path}.pkl", "wb") as file:
            pickle.dump(protein, file)

    def process_data(self, graph_transform=False, graph_key="graph"):
        if self.verbose:
            self.loader = tqdm(self.loader, colour="green")

        for i, batch in enumerate(self.loader):
            if graph_transform:
                batch[graph_key] = graph_transform(batch[graph_key])

            for protein, id, subset in zip(
                batch["graph"], batch["id"], batch["dataset"]
            ):
                self.pickle_item(protein, Path(self.output_dir) / subset / id)


if __name__ == "__main__":
    start_time = time.time()
    RAW_DATA_DIR = Path("data/raw")
    PROCESSED_DATA_DIR = Path("data/processed")

    df = get_path_df(data_dir=RAW_DATA_DIR)

    df.to_csv(RAW_DATA_DIR / "pdb_paths.csv", index=False)

    df = pd.read_csv(RAW_DATA_DIR / "pdb_paths.csv")

    df.reset_index(drop=True)

    # Dataset Transforms
    truncate_transform = transforms.TruncateProtein(max_length=1500, random=False)
    protein_view_transform = transforms.ProteinView(view="residue")
    transform = transforms.Compose([truncate_transform, protein_view_transform])

    preprocessing_dataset = PreProcessingDataset(
        path_df=df,
        transform=transform,
        pdb_dir=RAW_DATA_DIR,
    )

    # Batch Graph Transforms
    graph_construction_model = GraphConstruction(
        node_layers=[geometry.AlphaCarbonNode()],
        edge_layers=[
            geometry.SpatialEdge(radius=10.0, min_distance=5),
            geometry.KNNEdge(k=10, min_distance=5),
            geometry.SequentialEdge(max_distance=2),
        ],
        edge_feature="gearnet",
    )

    preprocessor = PreProcessor(
        preprocessing_dataset,
        batch_size=32,
        num_workers=16,
        output_dir=PROCESSED_DATA_DIR,
        verbose=True,
    )

    preprocessor.process_data(graph_transform=graph_construction_model)

    end_time = time.time()
    print(f"\nTOTAL TIME: {end_time - start_time}")
