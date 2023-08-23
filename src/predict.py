import os
import pickle

import torch
from torch.utils import Dataset
from torchdrug import models, utils
from torchdrug.data import DataLoader


class GeneOntology(Dataset):
    def __init__(
        self,
        path_df,
        subontology="BPO",
    ):
        self.subontology = subontology
        self.path_df = path_df[path_df[self.subontology] == 1].reset_index(drop=True)

        # Get all go labels for subontology
        self.tasks = self.label_df["term"].unique()

        self.go_term_to_idx = self.go_term_index_mapping(self.tasks)

    def __getitem__(self, idx):
        path = self.path_df.at[idx, "path"]
        item = {
            "graph": self.read_pickle(path),
            "EntryID": self.path_df.at[idx, "EntryID"],
        }

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
        return {idx: go_term for idx, go_term in enumerate(go_terms_list)}


class InferenceEngine:
    def __init__(
        self,
        dataset,
        model,  # Instance of the GeometryAwareRelationalGraphNeuralNetwork or its subclass
        checkpoint,
        device=torch.device("cpu"),
        batch_size=1,
    ):
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
        )

        self.model = model
        self.device = device
        self.load(checkpoint)

    def load(self, checkpoint, strict=True):
        """
        Load a checkpoint from file.
        """
        checkpoint = os.path.expanduser(checkpoint)
        state = torch.load(checkpoint, map_location=self.device)
        self.model.load_state_dict(state["model"], strict=strict)

    def predict(self, data_loader=None, device=None):
        """
        Predict the outputs for the input data in the desired format.
        """

        # If no specific data_loader is provided, use the one from the engine
        if data_loader is None:
            data_loader = self.loader

        # Set the model to evaluation mode
        self.model.eval()

        # If no device is provided, use the device of the model
        if device is None:
            device = self.device

        # List to store all formatted predictions
        formatted_preds = []

        # No need to track gradients during prediction
        with torch.no_grad():
            for batch in data_loader:
                # Move batch to the specified device
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                # Get predictions
                graph = batch["graph"]
                output = self.model(graph, graph.node_feature.float())
                preds = output[
                    "graph_feature"
                ]  # Assuming this is the tensor containing predictions

                # Convert tensor predictions to the desired format
                batch_formatted_preds = self.tensor_to_formatted_preds(
                    preds, batch["EntryID"]
                )
                formatted_preds.extend(batch_formatted_preds)

        return formatted_preds

    def tensor_to_formatted_preds(self, tensor, protein_ids):
        """
        Convert tensor predictions to the desired string format.
        """
        # TODO:  format the tensor predictions
        pass


def tensor_to_formatted_preds(self, tensor, protein_ids):
    """
    Convert tensor predictions to the desired format.

    Args:
    - tensor (torch.Tensor): Tensor predictions.
    - protein_ids (list[str]): List of protein IDs associated with the tensor predictions.

    Returns:
    - list[str]: List of strings containing predictions in the desired format.
    """

    # Inverse mapping from index to GO term
    idx_to_go_term = {
        idx: go_term for go_term, idx in self.train_set.go_term_to_idx.items()
    }

    # Convert tensor to scores and get scores greater than 0
    scores = tensor.tolist()
    formatted_preds = []

    for protein_id, score_vector in zip(protein_ids, scores):
        non_zero_scores = [
            (idx_to_go_term[i], score)
            for i, score in enumerate(score_vector)
            if 0 < score <= 1
        ]

        # Sort by score and keep the top 1500 predictions
        sorted_scores = sorted(non_zero_scores, key=lambda x: x[1], reverse=True)[:1500]

        for go_term, score in sorted_scores:
            formatted_preds.append(f"{protein_id}\t{go_term}\t{round(score, 3)}")

    return formatted_preds


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
