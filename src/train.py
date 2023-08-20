from torchdrug import models, core, layers
from dataset import GeneOntology, split_dataset
import pandas as pd
from pathlib import Path
import torch
from torchdrug.tasks import MultipleBinaryClassification
import json


class WeightedMultipleBinaryClassification(MultipleBinaryClassification):
    """
    Weighted Multiple Binary Classification task for graphs/molecules/proteins.

    This class extends the MultipleBinaryClassification by introducing a custom weight initialization
    based on a provided GO weights file. This allows for the model to consider the importance of certain
    classes over others during training.

    Parameters:
        weights_path (str): path to the GO weights file.
                            The file should have each line in the format: "GO_term\tweight"
        model (nn.Module): graph representation model
        task (list of int, optional): training task id(s).
        criterion (list or dict, optional): training criterion(s).
            For dict, the keys are criterions and the values are the corresponding weights.
            Available criterions are ``bce``.
        metric (str or list of str, optional): metric(s).
            Available metrics are ``auroc@macro``, ``auprc@macro``, ``auroc@micro``, ``auprc@micro`` and ``f1_max``.
        num_mlp_layer (int, optional): number of layers in the MLP prediction head
        normalization (bool, optional): whether to normalize the target
        reweight (bool, optional): whether to re-weight tasks according to the number of positive samples
        graph_construction_model (nn.Module, optional): graph construction model
        verbose (int, optional): output verbose level
    """

    def __init__(self, weights_path, *args, **kwargs):
        super(WeightedMultipleBinaryClassification, self).__init__(*args, **kwargs)

        # Load the GO weights
        self.go_weights = {
            line.split("\t")[0]: float(line.split("\t")[1].strip())
            for line in open(weights_path)
        }

    def preprocess(self, train_set, valid_set, test_set):
        """
        Compute the weight for each task on the training set.
        """
        # Get weights for the tasks from the GO weights dictionary
        task_weights = [self.go_weights.get(task, 1.0) for task in self.task]
        task_weights = torch.tensor(task_weights, dtype=torch.float32)

        # Check if reweighting is required
        if self.reweight:
            values = []
            for data in train_set:
                values.append(data["targets"][self.task_indices])
            values = torch.stack(values, dim=0)

            num_positive = values.sum(dim=0)
            weight = (num_positive.mean() / num_positive).clamp(1, 10)
            task_weights *= weight

        self.register_buffer("weight", task_weights)


if __name__ == "__main__":
    PROCESSED_DATA_DIR = Path("data/processed")
    # path_df = get_path_df(data_dir=PROCESSED_DATA_DIR, processed=True)

    labeled_path_df = pd.read_csv("data/labeled_paths.csv")
    label_df = pd.read_csv("data/terms.tsv", sep="\t")
    # pretrained_model = "data/pretrained_models/mc-gearnet-edge.pth"
    pretrained_model = "data/pretrained_models/attr_gearnet_edge.pth"

    # bp_go = GeneOntology(
    #     path_df=labeled_path_df,
    #     label_df=label_df,
    #     subontology="BPO",
    # )
    mf_go = GeneOntology(
        path_df=labeled_path_df,
        label_df=label_df,
        subontology="MFO",
    )
    # cc_go = GeneOntology(
    #     path_df=labeled_path_df,
    #     label_df=label_df,
    #     subontology="CCO",
    # )

    train, test, valid = split_dataset(mf_go)


    gearnet_edge = models.GearNet(
        input_dim=70,
        hidden_dims=[512, 512, 512],
        num_relation=7,
        edge_input_dim=40,
        num_angle_bin=8,
        batch_norm=True,
        concat_hidden=True,
        short_cut=True,
        readout="sum",
    )

    graph_construction_model = layers.GraphConstruction(edge_feature="gearnet")

    task = WeightedMultipleBinaryClassification(
        weights_path="data/IA.txt",
        model=gearnet_edge,
        num_mlp_layer=3,
        task=[_ for _ in range(len(mf_go.targets))],
        criterion="bce",
        metric=["auprc@micro", "f1_max"],
        verbose=1,
    )

    optimizer = torch.optim.AdamW(task.parameters(), lr=1e-4, weight_decay=0)
    solver = core.Engine(
        task,
        train,
        valid,
        test,
        optimizer,
        gpus=[0],
        batch_size=4,
        num_worker=16,
        log_interval=10,
    )

    checkpoint = torch.load(pretrained_model)["model"]
    checkpoint = {k: v for k, v in checkpoint.items() if not k.startswith("mlp")}
    task.load_state_dict(checkpoint, strict=False)

    solver.train(num_epoch=10)
    solver.evaluate("valid")

    with open("data/finetuned_models/mf_gearnet.json", "w") as fout:
        json.dump(solver.config_dict(), fout)

    solver.save("data/finetuned_models/mf_gearnet.pth")
