from torch.optim import lr_scheduler
import pandas as pd
import torch
import math
from torchdrug.tasks import MultipleBinaryClassification
from dataset import GeneOntology
from tqdm.auto import tqdm
import utils as util
import numpy as np
import os
from torchdrug.utils import comm
import random
import pprint


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


def train_and_validate(cfg, solver, scheduler):
    if cfg.train.num_epoch == 0:
        return

    step = math.ceil(cfg.train.num_epoch / 50)
    best_result = float("-inf")
    best_epoch = -1

    training_loop = tqdm(
        range(0, cfg.train.num_epoch, step),
        desc="Processing",
        disable=not cfg.verbose,
        colour="green",
    )
    for i in training_loop:
        kwargs = cfg.train.copy()
        kwargs["num_epoch"] = min(step, cfg.train.num_epoch - i)
        solver.train(**kwargs)
        solver.save(f"{cfg.checkpoints_dir}/model_epoch_{solver.epoch}.pth")
        metric = solver.evaluate("valid")
        solver.evaluate("test")
        result = metric[cfg.metric]
        if result > best_result:
            best_result = result
            best_epoch = solver.epoch
        if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            scheduler.step(result)

    solver.load(f"{cfg.checkpoints_dir}/model_epoch_{best_epoch}.pth")
    return solver


def test(cfg, solver):
    solver.evaluate("valid")
    return solver.evaluate("test")


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)

    # set system seed value
    seed = args.seed
    torch.manual_seed(seed + comm.get_rank())
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # configure logger
    logger = util.get_root_logger()
    if comm.get_rank() == 0:
        logger.warning(f"Config file: {args.config}")
        logger.warning(pprint.pformat(cfg))

    # build dataset
    dataset = GeneOntology(
        path_df=pd.read_csv(cfg.dataset.path_df),
        label_df=pd.read_csv(cfg.dataset.label_df, sep="\t"),
        subontology=cfg.dataset.subontology,
        sample=cfg.dataset.sample,
        split=cfg.dataset.split,
        frac=cfg.dataset.frac,
    )

    solver, scheduler = util.build_downstream_solver(cfg, dataset)

    train_and_validate(cfg, solver, scheduler)
    test(cfg, solver)

    print("\033[92m\033[1mCongratulations, Model has completed training!\033[0m")
