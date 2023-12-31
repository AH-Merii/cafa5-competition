import pandas as pd
from pathlib import Path
import re

import os
import time
import logging
import argparse

import yaml
import jinja2
from jinja2 import meta
import easydict

import torch
from torch import distributed as dist
from torch.optim import lr_scheduler

from torchdrug import core, utils, models
from torchdrug.utils import comm

from train import WeightedMultipleBinaryClassification


logger = logging.getLogger(__file__)


def get_root_logger(file=True):
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)
    format = logging.Formatter("%(asctime)-10s %(message)s", "%H:%M:%S")

    if file:
        handler = logging.FileHandler("log.txt")
        handler.setFormatter(format)
        logger.addHandler(handler)

    return logger


def create_working_directory(cfg):
    file_name = "working_dir.tmp"
    world_size = comm.get_world_size()
    if world_size > 1 and not dist.is_initialized():
        comm.init_process_group("nccl", init_method="env://")

    working_dir = os.path.join(
        os.path.expanduser(cfg.output_dir),
        cfg.task["class"],
        cfg.dataset["class"],
        cfg.task.model["class"],
        time.strftime("%Y-%m-%d-%H-%M-%S"),
    )

    # synchronize working directory
    if comm.get_rank() == 0:
        with open(file_name, "w") as fout:
            fout.write(working_dir)
        os.makedirs(working_dir)
    comm.synchronize()
    if comm.get_rank() != 0:
        with open(file_name, "r") as fin:
            working_dir = fin.read()
    comm.synchronize()
    if comm.get_rank() == 0:
        os.remove(file_name)

    os.chdir(working_dir)
    return working_dir


def detect_variables(cfg_file):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    env = jinja2.Environment()
    ast = env.parse(raw)
    vars = meta.find_undeclared_variables(ast)
    return vars


def load_config(cfg_file, context=None):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    template = jinja2.Template(raw)
    instance = template.render(context)
    cfg = yaml.safe_load(instance)
    cfg = easydict.EasyDict(cfg)
    return cfg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="yaml configuration file", required=True)
    parser.add_argument(
        "-s", "--seed", help="random seed for PyTorch", type=int, default=1024
    )
    # parser.add_argument(
    #     "-o",
    #     "--ont",
    #     help="subontology to train model on ('MFO','BPO','CCO')",
    #     type=str,
    #     default="MFO",
    # )

    args, unparsed = parser.parse_known_args()
    # get dynamic arguments defined in the config file
    vars = detect_variables(args.config)
    parser = argparse.ArgumentParser()
    for var in vars:
        parser.add_argument(f"--{var}", default="null")
    vars = parser.parse_known_args(unparsed)[0]
    vars = {k: utils.literal_eval(v) for k, v in vars._get_kwargs()}

    return args, vars


def build_downstream_solver(cfg, dataset):
    train_set, valid_set, test_set = dataset.split()
    if comm.get_rank() == 0:
        logger.warning(dataset)
        logger.warning(f"#train: {train_set}, #valid: {valid_set}, #test: {test_set}")

    cfg.task.task = [_ for _ in range(len(dataset.targets))]

    model = models.GearNet(**cfg.models)

    task = WeightedMultipleBinaryClassification(
        model=model,
        **cfg.task,
        verbose=1,
    )

    cfg.optimizer.params = task.parameters()
    optimizer = core.Configurable.load_config_dict(cfg.optimizer)

    if "scheduler" not in cfg:
        scheduler = None
    elif cfg.scheduler["class"] == "ReduceLROnPlateau":
        cfg.scheduler.pop("class")
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, **cfg.scheduler)
    else:
        cfg.scheduler.optimizer = optimizer
        scheduler = core.Configurable.load_config_dict(cfg.scheduler)
        cfg.engine.scheduler = scheduler

    solver = core.Engine(task, train_set, valid_set, test_set, optimizer, **cfg.engine)

    if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, **cfg.scheduler)
    elif scheduler is not None:
        cfg.scheduler.optimizer = optimizer
        scheduler = core.Configurable.load_config_dict(cfg.scheduler)
        solver.scheduler = scheduler

    if cfg.get("checkpoint") is not None:
        solver.load(cfg.checkpoint)

    if cfg.get("model_checkpoint") is not None:
        if comm.get_rank() == 0:
            logger.warning(f"Load checkpoint from {cfg.model_checkpoint}")
        cfg.model_checkpoint = os.path.expanduser(cfg.model_checkpoint)
        model_dict = torch.load(cfg.model_checkpoint, map_location=torch.device("cpu"))
        task.model.load_state_dict(model_dict)

    return solver, scheduler


def log_error_to_file(error_message, log_file="errors.log"):
    """Append error messages to a log file."""
    with open(log_file, "a") as file:
        file.write(error_message + "\n")


def extract_pdb_info(s, processed=False):
    if processed:
        pattern = r"^(?P<EntryID>[^-]+)\.pkl$"
    else:
        pattern = r"^(?P<database>[^-]+)-(?P<EntryID>[^-]+)-(?P<misc>.+)\.pdb$"
    match = re.match(pattern, s)
    if match:
        return match.groupdict()
    else:
        return None


def get_path_df(data_dir: Path, processed_dir: Path = None, **kwargs) -> pd.DataFrame:
    """
    Create a dataframe containing paths to the PDB files in the specified directory.

    This function traverses through the `data_dir` to find PDB files and returns
    a dataframe with details about each file. If `processed_dir` is provided,
    the function will skip PDB files for which corresponding pickle files already
    exist in the `processed_dir`.

    Parameters:
    - data_dir (Path): The directory containing datasets with PDB files.
    - processed_dir (Path, optional): The directory containing processed pickle files.
                                      If provided, PDB files with corresponding pickle
                                      files in this directory will be skipped.

    Returns:
    - DataFrame: A pandas dataframe with columns ["database", "EntryID", "misc", "path", "set"],
                 where "path" is the full path to the PDB file, and "set" is the dataset name.
    """

    path_list = []
    for dataset in data_dir.iterdir():
        if dataset.is_dir():
            for file in dataset.iterdir():
                d = extract_pdb_info(file.name, **kwargs)

                # If processed_dir is provided, check for existing pickle files and skip if found
                if processed_dir:
                    pickle_path = processed_dir / dataset.name / (d["EntryID"] + ".pkl")
                    if pickle_path.exists():
                        continue

                d["path"] = str(file)
                d["set"] = dataset.name
                path_list.append(d)

    return pd.DataFrame(path_list)
