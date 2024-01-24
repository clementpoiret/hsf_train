import logging
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import List

import hydra
import lightning as L
import torch
import wandb
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from rich.logging import RichHandler

from hsftrain.data.loader import load_from_config
from hsftrain.models.losses import FocalTversky_loss
from hsftrain.models.models import SegmentationModel

# from hydra import compose, initialize

# initialize(config_path="conf")
# cfg = compose(config_name="config")

if __name__ == "__main__":
    print("Hello World!")
