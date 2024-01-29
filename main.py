import logging
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import List

import hydra
import lightning as L
import torch
import torchio as tio
from dotenv import load_dotenv
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from rich.logging import RichHandler

import wandb
from hsftrain.data.loader import load_from_config
from hsftrain.models.losses import FocalTversky_loss
from hsftrain.models.models import SegmentationModel

VERSION = "2.0.0"
FORMAT = "%(message)s"
logging.basicConfig(level="INFO",
                    format=FORMAT,
                    datefmt="[%X]",
                    handlers=[RichHandler(markup=True)])
log = logging.getLogger(__name__)

load_dotenv()

# from hydra import compose, initialize

# initialize(config_path="conf")
# cfg = compose(config_name="config")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    dt = datetime.now()
    ts = int(datetime.timestamp(dt))
    name = f"arunet2_{ts}"
    log.info(f"Experiment name: {name}")

    mri_datamodule = load_from_config(cfg.datasets)(
        preprocessing_pipeline=tio.Compose([
            tio.ToCanonical(),
            tio.ZNormalization(),
        ]),
        augmentation_pipeline=tio.Compose([
            tio.RandomFlip(axes=('LR',), p=.5),
            tio.RandomMotion(degrees=5, translation=5, num_transforms=3, p=.1),
            tio.RandomBlur(std=(0, 0.5), p=.1),
            tio.RandomNoise(mean=0, std=0.5, p=.1),
            tio.RandomGamma(log_gamma=0.4, p=.1),
            tio.RandomAffine(scales=.3,
                             degrees=30,
                             translation=5,
                             isotropic=False,
                             p=.2),
            # tio.RandomAnisotropy(p=.1, scalars_only=False),
            tio.transforms.RandomElasticDeformation(num_control_points=4,
                                                    max_displacement=4,
                                                    locked_borders=0,
                                                    p=.1),
            # tio.RandomSpike(p=.01),
            # tio.RandomBiasField(coefficients=.2, p=.01),
        ]),
        postprocessing_pipeline=tio.Compose([
            tio.CopyAffine("mri"),
            tio.EnsureShapeMultiple(8),
            tio.OneHot(),
        ]))

    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb_logger = WandbLogger(name=name, project="arunet2")

    _dm = deepcopy(mri_datamodule)
    _dm.setup()
    N = len(_dm.subjects_train_list)
    N_val = len(_dm.subjects_val_list)
    log.info(f"Train dataset size: {N}")
    log.info(f"Validation dataset size: {N_val}")
    steps_per_epoch = N // cfg.datasets.batch_size
    steps_per_epoch = steps_per_epoch // cfg.lightning.accumulate_grad_batches

    wandb_logger.experiment.config.update({"train_size": N, "val_size": N_val})

    seg_loss = FocalTversky_loss({"apply_nonlin": None})

    hparams = cfg.models.hparams

    model = SegmentationModel(hparams=hparams,
                              seg_loss=seg_loss,
                              epochs=cfg.lightning.max_epochs,
                              steps_per_epoch=steps_per_epoch)

    trainer = L.Trainer(logger=wandb_logger,
                        callbacks=[
                            ModelCheckpoint(
                                monitor="val/epoch/loss",
                                mode="min",
                                save_top_k=1,
                                save_last=True,
                                verbose=True,
                                dirpath=f"{cfg.datasets.output_path}ckpt/",
                                filename=f"arunet_{VERSION}_{ts}"),
                        ],
                        **cfg.lightning)

    trainer.fit(model, datamodule=mri_datamodule)

    dummy_input = torch.randn(1, 1, 16, 16, 16)
    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        f"{cfg.datasets.output_path}onnx/arunet_{VERSION}_{ts}.onnx",
        input_names=["cropped_hippocampus"],
        output_names=["segmented_hippocampus"],
        dynamic_axes={
            'cropped_hippocampus': {
                0: 'batch',
                2: "x",
                3: "y",
                4: "z"
            },
            'segmented_hippocampus': {
                0: 'batch',
                2: "x",
                3: "y",
                4: "z"
            }
        },
        opset_version=17)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    L.seed_everything(42)
    main()
