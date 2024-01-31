import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from hsftrain.models.losses import FocalTversky_loss
from hsftrain.models.models import SegmentationModel

# from hydra import compose, initialize

# initialize(config_path="conf")
# cfg = compose(config_name="config")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    ckpt = "/mnt/hdd/models/hsf/arunet_3.0.0_1706633574.ckpt"

    seg_loss = FocalTversky_loss({"apply_nonlin": None})

    hparams = cfg.models.hparams

    model = SegmentationModel.load_from_checkpoint(ckpt,
                                                   hparams=hparams,
                                                   learning_rate=1e-2,
                                                   seg_loss=seg_loss)

    # To CPU as float32 model
    model = model.to("cpu").float()
    model.eval()
    dummy_input = torch.randn(1, 1, 16, 16, 16)

    # Export the model
    fname = ckpt.split("/")[-1].replace(".ckpt", ".onnx")
    torch.onnx.export(model,
                      dummy_input,
                      fname,
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
