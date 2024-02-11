import lightning as L
import pymia.evaluation.evaluator as eval_
import pymia.evaluation.metric as metric
import torch
import torch.nn as nn

from hsftrain.models.blocks import (Decoder, DoubleConv, Encoder,
                                    ExtResNetBlock, SingleConv)
from hsftrain.models.losses import FocalTversky_loss, forgiving_loss
from hsftrain.models.optimizer import AdamP
from hsftrain.models.scheduler import LinearWarmupCosineAnnealingLR
from hsftrain.models.helpers import number_of_features_per_level


class Abstract3DUNet(nn.Module):
    """
    Base class for standard and residual UNet.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the
            final 1x1 convolution, otherwise apply nn.Softmax. MUST be True if nn.BCELoss (two-class) is used
            to train the model. MUST be False if nn.CrossEntropyLoss (multi-class) is used to train the model.
        basic_module: basic model for the encoder/decoder (DoubleConv, ExtResNetBlock, ....)
        layer_order (string): determines the order of layers
            in `SingleConv` module. e.g. 'crg' stands for Conv3d+ReLU+GroupNorm3d.
            See `SingleConv` for more info
        f_maps (int, tuple): if int: number of feature maps in the first conv layer of the encoder (default: 64);
            if tuple: number of feature maps at each level
        num_levels (int): number of levels in the encoder/decoder path (applied only if f_maps is an int)
        is_segmentation (bool): if True (semantic segmentation problem) Sigmoid/Softmax normalization is applied
            after the final convolution; if False (regression problem) the normalization layer is skipped at the end
        testing (bool): if True (testing mode) the `final_activation` (if present, i.e. `is_segmentation=true`)
            will be applied as the last operation during the forward pass; if False the model is in training mode
            and the `final_activation` (even if present) won't be applied; default: False
        conv_kernel_size (int or tuple): size of the convolving kernel in the basic_module
        pool_kernel_size (int or tuple): the size of the window
        conv_padding (int or tuple): add zero-padding added to all three sides of the input
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 final_sigmoid,
                 basic_module,
                 f_maps=64,
                 layer_order='gcr',
                 num_levels=4,
                 is_segmentation=True,
                 testing=True,
                 conv_kernel_size=3,
                 pool_kernel_size=2,
                 conv_padding=1,
                 use_attention=False,
                 normalization="s",
                 using_bn=False,
                 **kwargs):
        super(Abstract3DUNet, self).__init__()

        self.testing = testing

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        # create encoder path consisting of Encoder modules. Depth of the encoder is equal to `len(f_maps)`
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(
                    in_channels,
                    out_feature_num,
                    apply_pooling=False,  # skip pooling in the firs encoder
                    basic_module=basic_module,
                    conv_layer_order=layer_order,
                    conv_kernel_size=conv_kernel_size,
                    padding=conv_padding)
            else:
                # TODO: adapt for anisotropy in the data, i.e. use proper pooling kernel to make the data isotropic after 1-2 pooling operations
                encoder = Encoder(f_maps[i - 1],
                                  out_feature_num,
                                  basic_module=basic_module,
                                  conv_layer_order=layer_order,
                                  conv_kernel_size=conv_kernel_size,
                                  pool_kernel_size=pool_kernel_size,
                                  padding=conv_padding)

            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            if basic_module == DoubleConv:
                in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            else:
                in_feature_num = reversed_f_maps[i]

            out_feature_num = reversed_f_maps[i + 1]
            # TODO: if non-standard pooling was used, make sure to use correct
            # striding for transpose conv
            # currently strides with a constant stride: (2, 2, 2)

            decoder = Decoder(in_feature_num,
                              out_feature_num,
                              basic_module=basic_module,
                              conv_layer_order=layer_order,
                              conv_kernel_size=conv_kernel_size,
                              padding=conv_padding,
                              use_attention=use_attention,
                              normalization=normalization,
                              using_bn=using_bn)
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

        if is_segmentation:
            # semantic segmentation problem
            if final_sigmoid:
                self.final_activation = nn.Sigmoid()
            else:
                self.final_activation = nn.Softmax(dim=1)
        else:
            # regression problem
            self.final_activation = None

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only during prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        if self.testing and self.final_activation is not None:
            x = self.final_activation(x)

        return x


class UNet3D(Abstract3DUNet):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.

    Uses `DoubleConv` as a basic_module and nearest neighbor upsampling in the decoder
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 final_sigmoid=False,
                 f_maps=64,
                 layer_order='gcr',
                 num_levels=4,
                 is_segmentation=True,
                 conv_padding=1,
                 **kwargs):
        super(UNet3D, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     final_sigmoid=final_sigmoid,
                                     basic_module=DoubleConv,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_levels=num_levels,
                                     is_segmentation=is_segmentation,
                                     conv_padding=conv_padding,
                                     **kwargs)


class ResidualUNet3D(Abstract3DUNet):
    """
    Residual 3DUnet model implementation based on https://arxiv.org/pdf/1706.00120.pdf.
    Uses ExtResNetBlock as a basic building block, summation joining instead
    of concatenation joining and transposed convolutions for upsampling (watch out for block artifacts).
    Since the model effectively becomes a residual net, in theory it allows for deeper UNet.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 final_sigmoid=False,
                 f_maps=64,
                 layer_order='gcr',
                 num_levels=5,
                 is_segmentation=True,
                 conv_padding=1,
                 **kwargs):
        super(ResidualUNet3D, self).__init__(in_channels=in_channels,
                                             out_channels=out_channels,
                                             final_sigmoid=final_sigmoid,
                                             basic_module=ExtResNetBlock,
                                             f_maps=f_maps,
                                             layer_order=layer_order,
                                             num_levels=num_levels,
                                             is_segmentation=is_segmentation,
                                             conv_padding=conv_padding,
                                             **kwargs)


class SegmentationModel(L.LightningModule):

    def __init__(self,
                 hparams,
                 seg_loss=FocalTversky_loss({"apply_nonlin": nn.Softmax(dim=1)
                                            }),
                 use_forgiving_loss=True,
                 optimizer=AdamP,
                 scheduler=LinearWarmupCosineAnnealingLR,
                 learning_rate=1e-3,
                 classes_names=None,
                 epochs=100,
                 steps_per_epoch=100,
                 precision="bf16-true",
                 finetuning_cfg=None):
        super(SegmentationModel, self).__init__()
        self.hp = hparams
        self.learning_rate = learning_rate
        self.seg_loss = seg_loss
        self.use_forgiving_loss = use_forgiving_loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.precision = precision
        self.finetuning_cfg = finetuning_cfg

        if classes_names:
            assert len(classes_names) == hparams['out_channels']
        self.classes_names = classes_names

        if hparams['modeltype'] == 'UNet3D':
            self._model = UNet3D(final_sigmoid=False, **hparams)

        elif hparams['modeltype'] == 'ResidualUNet3D':
            self._model = ResidualUNet3D(final_sigmoid=False, **hparams)

        else:
            raise ValueError(f"Unknown model: {hparams['modeltype']}")

        # Configure metrics
        self.metrics = [
            metric.DiceCoefficient(),
            metric.VolumeSimilarity(),
            metric.MahalanobisDistance()
        ]

        self.training_step_loss = torch.tensor([])
        self.validation_step_loss = torch.tensor([])

    def prepare_finetuning(self):
        if self.finetuning_cfg.depth != -1:
            if self.finetuning_cfg.mode == "encoder":
                assert self.finetuning_cfg.depth <= len(
                    self._model.encoders
                ), "Invalid depth for encoder finetuning"
            if self.finetuning_cfg.mode == "decoder":
                assert self.finetuning_cfg.depth <= len(
                    self._model.decoders
                ), "Invalid depth for decoder finetuning"

        if self.finetuning_cfg.out_channels != self.hp.out_channels:
            print(
                "Different number of classes detected. Replacing the last layer."
            )
            self._model.final_conv = nn.Conv3d(64,
                                               self.finetuning_cfg.out_channels,
                                               1)

        # Freeze all layers except the last one
        print("Freezing all layers except the last one")
        for name, param in self._model.named_parameters():
            if "final_conv" not in name:
                param.requires_grad = False

    def _adjust_finetuning(self, epoch):
        if epoch < self.finetuning_cfg.unfreeze_frequency:
            return

        if epoch % self.finetuning_cfg.unfreeze_frequency == 1:
            current_depth = epoch // self.finetuning_cfg.unfreeze_frequency

            if self.finetuning_cfg.mode == "encoder":
                max_depth = len(
                    self._model.encoders
                ) if self.finetuning_cfg.depth == -1 else self.finetuning_cfg.depth

                for i, encoder in enumerate(self._model.encoders):
                    if (i < current_depth) and (i <= max_depth):
                        print(f"Unfreezing encoder {i}")
                        for param in encoder.parameters():
                            param.requires_grad = True
                    else:
                        for param in encoder.parameters():
                            param.requires_grad = False

            if self.finetuning_cfg.mode == "decoder":
                max_depth = len(
                    self._model.decoders
                ) if self.finetuning_cfg.depth == -1 else self.finetuning_cfg.depth

                # Decoders are in reverse order
                for i, decoder in enumerate(reversed(self._model.decoders)):
                    if (i < current_depth) and (i <= max_depth):
                        print(f"Unfreezing decoder {i}")
                        for param in decoder.parameters():
                            param.requires_grad = True
                    else:
                        for param in decoder.parameters():
                            param.requires_grad = False

    def forward(self, x):
        return self._model(x)

    def configure_optimizers(self):
        num_iterations = self.epochs * self.steps_per_epoch

        optimizer = self.optimizer(self._model.parameters(),
                                   lr=self.learning_rate,
                                   epochs=self.epochs,
                                   steps_per_epoch=self.steps_per_epoch,
                                   betas=[0.9, 0.999],
                                   weight_decay=0,
                                   weight_decouple=True,
                                   fixed_decay=False,
                                   delta=0.1,
                                   wd_ratio=0.1,
                                   use_gc=False,
                                   nesterov=False,
                                   r=0.95,
                                   adanorm=False,
                                   adam_debias=False,
                                   demon=True,
                                   agc_clipping_value=1e-2,
                                   agc_eps=1e-3,
                                   eps=1e-8)
        scheduler = self.scheduler(
            optimizer,
            warmup_epochs=int(0.1 * self.epochs) * self.steps_per_epoch,
            max_epochs=num_iterations,
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def step(self, batch, batch_idx, step_name="Training"):
        x, y = batch["mri"]["data"], batch["label"]["data"]
        if "bf16" in self.precision:
            x = x.bfloat16()
        else:
            x = x.float()

        _, labels = y.max(dim=1)

        names = {v[0]: k for k, v in batch["labels_names"].items()}
        head = names.get("HEAD", -1)
        tail = names.get("TAIL", -2)

        y_hat = self.forward(x)

        if self.use_forgiving_loss:
            return forgiving_loss(self.seg_loss,
                                  y_hat,
                                  y,
                                  batch["ca_type"][0],
                                  head=head,
                                  tail=tail)

        return self.seg_loss(y_hat, y)

    def on_train_epoch_start(self):
        if self.finetuning_cfg:
            self._adjust_finetuning(self.current_epoch)

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, step_name="Training")

        self.training_step_loss = torch.cat(
            (self.training_step_loss, loss.unsqueeze(0).cpu().detach()))
        self.log("train/batch/loss", loss, prog_bar=False)

        return loss

    def on_train_epoch_end(self):
        loss = self.training_step_loss.mean()
        self.log("train/epoch/loss", loss, prog_bar=True)
        self.training_step_loss = torch.tensor([])

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, step_name="Validation")

        self.validation_step_loss = torch.cat(
            (self.validation_step_loss, loss.unsqueeze(0).cpu().detach()))
        self.log("val/batch/loss", loss, prog_bar=False)

        return loss

    def on_validation_epoch_end(self):
        loss = self.validation_step_loss.mean()
        self.log("val/epoch/loss", loss, prog_bar=True)
        self.validation_step_loss = torch.tensor([])
