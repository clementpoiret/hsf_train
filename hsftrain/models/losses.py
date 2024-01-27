"""
get_tp_fp_fn, SoftDiceLoss, and DC_and_CE/TopK_loss are from https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/training/loss_functions
"""

import torch
from torch import nn
from torch.autograd import Variable
from torch import einsum
import numpy as np


def get_tp_fp_fn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes:
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot

    if mask is not None:
        tp = torch.stack(tuple(
            x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)),
                         dim=1)
        fp = torch.stack(tuple(
            x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)),
                         dim=1)
        fn = torch.stack(tuple(
            x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)),
                         dim=1)

    if square:
        tp = tp**2
        fp = fp**2
        fn = fn**2

    tp = sum_tensor(tp, axes, keepdim=False)
    fp = sum_tensor(fp, axes, keepdim=False)
    fn = sum_tensor(fn, axes, keepdim=False)

    return tp, fp, fn


class TverskyLoss(nn.Module):

    def __init__(self,
                 apply_nonlin=None,
                 batch_dice=False,
                 do_bg=True,
                 smooth=1.,
                 square=False):
        """
        paper: https://arxiv.org/pdf/1706.05721.pdf
        """
        super(TverskyLoss, self).__init__()

        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.alpha = 0.3
        self.beta = 0.7

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, self.square)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn +
                                        self.smooth)

        if not self.do_bg:
            if self.batch_dice:
                tversky = tversky[1:]
            else:
                tversky = tversky[:, 1:]
        tversky = tversky.mean()

        return -tversky


class FocalTversky_loss(nn.Module):
    """
    paper: https://arxiv.org/pdf/1810.07842.pdf
    author code: https://github.com/nabsabraham/focal-tversky-unet/blob/347d39117c24540400dfe80d106d2fb06d2b99e1/losses.py#L65
    """

    def __init__(self, tversky_kwargs, gamma=0.75):
        super(FocalTversky_loss, self).__init__()
        self.gamma = gamma
        self.tversky = TverskyLoss(**tversky_kwargs)

    def forward(self, net_output, target, loss_mask=None):
        tversky_loss = 1 + self.tversky(
            net_output, target, loss_mask)  # = 1-tversky(net_output, target)
        focal_tversky = torch.pow(tversky_loss, self.gamma)
        return focal_tversky


def forgiving_loss(loss, input, target, ca_type, head=-1, tail=-2):
    # mask = torch.where((target == head) | (target == tail),
    #                    torch.tensor(0., device=input.device),
    #                    torch.tensor(1., device=input.device))
    # target *= mask.long()  # when target contains HEAD and TAIL channels
    if head > 0:
        # save where is head
        headmask = target[:, head:head + 1, :, :, :]
        # print(
        #     f"HEAD: {head}, TARGET: {target.shape}, HEADMASK: {headmask.shape}")
        #exclude head from target
        _pre = target[:, :head, :, :, :]
        _post = target[:, head + 1:, :, :, :]
        target = torch.cat([_pre, _post], dim=1)
        if tail > 0:
            tail -= 1
        # all positive classes are head
        if headmask.shape[1] > 0:
            target[:, 1:, :, :, :] += headmask

    if tail > 0:
        # save where is tail
        tailmask = target[:, tail:tail + 1, :, :, :]
        # print(
        #     f"TAIL: {tail}, TARGET: {target.shape}, HEADMASK: {tailmask.shape}, INPUT: {input.shape}"
        # )
        #exclude tail from target
        _pre = target[:, :tail, :, :, :]
        _post = target[:, tail + 1:, :, :, :]
        target = torch.cat([_pre, _post], dim=1)
        # all positive classes are tail
        if tailmask.shape[1] > 0:
            #CA1
            target[:, 2:3, :, :, :] += tailmask
            #DG
            # target[:, 1:2, :, :, :] += tailmask
            #SUB
            target[:, -1:, :, :, :] += tailmask

    if ca_type == "1/2/3":
        # 1 DG; 2 CA1; 3 CA2; 4 CA3; 5 SUB
        input_compat = input
    elif ca_type == "1/23":
        # 1 DG; 2 CA1; 3 CA2/3; 4 SUB
        _pre = input[:, :3, :, :, :]
        _in = input[:, 3:5, :, :, :].sum(1, keepdim=True)
        _post = input[:, 5:, :, :, :]

        input_compat = torch.cat((_pre, _in, _post), dim=1)
    elif ca_type == "123":
        # 1 DG; 2 CA1/2/3; 3 SUB
        _pre = input[:, :2, :, :, :]
        _in = input[:, 2:5, :, :, :].sum(1, keepdim=True)
        _post = input[:, 5:, :, :, :]

        input_compat = torch.cat((_pre, _in, _post), dim=1)

    # if head > 0:
    #     # 1DG 2-NCA N+1HEAD;]

    #     _pre = target[:, :head, :, :, :]
    #     _post = target[:, head + 1:, :, :, :]
    #     target = torch.cat((_pre, _post), dim=1)

    # if tail > 0:
    #     torch.where(target[:, tail, :, :, :] == 1)
    #     target = target[:, :tail, :, :, :]

    # print("input_compat", input_compat.shape)
    # print("target", target.shape)
    # print("mask", mask.shape)
    assert input_compat.shape == target.shape, f"Can't match input of shape {input_compat.shape} with a target of shape {target.shape}"

    return loss(input_compat.to(input.device), target)
