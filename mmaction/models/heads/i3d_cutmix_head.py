# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

import torch
from mmaction.evaluation import top_k_accuracy
from mmaction.registry import MODELS
from mmaction.utils import ConfigType, SampleList
from mmengine.model.weight_init import normal_init
from torch import Tensor, nn

from .base import BaseHead


@MODELS.register_module()
class I3DCutMixHead(BaseHead):
    """Classification head for I3D.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict or ConfigDict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        loss_cls: ConfigType = dict(type="CrossEntropyLoss"),
        spatial_type: str = "avg",
        dropout_ratio: float = 0.5,
        init_std: float = 0.01,
        **kwargs,
    ) -> None:
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

        if self.spatial_type == "avg":
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None

    def init_weights(self) -> None:
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """Defines the computation performed at every call.

        Args:
            x (Tensor): The input data.

        Returns:
            Tensor: The classification scores for input samples.
        """
        # [N, in_channels, 4, 7, 7]
        if self.avg_pool is not None:
            x = self.avg_pool(x)

        # [N, in_channels, 1, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)

        # [N, in_channels, 1, 1, 1]
        x = x.view(x.shape[0], -1)

        # [N, in_channels]
        cls_score = self.fc_cls(x)

        # [N, num_classes]
        return cls_score

    def loss_by_feat(self, cls_scores: torch.Tensor, data_samples: SampleList) -> Dict:
        """Calculate the loss based on the features extracted by the head.

        Args:
            cls_scores (torch.Tensor): Classification prediction results of
                all class, has shape (batch_size, num_classes).
            data_samples (list[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of loss components.
        """

        labels = [x.gt_label for x in data_samples]
        labels = torch.stack(labels).to(cls_scores.device)
        labels = labels.squeeze()
        losses = dict()

        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
        elif (
            labels.dim() == 1
            and labels.size()[0] == self.num_classes
            and cls_scores.size()[0] == 1
        ):
            # Fix a bug when training with soft labels and batch size is 1.
            # When using soft labels, `labels` and `cls_score` share the same
            # shape.
            labels = labels.unsqueeze(0)

        if cls_scores.size() != labels.size():
            top_k_acc = top_k_accuracy(
                cls_scores.detach().cpu().numpy(),
                labels.detach().cpu().numpy(),
                self.topk,
            )

            for k, a in zip(self.topk, top_k_acc):
                losses[f"top{k}_acc"] = torch.tensor(a, device=cls_scores.device)

        if self.label_smooth_eps != 0:
            if cls_scores.size() != labels.size():
                labels = F.one_hot(labels, num_classes=self.num_classes)

            labels = (
                1 - self.label_smooth_eps
            ) * labels + self.label_smooth_eps / self.num_classes

        # Without CutMix label mixing
        # loss_cls = self.loss_cls(cls_scores, labels)

        # With CutMix label mixing
        mix_idx = torch.tensor([hasattr(o, "scene_label") for o in data_samples])
        data_samples_mix = [d for d in data_samples if hasattr(d, "scene_label")]

        cls_scores_orig = cls_scores[~mix_idx]
        cls_scores_mix = cls_scores[mix_idx]
        loss_list = []

        if len(cls_scores_orig) > 0:
            labels_orig = labels[~mix_idx].type(torch.LongTensor).to(cls_scores.device)
            loss_orig = self.loss_cls(cls_scores_orig, labels_orig).unsqueeze(0)

            loss_list.append(loss_orig)

        if len(cls_scores_mix) > 0:
            action_labels_mix = (
                labels[mix_idx].type(torch.LongTensor).to(cls_scores.device)
            )

            scene_labels_mix = torch.tensor(
                [d.scene_label for d in data_samples_mix],
                device=cls_scores.device,
            )

            mask_ratios = torch.tensor(
                [d.mask_ratio for d in data_samples_mix],
                device=cls_scores.device,
            )

            loss_mix = (
                self.loss_cls(cls_scores_mix, action_labels_mix)
            ) * mask_ratios + self.loss_cls(cls_scores_mix, scene_labels_mix) * (
                1.0 - mask_ratios
            )

            loss_list.append(loss_mix)

        loss_cls = torch.mean(torch.cat(loss_list))

        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses["loss_cls"] = loss_cls

        return losses
