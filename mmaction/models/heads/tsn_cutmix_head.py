# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

import torch
from mmaction.evaluation import top_k_accuracy
from mmaction.registry import MODELS
from mmaction.utils import ConfigType, SampleList, get_str_type
from mmengine.model.weight_init import normal_init
from torch import Tensor, nn

from .base import AvgConsensus, BaseHead


@MODELS.register_module()
class TSNCutMixHead(BaseHead):
    """Class head for TSN.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict or ConfigDict): Config for building loss.
            Default: dict(type='CrossEntropyLoss').
        spatial_type (str or ConfigDict): Pooling type in spatial dimension.
            Default: 'avg'.
        consensus (dict): Consensus config dict.
        dropout_ratio (float): Probability of dropout layer. Default: 0.4.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 loss_cls: ConfigType = dict(type='CrossEntropyLoss'),
                 spatial_type: str = 'avg',
                 consensus: ConfigType = dict(type='AvgConsensus', dim=1),
                 dropout_ratio: float = 0.4,
                 init_std: float = 0.01,
                 **kwargs) -> None:
        super().__init__(num_classes, in_channels, loss_cls=loss_cls, **kwargs)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std

        consensus_ = consensus.copy()

        consensus_type = consensus_.pop('type')
        if get_str_type(consensus_type) == 'AvgConsensus':
            self.consensus = AvgConsensus(**consensus_)
        else:
            self.consensus = None

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool2d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.avg_pool = None

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

    def init_weights(self) -> None:
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x: Tensor, num_segs: int, **kwargs) -> Tensor:
        """Defines the computation performed at every call.

        Args:
            x (Tensor): The input data.
            num_segs (int): Number of segments into which a video
                is divided.
        Returns:
            Tensor: The classification scores for input samples.
        """
        # [N * num_segs, in_channels, 7, 7]
        if self.avg_pool is not None:
            if isinstance(x, tuple):
                shapes = [y.shape for y in x]
                assert 1 == 0, f'x is tuple {shapes}'
            x = self.avg_pool(x)
            # [N * num_segs, in_channels, 1, 1]
        x = x.reshape((-1, num_segs) + x.shape[1:])
        # [N, num_segs, in_channels, 1, 1]
        x = self.consensus(x)
        # [N, 1, in_channels, 1, 1]
        x = x.squeeze(1)
        # [N, in_channels, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)
            # [N, in_channels, 1, 1]
        x = x.view(x.size(0), -1)
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

            # ActorCutMix paper, page 6, equation 7
            # https://arxiv.org/pdf/2103.16565
            mask_ratios_compensated = 1 - torch.pow(
                torch.abs(1 - mask_ratios), self.label_mix_alpha
            )

            loss_mix = (
                self.loss_cls(cls_scores_mix, action_labels_mix)
                * mask_ratios_compensated
            ) + (
                self.loss_cls(cls_scores_mix, scene_labels_mix)
                * (1.0 - mask_ratios_compensated)
            )

            loss_list.append(loss_mix)

        loss_cls = torch.mean(torch.cat(loss_list))

        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses["loss_cls"] = loss_cls

        return losses