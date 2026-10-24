# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmdet.registry import MODELS


def _build_norm(channels: int) -> nn.GroupNorm:
    groups = min(32, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ConvNormAct(nn.Sequential):
    """Small conv-norm-act block used by lightweight modality stems."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
        self.norm = _build_norm(out_channels)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    """Basic residual block for SAR/topography/change stems."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = ConvNormAct(in_channels, out_channels, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            _build_norm(out_channels))
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                _build_norm(out_channels))
        else:
            self.downsample = nn.Identity()
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        return self.act(out + identity)


class LightweightModalityStem(nn.Module):
    """Lightweight encoder for auxiliary modalities.

    It produces three feature levels at strides 8, 16 and 32, matching the
    default Swin-B ``out_indices=(1, 2, 3)`` outputs used by PF-RPN.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: Sequence[int],
                 stem_channels: int = None):
        super().__init__()
        assert len(out_channels) == 3, 'Expected three feature levels.'
        stem_channels = stem_channels or max(out_channels[0] // 2, 32)
        self.stem = nn.Sequential(
            ConvNormAct(in_channels, stem_channels, stride=2),
            ConvNormAct(stem_channels, out_channels[0], stride=2))
        self.stage1 = ResidualBlock(
            out_channels[0], out_channels[0], stride=2)
        self.stage2 = ResidualBlock(
            out_channels[0], out_channels[1], stride=2)
        self.stage3 = ResidualBlock(
            out_channels[1], out_channels[2], stride=2)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        x = self.stem(x)
        feat0 = self.stage1(x)
        feat1 = self.stage2(feat0)
        feat2 = self.stage3(feat1)
        return feat0, feat1, feat2


@MODELS.register_module()
class GatedMultiModalFusion(nn.Module):
    """Spatial gated fusion with missing-modality masking.

    The fusion accepts one feature pyramid per modality and a per-sample
    modality mask.  Missing modalities receive zero gate probability and are
    excluded from the returned global modality tokens.

    Args:
        in_channels (Sequence[Sequence[int]]): Input channels for each
            modality and each feature level, shape ``(M, L)``.
        out_channels (Sequence[int]): Fused channel dimension per level.
        token_dim (int): Dimension of the global modality tokens.
        num_modalities (int): Number of modalities.
        dropout (float): Dropout probability for fused features.
    """

    def __init__(self,
                 in_channels: Sequence[Sequence[int]],
                 out_channels: Sequence[int],
                 token_dim: int,
                 num_modalities: int,
                 dropout: float = 0.0) -> None:
        super().__init__()
        assert len(in_channels) == num_modalities
        assert all(len(c) == len(out_channels) for c in in_channels)
        self.num_modalities = num_modalities
        self.num_levels = len(out_channels)
        self.out_channels = list(out_channels)
        self.token_dim = token_dim

        self.projections = nn.ModuleList()
        for modality_channels in in_channels:
            level_projs = nn.ModuleList()
            for in_ch, out_ch in zip(modality_channels, out_channels):
                level_projs.append(nn.Conv2d(in_ch, out_ch, 1))
            self.projections.append(level_projs)

        self.gate_layers = nn.ModuleList()
        for out_ch in out_channels:
            hidden_ch = max(out_ch // 2, num_modalities)
            gate = nn.Sequential(
                nn.Conv2d(out_ch * num_modalities, hidden_ch, 1),
                nn.GELU(),
                nn.Conv2d(hidden_ch, num_modalities, 1))
            nn.init.zeros_(gate[-1].weight)
            nn.init.zeros_(gate[-1].bias)
            self.gate_layers.append(gate)

        self.token_projs = nn.ModuleList(
            [nn.Linear(out_ch, token_dim) for out_ch in out_channels])
        self.level_logits = nn.Parameter(torch.zeros(self.num_levels))
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None
        self._last_gate_weights = None

    def _prepare_mask(self, modality_mask, batch_size: int,
                      device: torch.device) -> Tensor:
        if modality_mask is None:
            mask = torch.ones(
                batch_size, self.num_modalities, device=device)
        else:
            mask = torch.as_tensor(
                modality_mask, dtype=torch.float32, device=device)
            if mask.dim() == 1:
                mask = mask.unsqueeze(0).expand(batch_size, -1)
            if mask.shape != (batch_size, self.num_modalities):
                raise ValueError(
                    'modality_mask must have shape (B, M) or (M), got {}'
                    .format(tuple(mask.shape)))
            mask = mask.clone()
        # Optical is the fallback anchor, but it is only restored when a
        # sample has no modality at all -- that keeps the gated softmax
        # well defined while still honouring an explicit optical-missing
        # mask (RESEARCH_ROADMAP section 3.3 carries ``has_optical``).
        all_missing = mask.sum(dim=1) == 0
        if bool(all_missing.any()):
            mask[all_missing, 0] = 1.0
        return mask

    def forward(self,
                modality_feats: Sequence[Sequence[Tensor]],
                modality_mask: Optional[Tensor] = None):
        assert len(modality_feats) == self.num_modalities
        assert len(modality_feats[0]) == self.num_levels
        batch_size = modality_feats[0][0].shape[0]
        device = modality_feats[0][0].device
        mask = self._prepare_mask(modality_mask, batch_size, device)

        fused_levels = []
        token_levels = []
        self._last_gate_weights = []
        for level in range(self.num_levels):
            base_size = modality_feats[0][level].shape[-2:]
            projected = []
            for modality in range(self.num_modalities):
                feat = self.projections[modality][level](
                    modality_feats[modality][level])
                if feat.shape[-2:] != base_size:
                    feat = F.interpolate(
                        feat,
                        size=base_size,
                        mode='bilinear',
                        align_corners=False)
                projected.append(feat)

            stacked = torch.stack(projected, dim=1)  # B, M, C, H, W
            gate_logits = self.gate_layers[level](
                stacked.flatten(1, 2))  # B, M, H, W
            gate_logits = gate_logits.masked_fill(
                mask[:, :, None, None] == 0, -1e4)
            gate_weights = gate_logits.softmax(dim=1)
            fused = (gate_weights.unsqueeze(2) * stacked).sum(dim=1)
            if self.dropout is not None:
                fused = self.dropout(fused)
            fused_levels.append(fused)
            self._last_gate_weights.append(gate_weights.detach())

            level_tokens = []
            for modality in range(self.num_modalities):
                token = projected[modality].mean(dim=(-2, -1))
                level_tokens.append(self.token_projs[level](token))
            tokens = torch.stack(level_tokens, dim=1)
            tokens = tokens * mask[:, :, None]
            token_levels.append(tokens)

        level_weights = self.level_logits.softmax(dim=0)
        modality_tokens = sum(
            level_weights[level] * token_levels[level]
            for level in range(self.num_levels))
        return tuple(fused_levels), modality_tokens


@MODELS.register_module()
class CrossModalAlignmentLoss(nn.Module):
    """Cosine alignment between each available modality and fused tokens."""

    def __init__(self, loss_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, modality_tokens: Tensor,
                modality_mask: Tensor = None) -> Tensor:
        if modality_mask is None:
            modality_mask = modality_tokens.new_ones(
                modality_tokens.shape[:2])
        if modality_tokens.size(1) <= 1:
            return modality_tokens.sum() * 0.0
        fused = modality_tokens.mean(dim=1, keepdim=True)
        cos = F.cosine_similarity(modality_tokens, fused, dim=-1)
        cos = cos.clamp(-1.0, 1.0)
        loss = (1.0 - cos) * modality_mask
        return self.loss_weight * loss.sum() / modality_mask.sum().clamp_min(1)


@MODELS.register_module()
class PseudoTokenDiversityLoss(nn.Module):
    """Penalize overly similar modality pseudo-tokens."""

    def __init__(self, loss_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, modality_tokens: Tensor,
                modality_mask: Tensor = None) -> Tensor:
        if modality_tokens.size(1) <= 1:
            return modality_tokens.sum() * 0.0
        tokens = F.normalize(modality_tokens, dim=-1)
        similarity = tokens @ tokens.transpose(1, 2)
        num_modalities = tokens.size(1)
        eye = torch.eye(
            num_modalities, device=tokens.device, dtype=torch.bool)
        off_diag = similarity.masked_fill(eye.unsqueeze(0), 0.0)
        if modality_mask is not None:
            pair_mask = modality_mask.unsqueeze(2) * modality_mask.unsqueeze(1)
            pair_mask = pair_mask * (~eye).unsqueeze(0).float()
            off_diag = off_diag * pair_mask
            denom = pair_mask.sum().clamp_min(1)
            return self.loss_weight * off_diag.sum() / denom
        return self.loss_weight * off_diag.mean()


@MODELS.register_module()
class MissingModalityConsistencyLoss(nn.Module):
    """Feature/token consistency between full and modality-dropped fusion."""

    def __init__(self,
                 loss_weight: float = 1.0,
                 feature_weight: float = 1.0,
                 token_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.feature_weight = feature_weight
        self.token_weight = token_weight

    def forward(self,
                fused_full,
                fused_drop,
                tokens_full: Tensor,
                tokens_drop: Tensor) -> Tensor:
        feature_loss = 0.0
        for full, dropped in zip(fused_full, fused_drop):
            feature_loss = feature_loss + F.smooth_l1_loss(
                dropped, full.detach())
        feature_loss = feature_loss / max(len(fused_full), 1)
        token_loss = F.mse_loss(tokens_drop, tokens_full.detach())
        return self.loss_weight * (
            self.feature_weight * feature_loss +
            self.token_weight * token_loss)