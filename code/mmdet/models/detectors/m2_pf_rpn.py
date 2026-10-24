# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList

from .pf_rpn import PFRPN
from ..layers.multimodal_fusion import (CrossModalAlignmentLoss,
                                         GatedMultiModalFusion,
                                         LightweightModalityStem,
                                         MissingModalityConsistencyLoss,
                                         PseudoTokenDiversityLoss)


@MODELS.register_module()
class M2PFRPN(PFRPN):
    """M2-PF-RPN: multi-modal prompt-free region proposal network.

    Design goals follow ``RESEARCH_ROADMAP.md`` section 5:

    - optical, SAR, topography and temporal-change branches;
    - spatial gated cross-modal fusion with missing-modality masking;
    - modality-aware pseudo-text tokens and optional cross-modal router;
    - proposal objectness/score/level/uncertainty output hooks;
    - independently switchable router, alignment, diversity and
      missing-modality consistency losses.
    """

    def __init__(self,
                 *args,
                 modality_channels: Dict[str, int] = None,
                 modality_order: Sequence[str] = None,
                 modality_encoders: Dict[str, Dict] = None,
                 sar_backbone: Dict = None,
                 topography_backbone: Dict = None,
                 terrain_backbone: Dict = None,
                 change_backbone: Dict = None,
                 fusion_cfg: Dict = None,
                 candidate_name: str = 'landslide_candidate',
                 use_sar: bool = True,
                 use_topography: bool = True,
                 use_change: bool = True,
                 use_modality_tokens: bool = True,
                 use_cross_modal_router: bool = True,
                 use_router: bool = True,
                 router_loss_weight: float = 0.0,
                 router_z_loss_weight: float = 0.0,
                 cross_modal_align_loss_weight: float = 0.0,
                 pseudo_token_diversity_loss_weight: float = 0.0,
                 missing_modality_loss_weight: float = 0.0,
                 missing_modality_feature_weight: float = 1.0,
                 missing_modality_token_weight: float = 1.0,
                 modality_dropout_prob: float = 0.1,
                 allow_optical_dropout: bool = False,
                 **kwargs) -> None:
        modality_channels = modality_channels or dict(optical=3)
        modality_order = tuple(modality_order or modality_channels.keys())
        all_modality_order = modality_order
        all_modality_channels = dict(modality_channels)
        assert modality_order[0] == 'optical', (
            'The first modality must be optical so that the released optical '
            'backbone can be reused.')

        enabled = {
            'optical': True,
            'sar': use_sar,
            'topography': use_topography,
            'terrain': use_topography,
            'change': use_change,
        }
        filtered_order = []
        for modality in modality_order:
            if enabled.get(modality, True):
                filtered_order.append(modality)
        if 'topography' in filtered_order and 'terrain' in filtered_order:
            filtered_order.remove('terrain')
        modality_order = tuple(filtered_order)
        modality_channels = {
            modality: modality_channels[modality]
            for modality in modality_order
        }

        self.modality_channels = dict(modality_channels)
        self.modality_order = modality_order
        self._all_modality_order = all_modality_order
        self._all_modality_channels = all_modality_channels
        self.candidate_name = candidate_name
        self.router_loss_weight = router_loss_weight
        self.router_z_loss_weight = router_z_loss_weight
        self.use_modality_tokens = use_modality_tokens
        self.use_cross_modal_router = use_cross_modal_router
        self.use_router = use_router
        self.modality_dropout_prob = modality_dropout_prob
        self.allow_optical_dropout = allow_optical_dropout
        self.cross_modal_align_loss_weight = cross_modal_align_loss_weight
        self.pseudo_token_diversity_loss_weight = \
            pseudo_token_diversity_loss_weight
        self.missing_modality_loss_weight = missing_modality_loss_weight

        neck_cfg = kwargs.get('neck') or {}
        optical_channels = list(neck_cfg.get('in_channels', [256, 512, 1024]))

        super().__init__(*args, **kwargs)

        encoder_cfgs = dict(modality_encoders or {})
        if sar_backbone is not None:
            encoder_cfgs['sar'] = sar_backbone
        if topography_backbone is not None:
            encoder_cfgs['topography'] = topography_backbone
        if terrain_backbone is not None:
            encoder_cfgs['terrain'] = terrain_backbone
        if change_backbone is not None:
            encoder_cfgs['change'] = change_backbone

        self.modality_encoders = nn.ModuleDict()
        for modality in self.modality_order[1:]:
            if modality in encoder_cfgs:
                encoder = MODELS.build(encoder_cfgs[modality])
            else:
                encoder = LightweightModalityStem(
                    in_channels=self.modality_channels[modality],
                    out_channels=optical_channels)
            self.modality_encoders[modality] = encoder

        fusion_cfg = dict(fusion_cfg or {})
        fusion_in_channels = fusion_cfg.pop('in_channels', None)
        if (fusion_in_channels is not None and
                len(fusion_in_channels) == len(all_modality_order)):
            keep_indices = [
                all_modality_order.index(m) for m in self.modality_order
            ]
            fusion_in_channels = [
                fusion_in_channels[i] for i in keep_indices
            ]
        if fusion_in_channels is None:
            fusion_in_channels = [optical_channels] * len(self.modality_order)
        fusion_out_channels = list(
            fusion_cfg.pop('out_channels', optical_channels))
        token_dim = fusion_cfg.pop('token_dim', self.embed_dims)
        self.fusion = GatedMultiModalFusion(
            in_channels=fusion_in_channels,
            out_channels=fusion_out_channels,
            token_dim=token_dim,
            num_modalities=len(self.modality_order),
            **fusion_cfg)

        self.modal_text_proj = (
            nn.Linear(self.embed_dims, self.embed_dims)
            if (use_modality_tokens and not use_cross_modal_router) else None)
        self.cross_modal_align_loss = (
            CrossModalAlignmentLoss(cross_modal_align_loss_weight)
            if cross_modal_align_loss_weight > 0 else None)
        self.pseudo_token_diversity_loss = (
            PseudoTokenDiversityLoss(pseudo_token_diversity_loss_weight)
            if pseudo_token_diversity_loss_weight > 0 else None)
        self.missing_modality_loss = (
            MissingModalityConsistencyLoss(
                missing_modality_loss_weight,
                feature_weight=missing_modality_feature_weight,
                token_weight=missing_modality_token_weight)
            if missing_modality_loss_weight > 0 else None)

    def init_weights(self) -> None:
        super().init_weights()
        for module in self.modality_encoders.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for module in self.fusion.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for gate in self.fusion.gate_layers:
            nn.init.zeros_(gate[-1].weight)
            nn.init.zeros_(gate[-1].bias)

    def _get_modality_mask(self, batch_data_samples) -> Tensor:
        masks = []
        for data_sample in batch_data_samples:
            mask = None
            if hasattr(data_sample, 'get'):
                mask = data_sample.get('modality_mask', None)
            if mask is None:
                mask = getattr(data_sample, 'modality_mask', None)
            if mask is None:
                mask = [1] * len(self.modality_order)
            masks.append(
                torch.as_tensor(mask, dtype=torch.float32))
        modality_mask = torch.stack(masks, dim=0)
        if modality_mask.size(1) == len(self._all_modality_order):
            keep_indices = [
                self._all_modality_order.index(m) for m in self.modality_order
            ]
            modality_mask = modality_mask[:, keep_indices]
        return modality_mask

    def _split_inputs(self, batch_inputs: Tensor) -> Dict[str, Tensor]:
        active_channels = [
            self.modality_channels[m] for m in self.modality_order
        ]
        all_channels = [
            self._all_modality_channels[m] for m in self._all_modality_order
        ]
        num_input_channels = batch_inputs.shape[1]
        if num_input_channels == sum(active_channels):
            order = self.modality_order
            channels = active_channels
        elif num_input_channels == sum(all_channels):
            order = self._all_modality_order
            channels = all_channels
        else:
            raise ValueError(
                'Expected {} active channels or {} full channels, got {}.'
                .format(sum(active_channels), sum(all_channels),
                        num_input_channels))
        split = dict(zip(order, torch.split(batch_inputs, channels, dim=1)))
        return {modality: split[modality] for modality in self.modality_order}

    def extract_modality_feats(self, batch_inputs: Tensor):
        modality_inputs = self._split_inputs(batch_inputs)
        modality_feats = []
        for modality in self.modality_order:
            if modality == 'optical':
                feats = self.backbone(modality_inputs[modality])
            else:
                feats = self.modality_encoders[modality](
                    modality_inputs[modality])
            modality_feats.append(feats)
        return modality_feats

    def fuse_modality_feats(self,
                            modality_feats,
                            modality_mask: Tensor = None,
                            apply_neck: bool = True):
        fused_feats, modality_tokens = self.fusion(
            modality_feats, modality_mask)
        if apply_neck and self.with_neck:
            fused_feats = self.neck(fused_feats)
        return fused_feats, modality_tokens

    def extract_feat_with_modality_tokens(self,
                                          batch_inputs: Tensor,
                                          modality_mask: Tensor = None):
        modality_feats = self.extract_modality_feats(batch_inputs)
        return self.fuse_modality_feats(modality_feats, modality_mask)

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        feats, _ = self.extract_feat_with_modality_tokens(batch_inputs)
        return feats

    def _build_multimodal_pseudo_text(
            self,
            memory: Tensor,
            spatial_shapes: Tensor,
            modality_tokens: Tensor = None,
            modality_mask: Tensor = None) -> Tuple[Tensor, Tensor]:
        batch_size = memory.shape[0]
        multi_level_global_feats = []
        expert_feats = []

        offset = 0
        for height, width in spatial_shapes.tolist():
            num_points = height * width
            vis_feat_l = memory[:, offset:offset + num_points, :]
            offset += num_points
            vis_feat_l = vis_feat_l.reshape(
                batch_size, height, width, -1).permute(0, 3, 1,
                                                       2).contiguous()
            vis_global_l, vis_local_l = self.vis_proj(vis_feat_l)
            multi_level_global_feats.append(vis_global_l)
            expert_feat = torch.cat(
                [vis_global_l[:, :, None], vis_local_l.flatten(2)], dim=2)
            expert_feats.append(expert_feat.permute(0, 2, 1).contiguous())

        expert_feat = torch.stack(multi_level_global_feats, dim=1)
        expert_mask = expert_feat.new_ones(expert_feat.shape[:2])

        if (self.use_modality_tokens and self.use_cross_modal_router and
                modality_tokens is not None):
            modality_tokens = modality_tokens.to(expert_feat.dtype)
            expert_feat = torch.cat([expert_feat, modality_tokens], dim=1)
            if modality_mask is None:
                modality_mask = modality_tokens.new_ones(
                    modality_tokens.shape[:2])
            expert_mask = torch.cat(
                [expert_mask, modality_mask.to(expert_mask.dtype)], dim=1)

        if self.use_router:
            router_weights = self.router(expert_feat)
        else:
            router_weights = expert_feat.new_ones(expert_feat.shape[:2] +
                                                  (1, ))
        router_weights = router_weights.masked_fill(
            expert_mask[:, :, None] == 0, -1e4)

        text_feat = self.learnable_text_embedding.expand(
            batch_size, -1, -1).to(memory.device)
        k = min(self.topk, router_weights.size(1))
        idx_selected = torch.topk(
            router_weights, k=k, dim=1).indices.squeeze(-1)
        self.router._last_idx_selected = idx_selected.detach()

        selected_weights = F.softmax(
            router_weights.gather(1, idx_selected.unsqueeze(-1)),
            dim=1).squeeze(-1)
        meta_out = torch.zeros(
            batch_size,
            idx_selected.size(1),
            text_feat.size(1),
            text_feat.size(2),
            device=text_feat.device,
            dtype=text_feat.dtype)

        num_level_experts = len(expert_feats)
        for expert_id, level_feat in enumerate(expert_feats):
            selected = idx_selected == expert_id
            if not selected.any():
                continue
            batch_idx, k_idx = selected.nonzero(as_tuple=True)
            meta_out[batch_idx, k_idx] = self.meta_net(text_feat[batch_idx], level_feat[batch_idx]).to(meta_out.dtype)

        if modality_tokens is not None:
            for modality_id in range(modality_tokens.size(1)):
                expert_id = num_level_experts + modality_id
                selected = (idx_selected == expert_id)
                if not selected.any():
                    continue
                batch_idx, k_idx = selected.nonzero(as_tuple=True)
                modality_feat = modality_tokens[
                    batch_idx, modality_id:modality_id + 1, :]
                meta_out[batch_idx, k_idx] = self.meta_net(text_feat[batch_idx], modality_feat).to(meta_out.dtype)

        pseudo_text_embeds = (selected_weights[:, :, None, None] *
                              meta_out).sum(dim=1)
        if (self.use_modality_tokens and
                not self.use_cross_modal_router and
                modality_tokens is not None and
                self.modal_text_proj is not None):
            modal_summary = modality_tokens.mean(dim=1, keepdim=True)
            pseudo_text_embeds = (
                pseudo_text_embeds + self.modal_text_proj(modal_summary))
        return pseudo_text_embeds, router_weights

    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        text_dict: Dict,
        batch_data_samples: OptSampleList = None,
        modality_tokens: Tensor = None,
        modality_mask: Tensor = None,
    ) -> Dict:
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)
        spatial_shapes = encoder_inputs_dict['spatial_shapes']
        memory = encoder_inputs_dict['feat']

        pseudo_text_embeds, router_weights = \
            self._build_multimodal_pseudo_text(
                memory, spatial_shapes, modality_tokens, modality_mask)
        enhanced = self.cascade_self_prompt_enhancement(
            vis_embedding=memory,
            text_embedding=pseudo_text_embeds[:, 1:2, :],
            spatial_shapes=spatial_shapes.tolist(),
            thr=self.sp_thr,
            iter_num=self.sp_iter_num)
        text_embeddings = pseudo_text_embeds.clone()
        text_embeddings[:, 1:2, :] = enhanced
        self._init_text_dict(text_dict, text_embeddings)

        encoder_outputs_dict = self.forward_encoder(
            **encoder_inputs_dict, text_dict=text_dict)
        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        head_inputs_dict['router_weights'] = router_weights
        return head_inputs_dict

    def _drop_modalities(self, modality_mask: Tensor) -> Tensor:
        if modality_mask.size(1) <= 1:
            return modality_mask
        dropped = modality_mask.clone()
        for batch_idx in range(dropped.size(0)):
            active = (dropped[batch_idx] > 0).nonzero(
                as_tuple=False).flatten().tolist()
            if self.allow_optical_dropout:
                candidates = list(active)
            else:
                candidates = [idx for idx in active if idx != 0]
            # Never drop the last modality; keeping the draw after the
            # emptiness guard preserves the original RNG stream.
            if (candidates and len(active) >= 2 and
                    torch.rand(1).item() < self.modality_dropout_prob):
                drop_idx = candidates[
                    torch.randint(len(candidates), (1, )).item()]
                dropped[batch_idx, drop_idx] = 0.0
        return dropped

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Dict[str, Tensor]:
        text_dict = {}
        modality_mask = self._get_modality_mask(batch_data_samples)
        modality_mask = modality_mask.to(batch_inputs.device)
        modality_feats = self.extract_modality_feats(batch_inputs)
        fused_feats, modality_tokens = self.fuse_modality_feats(
            modality_feats, modality_mask)

        dropped_feats = None
        dropped_tokens = None
        if self.training and self.missing_modality_loss is not None:
            dropped_mask = self._drop_modalities(modality_mask)
            dropped_feats, dropped_tokens = self.fuse_modality_feats(
                modality_feats, dropped_mask)

        head_inputs_dict = self.forward_transformer(
            fused_feats,
            text_dict,
            batch_data_samples,
            modality_tokens=modality_tokens,
            modality_mask=modality_mask)
        router_weights = head_inputs_dict.pop('router_weights', None)

        num_tokens = text_dict['text_token_mask'].shape[1]
        positive_idx = 1 if num_tokens > 1 else 0
        for i, data_samples in enumerate(batch_data_samples):
            text_token_mask = text_dict['text_token_mask'][i]
            num_gt = data_samples.gt_instances.labels.shape[0]
            positive_map = torch.zeros(
                (num_gt, num_tokens),
                dtype=torch.float32,
                device=batch_inputs.device)
            if num_gt > 0:
                positive_map[:, positive_idx] = 1
            data_samples.gt_instances.positive_maps = positive_map
            data_samples.gt_instances.text_token_mask = \
                text_token_mask.unsqueeze(0).repeat(num_gt, 1)

        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)

        if self.router_loss_weight > 0 and router_weights is not None:
            router_probs = router_weights.squeeze(-1).softmax(dim=1)
            usage = router_probs.mean(dim=0)
            uniform = torch.full_like(usage, 1.0 / usage.numel())
            losses['loss_router_balance'] = \
                self.router_loss_weight * (usage - uniform).pow(2).mean()

        if self.router_z_loss_weight > 0 and router_weights is not None:
            logits = router_weights.squeeze(-1)
            losses['loss_router_z'] = self.router_z_loss_weight * \
                torch.logsumexp(logits, dim=1).pow(2).mean()

        if (self.cross_modal_align_loss is not None and
                modality_tokens is not None):
            losses['loss_cross_modal_align'] = self.cross_modal_align_loss(
                modality_tokens, modality_mask)

        if (self.pseudo_token_diversity_loss is not None and
                modality_tokens is not None):
            losses['loss_pseudo_token_diversity'] = \
                self.pseudo_token_diversity_loss(
                    modality_tokens, modality_mask)

        if (self.missing_modality_loss is not None and
                dropped_feats is not None):
            losses['loss_missing_modality_consistency'] = \
                self.missing_modality_loss(
                    fused_feats, dropped_feats,
                    modality_tokens, dropped_tokens)
        return losses

    @staticmethod
    def _assign_levels(bboxes: Tensor) -> Tensor:
        """Assign a feature-level proxy from box area.

        Level 0/1/2 correspond to the stride-8/16/32 proposal scales used by
        the multi-scale encoder.  This is a stable M1 export proxy; M2 can
        refine it with RoI heads.
        """
        if bboxes.numel() == 0:
            return bboxes.new_zeros((0, ), dtype=torch.long)
        widths = (bboxes[:, 2] - bboxes[:, 0]).clamp_min(1.0)
        heights = (bboxes[:, 3] - bboxes[:, 1]).clamp_min(1.0)
        areas = widths * heights
        levels = torch.zeros_like(areas, dtype=torch.long)
        levels[(areas >= 32 * 32) & (areas < 96 * 96)] = 1
        levels[areas >= 96 * 96] = 2
        return levels

    def extract_roi_features(self,
                             batch_inputs: Tensor,
                             bboxes: Tensor,
                             modality_mask: Tensor = None,
                             output_size: int = 7) -> Tensor:
        """Extract stride-8 RoI features for M2 candidate features."""
        from torchvision.ops import roi_align

        feats, _ = self.extract_feat_with_modality_tokens(
            batch_inputs, modality_mask)
        feat = feats[0]
        if bboxes.numel() == 0:
            return feat.new_zeros((0, feat.shape[1], output_size,
                                   output_size))
        rois = torch.cat([
            bboxes.new_zeros((bboxes.shape[0], 1)),
            bboxes,
        ], dim=1)
        return roi_align(
            feat,
            rois,
            output_size=output_size,
            spatial_scale=1.0 / 8.0,
            aligned=True)

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        modality_mask = self._get_modality_mask(batch_data_samples)
        modality_mask = modality_mask.to(batch_inputs.device)
        visual_feats, modality_tokens = \
            self.extract_feat_with_modality_tokens(
                batch_inputs, modality_mask)
        text_dict = {}
        entities = ([self.candidate_name],) * batch_inputs.shape[0]
        for data_samples in batch_data_samples:
            data_samples.token_positive_map = [1]

        head_inputs_dict = self.forward_transformer(
            visual_feats,
            text_dict,
            batch_data_samples,
            modality_tokens=modality_tokens,
            modality_mask=modality_mask)
        head_inputs_dict.pop('router_weights', None)
        results_list = self.bbox_head.predict(
            **head_inputs_dict,
            rescale=rescale,
            batch_data_samples=batch_data_samples)

        for data_sample, pred_instances, entity in zip(
                batch_data_samples, results_list, entities):
            if len(pred_instances) > 0:
                scores = pred_instances.scores
                pred_instances.objectness = scores
                pred_instances.proposal_score = scores
                pred_instances.uncertainty = (1.0 - scores).clamp(0.0, 1.0)
                pred_instances.level = self._assign_levels(
                    pred_instances.bboxes)
                label_names = []
                for label in pred_instances.labels:
                    label_idx = int(label)
                    if label_idx >= len(entity):
                        warnings.warn(
                            'Unexpected label index; fallback to "unobject".')
                        label_names.append('unobject')
                    else:
                        label_names.append(entity[label_idx])
                pred_instances.label_names = label_names
            data_sample.pred_instances = pred_instances
        return batch_data_samples


# Backward-compatible name used by the first module draft.
MMPFRPN = M2PFRPN