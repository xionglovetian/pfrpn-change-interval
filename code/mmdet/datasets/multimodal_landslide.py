# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
from typing import List, Optional

from mmengine.fileio import get

from mmdet.registry import DATASETS

from .base_det_dataset import BaseDetDataset


@DATASETS.register_module()
class MultiModalLandslideDataset(BaseDetDataset):
    """Minimal adapter for the M0 multimodal landslide data contract.

    The dataset reads ``dataset_manifest.json`` or ``.jsonl`` entries whose
    fields follow ``RESEARCH_ROADMAP.md`` section 3.  It is intentionally a
    thin adapter: M0 remains responsible for co-registration, split
    generation, leakage checks and final data validation.

    Args:
        ann_file (str): Manifest path.
        data_root (str, optional): Root for relative modality paths.
        modality_keys (dict, optional): Mapping from logical modality key to
            the key stored in the manifest's ``modalities`` dictionary.
        return_classes (bool): Add prompt-free class metadata.
    """

    def __init__(self,
                 ann_file: str,
                 data_root: Optional[str] = None,
                 modality_keys: Optional[dict] = None,
                 return_classes: bool = True,
                 **kwargs) -> None:
        self.modality_keys = modality_keys or dict(
            optical_pre='optical_pre',
            optical_post='optical_post',
            sar_pre='sar_pre',
            sar_post='sar_post',
            dem='dem',
            topography='topography',
            change='change',
            valid_mask='valid_mask')
        super().__init__(
            ann_file=ann_file,
            data_root=data_root,
            return_classes=return_classes,
            **kwargs)

    def _read_manifest(self) -> List[dict]:
        path = self.ann_file
        if not osp.isabs(path) and not osp.exists(path):
            root = self.data_root or ''
            path = osp.join(root, path)
        if path.endswith('.jsonl'):
            text = get(path, backend_args=self.backend_args)
            if isinstance(text, bytes):
                text = text.decode('utf-8')
            return [
                json.loads(line) for line in text.splitlines()
                if line.strip()
            ]
        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        if isinstance(data, dict):
            data = data.get('samples', data.get('data', []))
        if not isinstance(data, list):
            raise TypeError(
                'Manifest must be a list or contain a "samples" list.')
        return data

    def _resolve(self, path: Optional[str]) -> Optional[str]:
        if path is None:
            return None
        if osp.isabs(path):
            return path
        return osp.join(self.data_root or '', path)

    def _load_labels(self, sample: dict):
        labels = sample.get('labels', sample.get('annotations', []))
        if isinstance(labels, str):
            label_path = self._resolve(labels)
            with open(label_path, 'r', encoding='utf-8') as file:
                labels = json.load(file)
        if isinstance(labels, dict):
            annotations = labels.get('annotations', [])
            negatives = labels.get('negatives', [])
        else:
            annotations = labels or []
            negatives = sample.get('negatives', [])
        return annotations, negatives

    def _build_instances(self, annotations, negatives) -> List[dict]:
        instances = []
        for ann in annotations:
            bbox = ann.get('bbox_xyxy')
            if bbox is None:
                bbox = ann.get('bbox')
            if bbox is None:
                continue
            label = ann.get('bbox_label', ann.get('category_id', 1))
            label = int(label)
            if label > 0:
                label -= 1
            instance = dict(
                bbox=[float(v) for v in bbox],
                bbox_label=label,
                ignore_flag=int(ann.get('ignore_flag', 0)))
            if ann.get('segmentation') is not None:
                instance['mask'] = ann['segmentation']
            instances.append(instance)

        for neg in negatives:
            bbox = neg.get('bbox_xyxy', neg.get('bbox'))
            if bbox is None:
                continue
            instances.append(
                dict(
                    bbox=[float(v) for v in bbox],
                    bbox_label=0,
                    ignore_flag=1))
        return instances

    def load_data_list(self) -> List[dict]:
        data_list = []
        for sample in self._read_manifest():
            modalities = sample.get('modalities', {})
            path_fields = {}
            for logical_key, manifest_key in self.modality_keys.items():
                path = modalities.get(manifest_key, sample.get(manifest_key))
                path_fields[f'{logical_key}_path'] = self._resolve(path)

            optical_path = (path_fields.get('optical_pre_path') or
                            path_fields.get('optical_post_path'))
            sar_paths = [
                path for path in (path_fields.get('sar_pre_path'),
                                  path_fields.get('sar_post_path'))
                if path is not None
            ]
            topography_path = path_fields.get('topography_path')
            change_path = path_fields.get('change_path')
            modality_mask = sample.get('modality_mask')
            if modality_mask is None:
                modality_mask = [
                    int(optical_path is not None),
                    int(bool(sar_paths)),
                    int(topography_path is not None),
                    int(change_path is not None),
                ]

            shape = sample.get('shape', [None, None])
            height, width = (
                shape + [None, None])[:2] if shape else (None, None)
            annotations, negatives = self._load_labels(sample)
            data_info = dict(
                img_path=optical_path or '',
                optical_path=optical_path,
                sar_path=sar_paths or None,
                topography_path=topography_path,
                change_path=change_path,
                img_id=sample.get('sample_id', sample.get('image_id')),
                sample_id=sample.get('sample_id'),
                event_id=sample.get('event_id'),
                image_id=sample.get('image_id'),
                height=height,
                width=width,
                instances=self._build_instances(annotations, negatives),
                modality_mask=modality_mask)
            data_info.update(path_fields)
            if self.return_classes:
                data_info['text'] = self.metainfo['classes']
                data_info['custom_entities'] = True
            data_list.append(data_info)
        return data_list
    def filter_data(self) -> List[dict]:
        """Keep event-level samples unchanged; filtering is M0's job."""
        return self.data_list