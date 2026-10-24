# Copyright (c) OpenMMLab. All rights reserved.
"""Deprecated first-draft alias.

Use ``mmdet.datasets.multimodal_landslide.MultiModalLandslideDataset``.
"""
from .multimodal_landslide import MultiModalLandslideDataset

MultiModalCocoDataset = MultiModalLandslideDataset

__all__ = ['MultiModalCocoDataset']