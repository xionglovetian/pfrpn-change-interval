# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence

import torch

from mmdet.registry import MODELS

from .data_preprocessor import DetDataPreprocessor


@MODELS.register_module()
class MultiModalDetDataPreprocessor(DetDataPreprocessor):
    """Detection preprocessor with arbitrary multi-modal channel counts.

    MMEngine's default ``DetDataPreprocessor`` only accepts 1 or 3 channel
    normalization statistics.  M2-PF-RPN concatenates optical, SAR,
    topography and change channels, so this class registers per-channel
    ``mean``/``std`` buffers without the RGB/gray assertion.
    """

    def __init__(self,
                 mean: Optional[Sequence[float]] = None,
                 std: Optional[Sequence[float]] = None,
                 **kwargs) -> None:
        super().__init__(mean=None, std=None, **kwargs)
        if mean is not None:
            assert std is not None, 'std must be provided with mean'
            assert len(mean) == len(std), 'mean and std length mismatch'
            self._enable_normalize = True
            self.register_buffer(
                'mean', torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1),
                False)
            self.register_buffer(
                'std', torch.tensor(std, dtype=torch.float32).view(-1, 1, 1),
                False)