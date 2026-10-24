# Copyright (c) OpenMMLab. All rights reserved.
"""Compatibility import for the first M1 draft.

The canonical M2-PF-RPN implementation lives in
``mmdet.models.detectors.m2_pf_rpn``.
"""
from .m2_pf_rpn import M2PFRPN

MMPFRPN = M2PFRPN

__all__ = ['MMPFRPN']