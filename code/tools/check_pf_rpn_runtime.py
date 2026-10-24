# Copyright (c) OpenMMLab. All rights reserved.
"""Check whether the local PF-RPN runtime imports are available."""
import importlib


def main():
    for name in ('torch', 'torchvision', 'mmengine', 'mmcv', 'mmdet'):
        try:
            module = importlib.import_module(name)
            version = getattr(module, '__version__', 'unknown')
            print(f'{name}: {version}')
        except Exception as exc:  # noqa: BLE001
            print(f'{name}: ERROR {type(exc).__name__}: {exc}')
            continue
    try:
        from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention
        print('MultiScaleDeformableAttention: available')
    except Exception as exc:  # noqa: BLE001
        print(f'MultiScaleDeformableAttention: ERROR {exc}')


if __name__ == '__main__':
    main()