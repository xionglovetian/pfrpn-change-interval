#!/bin/bash
set -e
export PIP=/root/miniconda3/bin/pip
cd /root/autodl-tmp/PF-RPN
echo "=== pin numpy<2 and setuptools<81 ==="
$PIP install -q "numpy==1.26.4" "setuptools<81" wheel
echo "=== verify pkg_resources ==="
/root/miniconda3/bin/python3 -c "import pkg_resources, numpy; print('numpy', numpy.__version__)"
echo "=== mmdet editable install ==="
$PIP install -q -e . --no-build-isolation
echo "=== extras ==="
$PIP install -q pycocotools scipy shapely "opencv-python==4.9.0.80" huggingface_hub
echo "=== verify all ==="
/root/miniconda3/bin/python3 - <<PYEOF
import torch, mmcv, mmengine, mmdet, numpy, scipy, shapely, cv2, pycocotools
print("torch", torch.__version__, torch.cuda.is_available())
print("mmcv", mmcv.__version__)
print("mmengine", mmengine.__version__)
print("mmdet", mmdet.__version__)
print("numpy", numpy.__version__)
print("cv2", cv2.__version__)
PYEOF
echo "FIX_ALL_OK"