#!/bin/bash
set -e
export PY=/root/miniconda3/bin/python3
export PIP="/root/miniconda3/bin/pip"
export HF_ENDPOINT=https://hf-mirror.com
cd /root/autodl-tmp/PF-RPN
echo "=== step1: upgrade pip/setuptools ==="
$PIP install -q --upgrade pip setuptools wheel
echo "=== step2: mmengine ==="
$PIP install -q "mmengine>=0.10.0,<1.0.0"
echo "=== step3: mmcv 2.1.0 (cu121/torch2.1.0) ==="
$PIP install -q mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html
echo "=== step4: mmdet editable ==="
$PIP install -q -e . --no-build-isolation
echo "=== step5: extras ==="
$PIP install -q pycocotools scipy shapely "opencv-python==4.9.0.80" "numpy<2" huggingface_hub
echo "=== done ==="
$PY - <<PYEOF
import torch, mmcv, mmengine, mmdet, numpy, scipy, shapely, cv2, pycocotools
print("torch", torch.__version__, torch.cuda.is_available())
print("mmcv", mmcv.__version__)
print("mmengine", mmengine.__version__)
print("mmdet", mmdet.__version__)
print("numpy", numpy.__version__)
PYEOF
echo "ALL_OK"