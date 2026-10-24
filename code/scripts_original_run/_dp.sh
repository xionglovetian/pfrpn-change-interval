#!/bin/bash
cd /root/autodl-tmp || exit 1
echo "=== 1) lb_pos/events 结构（物理数据长什么样）==="
ls -1 PF-RPN/data/landslide_multimodal_lb_pos/events/ | head
echo "--- 单个事件内部 ---"
ev=$(ls -1 PF-RPN/data/landslide_multimodal_lb_pos/events/ | head -1)
ls -la "PF-RPN/data/landslide_multimodal_lb_pos/events/$ev" | head -8
t=$(ls -1 "PF-RPN/data/landslide_multimodal_lb_pos/events/$ev/tiles" 2>/dev/null | head -1)
echo "--- 单个瓦片 ---"
ls -la "PF-RPN/data/landslide_multimodal_lb_pos/events/$ev/tiles/$t" 2>/dev/null
echo
echo "=== 2) 数据是从哪下载的？（cloud_download*.sh）==="
for f in cloud_download.sh cloud_download2.sh cloud_download3.sh cloud_download_final.sh; do
  echo "----- $f -----"
  grep -vE "^\s*#|^\s*$" $f 2>/dev/null | head -12
done
echo
echo "=== 3) 搜索任何构建 npy 瓦片的脚本 ==="
grep -rln "optical_pre\|sar_pre\|change.npy\|topography.npy" --include="*.py" --include="*.sh" . 2>/dev/null | grep -v work_dirs | head -15
echo
echo "=== 4) 是否有数据处理/切片脚本 ==="
ls -1 PF-RPN/tools/ | grep -viE "train|eval|test|analy|summar|report|check|convert|distill|misc|visual" | head -20

