#!/bin/bash
cd /root/autodl-tmp || exit 1
echo "=== prepare_sen12landslides.py 的 main() 逻辑（输出结构）==="
sed -n "207,260p" PF-RPN/tools/data/prepare_sen12landslides.py
echo
echo "=== merge_positive_negative.py 做什么 ==="
head -20 PF-RPN/tools/data/merge_positive_negative.py
echo
echo "=== 是否有构建 lb_pos / lb_neg 的历史命令记录 ==="
grep -rn "prepare_sen12landslides\|merge_positive_negative" *.sh *.log PF-RPN/docs/*.md 2>/dev/null | head -15
echo
echo "=== 数据构建相关日志 ==="
ls -la /root/autodl-tmp/Sen12Landslides/logs/ 2>/dev/null | head
echo
echo "=== xarray 是否在依赖里 ==="
grep -rn "xarray\|netcdf" PF-RPN/requirements* PF-RPN/setup.py 2>/dev/null | head
echo "  实际是否安装:"; /root/miniconda3/bin/python3 -c "import xarray; print('  xarray', xarray.__version__)" 2>&1 | tail -1

