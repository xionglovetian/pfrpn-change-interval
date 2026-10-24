#!/bin/bash
OUT=/root/autodl-tmp/models/grounding-dino-base
mkdir -p "$OUT"; cd "$OUT" || exit 1
B="https://modelscope.cn/api/v1/models/IDEA-Research/grounding-dino-base/repo?Revision=master&FilePath="
for f in config.json preprocessor_config.json tokenizer.json tokenizer_config.json vocab.txt special_tokens_map.json model.safetensors; do
  echo "[dl] $f" 
  curl -sL -C - -o "$f" "$B$f" || echo "[FAIL] $f"
done
echo "ALL DONE $(date)"
