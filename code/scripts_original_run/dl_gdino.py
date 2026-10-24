import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/root/autodl-tmp/hf_cache"
from transformers import AutoProcessor, GroundingDinoForObjectDetection
mid = "IDEA-Research/grounding-dino-base"
print("downloading processor...", flush=True)
AutoProcessor.from_pretrained(mid)
print("downloading model...", flush=True)
GroundingDinoForObjectDetection.from_pretrained(mid)
print("DONE", flush=True)
