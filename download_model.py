"""Download Fun-ASR-Nano-2512 model from ModelScope."""
import sys
from modelscope.hub.api import HubApi
from modelscope.hub.snapshot_download import snapshot_download

MODEL_ID = "FunAudioLLM/Fun-ASR-Nano-2512"
CACHE_DIR = r"c:\Users\EDY\Desktop\ASR-Fun-nano\models"

print(f"Downloading {MODEL_ID} to {CACHE_DIR}...")
print("This may take a while (model is ~800M params, several GB on disk)...")

try:
    model_dir = snapshot_download(
        MODEL_ID,
        cache_dir=CACHE_DIR,
        revision="master",
    )
    print(f"Model downloaded to: {model_dir}")
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)