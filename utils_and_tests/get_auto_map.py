import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import warnings
warnings.filterwarnings("ignore")
from transformers import AutoConfig

try:
    config = AutoConfig.from_pretrained('emvo-ai/voiceSHIELD-small', trust_remote_code=True)
    auto_map = getattr(config, "auto_map", "NO_MAP")
    print("\n--- RESULTS ---")
    print("AUTO MAP RESULT:", auto_map)
except Exception as e:
    print("Error:", e)
