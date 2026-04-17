import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import warnings
warnings.filterwarnings("ignore")
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, AutoModelForCTC, AutoModel, AutoModelForCausalLM

model_id = "emvo-ai/voiceSHIELD-small"

print("--- TESTING ---")
print("Trying AutoModelForCTC...")
try:
    model = AutoModelForCTC.from_pretrained(model_id, trust_remote_code=True)
    print("AutoModelForCTC Success!", type(model))
except Exception as e:
    print("AutoModelForCTC Failed")

print("Trying AutoModelForSpeechSeq2Seq...")
try:
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, trust_remote_code=True)
    print("AutoModelForSpeechSeq2Seq Success!", type(model))
except Exception as e:
    print("AutoModelForSpeechSeq2Seq Failed")

print("Trying AutoModel...")
try:
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    print("AutoModel Success!", type(model))
except Exception as e:
    print("AutoModel Failed")
