from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, AutoModelForCTC, AutoModel

model_id = "emvo-ai/voiceSHIELD-small"

print("Trying AutoProcessor...")
try:
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    print("Processor loaded!")
except Exception as e:
    print("Processor failed:", e)

print("\nTrying AutoModelForSpeechSeq2Seq...")
try:
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, trust_remote_code=True)
    print("AutoModelForSpeechSeq2Seq loaded!")
except Exception as e:
    print("AutoModelForSpeechSeq2Seq failed:", e)

print("\nTrying AutoModelForCTC...")
try:
    model = AutoModelForCTC.from_pretrained(model_id, trust_remote_code=True)
    print("AutoModelForCTC loaded!")
except Exception as e:
    print("AutoModelForCTC failed:", e)

print("\nTrying AutoModel...")
try:
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    print("AutoModel loaded!", type(model))
except Exception as e:
    print("AutoModel failed:", e)
