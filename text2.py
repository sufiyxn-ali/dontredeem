from transformers import pipeline

classifier = pipeline("text-classification", model="distilbert-base-uncased")

def text_model(transcript):
    result = classifier(transcript)[0]
    
    label = result['label']
    score = result['score']
    
    # Convert to scam score
    if "NEGATIVE" in label:
        return score, "Suspicious tone detected"
    else:
        return 1 - score, "Likely safe"