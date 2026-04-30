def fuse_scores(audio_score, text_score, meta_score):
    """
    Fuses the scores continuously for each window.
    S_t = W_a*A_t + W_t*T_t + W_m*M
    
    Weights: W_a = 0.5, W_t = 0.3, W_m = 0.2
    """
    w_a = 0.5
    w_t = 0.3
    w_m = 0.2
    
    # Critical Text Override: If the text engine is absolutely confident (>0.85) 
    # it is a scam, bypass the audio calmness veto and shift power to text.
    if text_score > 0.85:
        w_a = 0.1
        w_t = 0.7
        w_m = 0.2
        
    # Veto Logic: If both audio and text are very low, suppress the metadata score's influence
    # This prevents unsaved/late-night calls from being flagged if the conversation is normal.
    if audio_score < 0.15 and text_score < 0.15:
        w_m = 0.05 
        
    # Semantic Veto: If text is extremely safe (< 0.1), cap the influence of aggressive audio.
    # We don't want to flag someone just because they are talking loudly or stressed about something safe.
    if text_score < 0.1:
        audio_score = min(audio_score, 0.4)
        w_m = min(w_m, 0.05) # Metadata shouldn't matter if text is perfectly safe
        
    S_t = (w_a * audio_score) + (w_t * text_score) + (w_m * meta_score)
    return S_t

def final_decision(S_final):
    """
    If S_final > 0.75 -> Likely Scam
    If 0.49 < S_final <= 0.75 -> Suspicious
    Else -> Safe
    """
    if S_final > 0.75:
        return "Likely Scam"
    elif S_final > 0.49:
        return "Suspicious"
    else:
        return "Safe"
