class RiskAggregator:
    def __init__(self, alpha_rise=0.6, alpha_decay=0.2, peak_hold_ratio=0.70):
        """
        Asymmetric EMA Ratchet with Peak-Hold Floor.
        
        - alpha_rise: weight for new score when threat is INCREASING (fast rise).
        - alpha_decay: weight for new score when threat is DECREASING (slow decay).
        - peak_hold_ratio: the EMA will never drop below this fraction of the
          historical peak, preventing safe tail-end windows from diluting
          earlier confirmed threat spikes.
        """
        self.alpha_rise = alpha_rise
        self.alpha_decay = alpha_decay
        self.peak_hold_ratio = peak_hold_ratio
        self.ema_score = None
        self.peak = 0.0

    def update(self, new_score):
        if self.ema_score is None:
            self.ema_score = new_score
        elif new_score > self.ema_score:
            # Threat is rising — react quickly
            self.ema_score = (new_score * self.alpha_rise) + (self.ema_score * (1 - self.alpha_rise))
        else:
            # Threat is falling — decay slowly to preserve context
            self.ema_score = (new_score * self.alpha_decay) + (self.ema_score * (1 - self.alpha_decay))
        
        # Track the all-time peak
        self.peak = max(self.peak, self.ema_score)
        
        # Peak-hold floor: never let the score drop below 70% of peak
        self.ema_score = max(self.ema_score, self.peak * self.peak_hold_ratio)
        
        return self.ema_score


class SessionStateManager:
    def __init__(self):
        self.total_windows = 0
        self.risk_history = []
        self.suspicious_tokens = set()
        self.aggregator = RiskAggregator()

    def process_window(self, fused_score, tokens):
        """Records the window and returns the temporally smoothed EMA score."""
        self.total_windows += 1
        self.risk_history.append(fused_score)
        
        if tokens:
            self.suspicious_tokens.update(t[0] for t in tokens)
            
        smoothed_score = self.aggregator.update(fused_score)
        return smoothed_score

    def get_session_summary(self):
        return {
            "total_windows": self.total_windows,
            "max_spike": max(self.risk_history) if self.risk_history else 0,
            "all_tokens": list(self.suspicious_tokens)
        }
