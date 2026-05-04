class RiskAggregator:
    def __init__(self, alpha_rise=0.6, alpha_decay=0.2, peak_hold_ratio=0.7):
        """Asymmetric EMA with a peak-hold floor.

        Scam calls can contain one short high-risk burst followed by harmless
        filler. A symmetric EMA can dilute that burst. This aggregator rises
        quickly, decays slowly, and never drops below a fraction of the session
        peak.
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
            self.ema_score = (new_score * self.alpha_rise) + (self.ema_score * (1 - self.alpha_rise))
        else:
            self.ema_score = (new_score * self.alpha_decay) + (self.ema_score * (1 - self.alpha_decay))

        self.peak = max(self.peak, self.ema_score)
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
