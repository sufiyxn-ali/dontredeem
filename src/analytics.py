class RiskAggregator:
    def __init__(self, alpha=0.4):
        """
        Exponential Moving Average (EMA) Aggregator.
        Alpha controls how much weight the newest window has. 
        Higher alpha = more reactive to sudden threats.
        """
        self.alpha = alpha
        self.ema_score = None

    def update(self, new_score):
        if self.ema_score is None:
            self.ema_score = new_score
        else:
            self.ema_score = (new_score * self.alpha) + (self.ema_score * (1 - self.alpha))
        return self.ema_score


class SessionStateManager:
    def __init__(self):
        self.total_windows = 0
        self.risk_history = []
        self.suspicious_tokens = set()
        self.aggregator = RiskAggregator(alpha=0.35)

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
