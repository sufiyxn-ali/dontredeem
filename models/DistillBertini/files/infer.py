"""
=============================================================================
Scam Detection — Inference  (T_t score generator)
=============================================================================
Loads the fine-tuned DistilBERT model and returns a scam probability score
P(scam | text) ∈ [0, 1] — the T_t score in your multimodal system.

Usage (CLI):
  python infer.py --model_dir model/ --text "Your account has been suspended."

Usage (Python API):
  from infer import ScamScorer
  scorer = ScamScorer("model/")
  result = scorer.score("Congratulations! You have won a $500 gift card.")
  print(result)
  # → {'text': '...', 'scam_probability': 0.97, 'label': 'scam', 'confidence': 'high'}

Batch usage:
  results = scorer.score_batch(["text1", "text2", ...])
=============================================================================
"""

import argparse
import json
from dataclasses import dataclass

import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification


# ---------------------------------------------------------------------------
# Confidence bucketing
# ---------------------------------------------------------------------------

def _confidence_label(prob: float) -> str:
    if prob >= 0.90 or prob <= 0.10:
        return "high"
    if prob >= 0.75 or prob <= 0.25:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Scorer class
# ---------------------------------------------------------------------------

@dataclass
class ScamResult:
    text:               str
    scam_probability:   float           # T_t score  ∈ [0, 1]
    label:              str             # "scam" | "non_scam"
    confidence:         str             # "high" | "medium" | "low"
    suspicious_tokens:  list = None     # [(token, attention_weight), ...] if explain=True

    def to_dict(self) -> dict:
        return {
            "text":               self.text,
            "scam_probability":   round(self.scam_probability, 4),
            "label":              self.label,
            "confidence":         self.confidence,
            "suspicious_tokens":  [(t, round(w, 4)) for t, w in (self.suspicious_tokens or [])],
        }

    def __repr__(self):
        tokens_str = f", tokens={len(self.suspicious_tokens)}" if self.suspicious_tokens else ""
        return (
            f"ScamResult(label={self.label!r}, "
            f"prob={self.scam_probability:.4f}, "
            f"confidence={self.confidence!r}{tokens_str})"
        )


class ScamScorer:
    """
    Wrapper around the fine-tuned DistilBERT model.

    Parameters
    ----------
    model_dir : str
        Directory produced by train_distilbert.py (contains config.json,
        pytorch_model.bin / model.safetensors, tokenizer files).
    max_len : int
        Tokenization max length. Should match what was used in training (256).
    threshold : float
        Decision boundary for the "scam" label. Default 0.5.
    device : str | None
        "cuda", "cpu", "mps", or None for auto-detect.
    explain : bool
        If True, extract attention weights to identify suspicious tokens. Default False.
    """

    def __init__(
        self,
        model_dir: str,
        max_len: int = 256,
        threshold: float = 0.5,
        device: str | None = None,
        explain: bool = False,
    ):
        self.max_len   = max_len
        self.threshold = threshold
        self.device    = self._resolve_device(device)
        self.explain   = explain

        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_dir, output_attentions=explain)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _resolve_device(device: str | None) -> torch.device:
        if device:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _encode(self, texts: list[str]) -> dict:
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

    @torch.no_grad()
    def score(self, text: str) -> ScamResult:
        """Score a single text and return a ScamResult."""
        enc = self._encode([text])
        out = self.model(
            input_ids=enc["input_ids"].to(self.device),
            attention_mask=enc["attention_mask"].to(self.device),
        )
        prob = torch.softmax(out.logits, dim=1)[0, 1].item()
        label = "scam" if prob >= self.threshold else "non_scam"
        
        # Extract suspicious tokens if explain=True
        suspicious_tokens = []
        if self.explain and prob > 0.3:  # Only extract if likely scam
            try:
                # Get attention weights from last layer
                attentions = out.attentions  # Tuple of attention tensors
                if attentions:
                    last_layer_attn = attentions[-1]  # Last layer attention
                    # Average attention heads: [batch, heads, seq_len, seq_len]
                    avg_attn = last_layer_attn.mean(dim=1)[0]  # [seq_len, seq_len]
                    
                    # Get attention to [CLS] token (first token)
                    cls_attention = avg_attn[0]  # Attention from CLS to all tokens
                    
                    # Get tokens
                    tokens = self.tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
                    
                    # Map attention to tokens (skip special tokens)
                    token_attention = []
                    for i, (token, attn_weight) in enumerate(zip(tokens, cls_attention.cpu().numpy())):
                        if token not in ["[CLS]", "[SEP]", "[PAD]", "[UNK]"]:
                            # Clean up token (remove ## prefix from wordpiece tokens)
                            clean_token = token.replace("##", "")
                            token_attention.append((clean_token, float(attn_weight)))
                    
                    # Sort by attention weight and get top suspicious tokens
                    suspicious_tokens = sorted(token_attention, key=lambda x: x[1], reverse=True)[:10]
            except Exception as e:
                pass  # Silently skip if attention extraction fails
        
        return ScamResult(
            text=text,
            scam_probability=prob,
            label=label,
            confidence=_confidence_label(prob),
            suspicious_tokens=suspicious_tokens if suspicious_tokens else None,
        )

    @torch.no_grad()
    def score_batch(
        self, texts: list[str], batch_size: int = 32
    ) -> list[ScamResult]:
        """Score a list of texts efficiently in mini-batches."""
        results = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            enc   = self._encode(chunk)
            out   = self.model(
                input_ids=enc["input_ids"].to(self.device),
                attention_mask=enc["attention_mask"].to(self.device),
            )
            probs = torch.softmax(out.logits, dim=1)[:, 1].cpu().tolist()
            
            for j, (text, prob) in enumerate(zip(chunk, probs)):
                label = "scam" if prob >= self.threshold else "non_scam"
                
                # Extract suspicious tokens if explain=True
                suspicious_tokens = []
                if self.explain and prob > 0.3:
                    try:
                        attentions = out.attentions
                        if attentions:
                            last_layer_attn = attentions[-1]  # Last layer
                            avg_attn = last_layer_attn.mean(dim=1)[j]  # For this text in batch
                            cls_attention = avg_attn[0]
                            
                            tokens = self.tokenizer.convert_ids_to_tokens(enc["input_ids"][j])
                            
                            token_attention = []
                            for token, attn_weight in zip(tokens, cls_attention.cpu().numpy()):
                                if token not in ["[CLS]", "[SEP]", "[PAD]", "[UNK]"]:
                                    clean_token = token.replace("##", "")
                                    token_attention.append((clean_token, float(attn_weight)))
                            
                            suspicious_tokens = sorted(token_attention, key=lambda x: x[1], reverse=True)[:10]
                    except Exception as e:
                        pass
                
                results.append(
                    ScamResult(
                        text=text,
                        scam_probability=prob,
                        label=label,
                        confidence=_confidence_label(prob),
                        suspicious_tokens=suspicious_tokens if suspicious_tokens else None,
                    )
                )
        return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Scam probability scorer")
    parser.add_argument("--model_dir",  required=True, help="Path to saved model dir")
    parser.add_argument("--text",       default=None,  help="Single text to score")
    parser.add_argument("--file",       default=None,  help="File with one text per line")
    parser.add_argument("--threshold",  type=float, default=0.5)
    parser.add_argument("--max_len",    type=int,   default=256)
    parser.add_argument("--json_out",   default=None, help="Write results to JSON file")
    args = parser.parse_args()

    scorer = ScamScorer(
        model_dir=args.model_dir,
        max_len=args.max_len,
        threshold=args.threshold,
    )

    if args.text:
        result = scorer.score(args.text)
        print(json.dumps(result.to_dict(), indent=2))

    elif args.file:
        with open(args.file, encoding="utf-8") as fh:
            texts = [line.strip() for line in fh if line.strip()]
        results = scorer.score_batch(texts)
        output  = [r.to_dict() for r in results]
        if args.json_out:
            with open(args.json_out, "w") as fh:
                json.dump(output, fh, indent=2)
            print(f"Results written to {args.json_out}")
        else:
            print(json.dumps(output, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
