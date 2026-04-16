"""
=============================================================================
Scam Detection Dataset Preprocessor
=============================================================================
Combines three data sources into a balanced, DistilBERT-ready dataset:
  1. FTC Robocall Audio Dataset  → metadata.csv  (label=1, scam)
  2. SMS Spam Collection         → spam_ham.txt   (spam=1, ham=0)
  3. DailyDialog                 → dialogues.txt  (label=0, non-scam)

Output files (in ./output/):
  full_dataset.csv       – merged, balanced, shuffled dataset
  train.csv              – 90 % split
  test.csv               – 10 % split
  preprocessing_report.txt – summary statistics

Usage:
  python preprocess_dataset.py \
      --ftc_csv       data/metadata.csv \
      --sms_txt       data/spam_ham.txt \
      --dialog_txt    data/dialogues.txt \
      --output_dir    output/
=============================================================================
"""

import os
import re
import csv
import random
import argparse
import logging
from collections import Counter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Keywords that indicate scam-like language — used to filter non-scam data
SCAM_PATTERNS = [
    # Urgency + threat combos
    r"\burgent(?:ly)?\b.{0,30}\b(call|press|act|respond|verify|confirm)\b",
    r"\bimmediately\b.{0,30}\b(suspended|blocked|arrested|cancelled|frozen)\b",
    r"\byour\s+\w+\s+(will\s+be|has\s+been)\s+(suspended|blocked|cancelled|frozen|terminated)\b",

    # Legal/government impersonation threats
    r"\b(warrant|arrest|lawsuit|legal\s+action)\b.{0,40}\b(issued|filed|pending|against\s+you)\b",
    r"\b(irs|social\s+security\s+administration|ssa|fbi|dea|medicare)\b.{0,50}\b(suspend|arrest|block|cancel)\b",

    # Prize / lottery scams
    r"\b(won|winner|selected|chosen|lucky)\b.{0,40}\b(prize|reward|gift|cash|lottery|sweepstake)\b",
    r"\bcongratulations\b.{0,60}\b(claim|collect|redeem|press\s+\d|call\s+now)\b",

    # OTP / credential harvesting
    r"\b(otp|one.time.pass\w*|verification\s+code|pin)\b.{0,30}\b(share|give|provide|send|read)\b",
    r"\bdo\s+not\s+share\s+(your\s+)?(otp|pin|password|code)\b",  # ironically, legit banks say this

    # Payment pressure
    r"\b(gift\s+card|google\s+play|itunes|amazon\s+card)\b.{0,40}\b(pay|purchase|buy|send)\b",
    r"\b(wire\s+transfer|western\s+union|moneygram|zelle|venmo)\b.{0,30}\b(send|pay|transfer)\b",
    r"\b(bitcoin|crypto|cryptocurrency)\b.{0,40}\b(pay|send|transfer|owe)\b",

    # Threats of immediate consequence
    r"\b(press|dial)\s+\d\b.{0,30}\b(avoid|prevent|stop|speak|talk)\b",
    r"\byour\s+(account|number|ssn|social\s+security)\b.{0,30}\b(compromised|hacked|flagged|suspicious)\b",

    # Refund / overpayment scams
    r"\b(refund|rebate|reimbursement)\b.{0,40}\b(owed|due|process|claim|receive)\b.{0,20}\b(click|call|press|link)\b",
]

COMPILED_SCAM_PATTERNS = [re.compile(p, re.IGNORECASE) for p in SCAM_PATTERNS]


MIN_WORDS = 3          # Drop sentences shorter than this
RANDOM_SEED = 42
TRAIN_RATIO = 0.90


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Normalize whitespace and strip surrounding quotes/spaces."""
    if not isinstance(text, str):
        return ""
    text = text.strip().strip('"').strip("'")
    # Collapse multiple spaces / newlines
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def contains_scam_pattern(text: str) -> bool:
    """
    Returns True only when the text matches a phrase-level scam intent pattern.
    Safe for legitimate sentences like:
      - "I'm calling from Bank of America about your credit limit increase"
      - "You are approved for your loan"
      - "Your account balance is $500"
    """
    return any(pattern.search(text) for pattern in COMPILED_SCAM_PATTERNS)


def word_count(text: str) -> int:
    return len(text.split())


def make_record(text: str, label: int) -> dict:
    return {"text": text, "label": label}


# ---------------------------------------------------------------------------
# Source 1 – FTC Robocall (scam = 1)
# ---------------------------------------------------------------------------

def load_ftc(csv_path: str) -> list[dict]:
    """
    CSV columns: file_name, language, transcript, case_details, case_pdf
    We extract the 'transcript' column (English only) and label as 1.
    """
    records = []
    log.info(f"Loading FTC dataset from: {csv_path}")

    with open(csv_path, newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            lang = row.get("language", "en").strip().lower()
            if lang not in ("en", "english", ""):
                continue  # keep English only
            transcript = clean_text(row.get("transcript", ""))
            if not transcript or word_count(transcript) < MIN_WORDS:
                continue
            records.append(make_record(transcript, 1))

    log.info(f"  → {len(records):,} scam records from FTC")
    return records


# ---------------------------------------------------------------------------
# Source 2 – SMS Spam Collection (spam=1, ham=0)
# ---------------------------------------------------------------------------

def load_sms(txt_path: str) -> tuple[list[dict], list[dict]]:
    """
    TSV format: <label>\\t<message>
    label is 'spam' or 'ham'.
    Returns (spam_records, ham_records).
    """
    spam_records, ham_records = [], []
    log.info(f"Loading SMS dataset from: {txt_path}")

    with open(txt_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            tag, msg = parts[0].strip().lower(), clean_text(parts[1])
            if not msg or word_count(msg) < MIN_WORDS:
                continue
            if tag == "spam":
                spam_records.append(make_record(msg, 1))
            elif tag == "ham":
                if contains_scam_pattern(msg):
                    continue  # filter noisy ham
                ham_records.append(make_record(msg, 0))

    log.info(f"  → {len(spam_records):,} spam  |  {len(ham_records):,} ham from SMS")
    return spam_records, ham_records


# ---------------------------------------------------------------------------
# Source 3 – DailyDialog (non-scam = 0)
# ---------------------------------------------------------------------------

def load_dailydialog(txt_path: str) -> list[dict]:
    """
    DailyDialog format (one dialogue per line):
      utterance1 __eou__ utterance2 __eou__ ...  \\t act \\t emotion \\t topic

    We split on __eou__ to get individual utterances, then label each 0.
    The tab-separated metadata fields at the end are discarded.
    """
    records = []
    log.info(f"Loading DailyDialog from: {txt_path}")

    with open(txt_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            # Strip metadata columns (separated by \\t)
            dialog_part = line.split("\t")[0]
            utterances = dialog_part.split("__eou__")
            for utt in utterances:
                utt = clean_text(utt)
                if not utt or word_count(utt) < MIN_WORDS:
                    continue
                if contains_scam_pattern(utt):
                    continue  # filter label noise
                records.append(make_record(utt, 0))

    log.info(f"  → {len(records):,} non-scam utterances from DailyDialog")
    return records


# ---------------------------------------------------------------------------
# Balancing & Splitting
# ---------------------------------------------------------------------------

def balance(scam: list[dict], non_scam: list[dict], rng: random.Random) -> list[dict]:
    """Downsample the majority class to match the minority class size."""
    n = min(len(scam), len(non_scam))
    log.info(f"Balancing to {n:,} samples per class  (total = {2*n:,})")
    balanced_scam     = rng.sample(scam,     n)
    balanced_non_scam = rng.sample(non_scam, n)
    combined = balanced_scam + balanced_non_scam
    rng.shuffle(combined)
    return combined


def train_test_split(
    dataset: list[dict], ratio: float, rng: random.Random
) -> tuple[list[dict], list[dict]]:
    data = dataset.copy()
    rng.shuffle(data)
    split = int(len(data) * ratio)
    return data[:split], data[split:]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_csv(records: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(records)
    log.info(f"Saved {len(records):,} records → {path}")


def write_report(
    stats: dict, train: list[dict], test: list[dict], out_path: str
) -> None:
    def class_counts(ds):
        c = Counter(r["label"] for r in ds)
        return c[1], c[0]

    tr_scam, tr_ham = class_counts(train)
    te_scam, te_ham = class_counts(test)

    lines = [
        "=" * 60,
        "  SCAM DETECTION — PREPROCESSING REPORT",
        "=" * 60,
        "",
        "── Source breakdown (before balancing) ──────────────────",
        f"  FTC robocall transcripts (scam)   : {stats['ftc']:>8,}",
        f"  SMS spam messages        (scam)   : {stats['sms_spam']:>8,}",
        f"  SMS ham messages         (non-scam): {stats['sms_ham']:>8,}",
        f"  DailyDialog utterances   (non-scam): {stats['dialog']:>8,}",
        "",
        "── After merging ─────────────────────────────────────────",
        f"  Total scam samples                : {stats['total_scam']:>8,}",
        f"  Total non-scam samples            : {stats['total_nonscam']:>8,}",
        "",
        "── After balancing ───────────────────────────────────────",
        f"  Samples per class                 : {stats['per_class']:>8,}",
        f"  Grand total                       : {stats['per_class']*2:>8,}",
        "",
        "── Train / Test split ────────────────────────────────────",
        f"  Train total   : {len(train):,}  (scam={tr_scam:,}, non-scam={tr_ham:,})",
        f"  Test  total   : {len(test):,}   (scam={te_scam:,}, non-scam={te_ham:,})",
        "",
        "── Preprocessing filters applied ─────────────────────────",
        f"  Min word count threshold          : {MIN_WORDS}",
        f"  Scam patterns filtered (non-scam) : {len(SCAM_PATTERNS)} patterns",
        f"  Language filter (FTC)             : English only",
        "=" * 60,
    ]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    log.info(f"Report saved → {out_path}")
    print("\n" + "\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Scam detection dataset preprocessor")
    parser.add_argument("--ftc_csv",    required=True, help="Path to FTC metadata.csv")
    parser.add_argument("--sms_txt",    required=True, help="Path to SMS spam/ham .txt")
    parser.add_argument("--dialog_txt", required=True, help="Path to DailyDialog .txt")
    parser.add_argument("--output_dir", default="output", help="Output directory")
    parser.add_argument("--seed",       type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # ── Load ──────────────────────────────────────────────────────────────
    ftc_records              = load_ftc(args.ftc_csv)
    sms_spam, sms_ham        = load_sms(args.sms_txt)
    dialog_records           = load_dailydialog(args.dialog_txt)

    # ── Merge by class ────────────────────────────────────────────────────
    all_scam    = ftc_records + sms_spam
    all_nonscam = sms_ham + dialog_records

    stats = {
        "ftc":           len(ftc_records),
        "sms_spam":      len(sms_spam),
        "sms_ham":       len(sms_ham),
        "dialog":        len(dialog_records),
        "total_scam":    len(all_scam),
        "total_nonscam": len(all_nonscam),
    }

    log.info(f"Total scam     : {len(all_scam):,}")
    log.info(f"Total non-scam : {len(all_nonscam):,}")

    # ── Balance ───────────────────────────────────────────────────────────
    balanced = balance(all_scam, all_nonscam, rng)
    stats["per_class"] = len(balanced) // 2

    # ── Split ─────────────────────────────────────────────────────────────
    train, test = train_test_split(balanced, TRAIN_RATIO, rng)

    # ── Save ──────────────────────────────────────────────────────────────
    out = args.output_dir
    save_csv(balanced, os.path.join(out, "full_dataset.csv"))
    save_csv(train,    os.path.join(out, "train.csv"))
    save_csv(test,     os.path.join(out, "test.csv"))
    write_report(stats, train, test, os.path.join(out, "preprocessing_report.txt"))

    log.info("Preprocessing complete ✓")


if __name__ == "__main__":
    main()
