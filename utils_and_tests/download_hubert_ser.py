"""Download the HuBERT speech emotion recognition model for local inference."""

from pathlib import Path

from huggingface_hub import snapshot_download


ROOT_DIR = Path(__file__).resolve().parents[1]
TARGET_DIR = ROOT_DIR / "models" / "hubert-large-superb-er"


def main() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="superb/hubert-large-superb-er",
        local_dir=str(TARGET_DIR),
        local_dir_use_symlinks=False,
        ignore_patterns=["*.h5", "*.msgpack", "*.onnx", "*.tflite"],
    )
    print(f"HuBERT SER model downloaded to {TARGET_DIR}")


if __name__ == "__main__":
    main()
