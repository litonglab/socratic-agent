import os
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    model_id = os.getenv("EMBEDDING_MODEL_ID", "BAAI/bge-m3")
    target_dir = Path(os.getenv("EMBEDDING_MODEL_DIR", "models/bge-m3")).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {model_id} to {target_dir} ...")
    snapshot_download(
        repo_id=model_id,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
    )
    print("Download complete.")


if __name__ == "__main__":
    main()

