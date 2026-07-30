"""Fetch the default model (TinyLlama-1.1B-Chat Q4_K_M, ~638 MB)."""

import sys

from huggingface_hub import hf_hub_download

REPO = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
FILE = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"


def main() -> int:
    path = hf_hub_download(repo_id=REPO, filename=FILE, local_dir="models")
    print(f"model ready: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
