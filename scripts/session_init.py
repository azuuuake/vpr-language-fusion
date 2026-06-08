# scripts/session_init.py

import os
import subprocess
from pathlib import Path

REPOS = {
    "vpr-language-fusion": "https://github.com/azuuuake/vpr-language-fusion",
    "auto_VPR": "https://github.com/gmberton/auto_VPR",
    "LaVPR": "https://github.com/oferidan1/LaVPR",
    "VPR-datasets-downloader": "https://github.com/gmberton/VPR-datasets-downloader",
}

BASE = Path("/content")
DRIVE_BASE = Path("/content/drive/MyDrive/vpr_research")

def run(cmd):
    print(f"\n$ {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def clone_repos():
    os.chdir(BASE)

    for name, url in REPOS.items():
        if not Path(name).exists():
            run(f"git clone {url}")
        else:
            print(f"{name} already exists — skipping clone")

def create_drive_dirs():
    for folder in ["results", "figures", "embeddings", "notes", "evidence"]:
        path = DRIVE_BASE / folder
        path.mkdir(parents=True, exist_ok=True)
        print("Ready:", path)

def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            print("GPU ready:", torch.cuda.get_device_name(0))
            print("VRAM GB:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
        else:
            print("No GPU detected — CPU mode")
    except Exception as e:
        print("Could not check GPU:", e)

if __name__ == "__main__":
    clone_repos()
    create_drive_dirs()
    check_gpu()

    print("\nSession initialisation complete.")
    print("Drive base:", DRIVE_BASE)
