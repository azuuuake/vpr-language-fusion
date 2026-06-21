"""
run_b1_baseline.py

Runs the EigenPlaces visual-only B1 baseline.

Usage:
    python scripts/run_b1_baseline.py --dataset amstertime
    python scripts/run_b1_baseline.py --dataset msls_val
    python scripts/run_b1_baseline.py --dataset nordland_clean

Expected R@1 / R@5 / R@10:
    amstertime:      43.5 / 63.4 / 70.2
    msls_val:        84.9 / 90.9 / 92.8
    nordland_clean:  52.8 / 76.2 / 87.8
"""

import argparse
import subprocess
import sys


DATASETS = {
    "amstertime": {
        "database": "/content/VPR-datasets-downloader/datasets/amstertime/images/test/database",
        "queries": "/content/VPR-datasets-downloader/datasets/amstertime/images/test/queries",
        "log_dir": "amstertime_eigenplaces_b1",
        "expected": "43.5 / 63.4 / 70.2",
    },
    "msls_val": {
        "database": "/content/VPR-datasets-downloader/datasets/msls_val/images/test/database",
        "queries": "/content/VPR-datasets-downloader/datasets/msls_val/images/test/queries",
        "log_dir": "msls_val_eigenplaces_b1",
        "expected": "84.9 / 90.9 / 92.8",
    },
    "nordland_clean": {
        "database": "/content/VPR-datasets-downloader/datasets/nordland_clean/images/test/database",
        "queries": "/content/VPR-datasets-downloader/datasets/nordland_clean/images/test/queries",
        "log_dir": "nordland_clean_eigenplaces_b1",
        "expected": "52.8 / 76.2 / 87.8",
    },
}


def run(dataset: str, device: str, batch_size: int) -> None:
    cfg = DATASETS[dataset]

    cmd = [
        "python",
        "/content/auto_VPR/main.py",
        "--method=eigenplaces",
        "--backbone=ResNet50",
        "--descriptors_dimension=2048",
        f"--database_folder={cfg['database']}",
        f"--queries_folder={cfg['queries']}",
        f"--device={device}",
        "--image_size",
        "320",
        "320",
        "--batch_size",
        str(batch_size),
        "--num_workers",
        "2",
        "--recall_values",
        "1",
        "5",
        "10",
        "--save_descriptors",
        "--log_dir",
        cfg["log_dir"],
    ]

    print("=" * 70)
    print(f"Running B1 EigenPlaces baseline: {dataset}")
    print(f"Expected R@1 / R@5 / R@10: {cfg['expected']}")
    print("=" * 70)
    print("Command:")
    print(" ".join(cmd))
    print("=" * 70)

    result = subprocess.run(cmd, capture_output=True, text=True)

    print(result.stdout)

    if result.returncode != 0:
        print("ERROR:")
        print(result.stderr)
        sys.exit(result.returncode)

    print("=" * 70)
    print("Finished.")
    print(f"Expected R@1 / R@5 / R@10: {cfg['expected']}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["amstertime", "msls_val", "nordland_clean"],
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch_size", type=int, default=16)

    args = parser.parse_args()

    run(
        dataset=args.dataset,
        device=args.device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
