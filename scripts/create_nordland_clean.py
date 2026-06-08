"""
create_nordland_clean.py

Creates the Nordland-clean aligned 400-pair summer-winter subset.

This script assumes that raw Nordland frames have already been extracted into:

    /content/drive/MyDrive/vpr_research/nordland_raw/summer_raw/
    /content/drive/MyDrive/vpr_research/nordland_raw/winter_raw/

The cleaned subset uses the final verified alignment:

    summer database start index = 1900
    winter query start index    = 2000
    number of pairs             = 400
    coordinate spacing          = 1 metre
    positive threshold          = 25 metres = ±25 frames

Output:

    /content/drive/MyDrive/vpr_research/nordland_clean_subset/images/test/database/
    /content/drive/MyDrive/vpr_research/nordland_clean_subset/images/test/queries/

The output filenames use gmberton-style synthetic coordinate labels so that
auto_VPR can evaluate the subset using its standard 25m threshold.
"""

import argparse
import shutil
from pathlib import Path


DRIVE_BASE = Path("/content/drive/MyDrive/vpr_research")

DEFAULT_SUMMER_RAW = DRIVE_BASE / "nordland_raw" / "summer_raw"
DEFAULT_WINTER_RAW = DRIVE_BASE / "nordland_raw" / "winter_raw"
DEFAULT_OUTPUT = DRIVE_BASE / "nordland_clean_subset"

SUMMER_START = 1900
WINTER_START = 2000
N_PAIRS = 400
COORD_SPACING_METRES = 1


def list_images(folder: Path):
    exts = ["*.jpg", "*.jpeg", "*.png"]
    files = []
    for ext in exts:
        files.extend(folder.glob(ext))
    return sorted(files)


def make_gmberton_filename(index: int) -> str:
    """
    Creates a synthetic-coordinate filename compatible with gmberton/auto_VPR.

    The second field acts as the synthetic coordinate.
    With 1m spacing, image i has coordinate i.
    With auto_VPR 25m threshold, positives correspond to ±25 frames.
    """
    coord = index * COORD_SPACING_METRES
    return f"@0@{coord}@@@@@{index:06d}@@@@@@@@.jpg"


def create_subset(
    summer_raw: Path,
    winter_raw: Path,
    output_dir: Path,
    summer_start: int,
    winter_start: int,
    n_pairs: int,
):
    summer_files = list_images(summer_raw)
    winter_files = list_images(winter_raw)

    if not summer_files:
        raise FileNotFoundError(f"No summer frames found in {summer_raw}")

    if not winter_files:
        raise FileNotFoundError(f"No winter frames found in {winter_raw}")

    if summer_start + n_pairs > len(summer_files):
        raise ValueError(
            f"Not enough summer frames: need index {summer_start + n_pairs}, "
            f"but only found {len(summer_files)} frames."
        )

    if winter_start + n_pairs > len(winter_files):
        raise ValueError(
            f"Not enough winter frames: need index {winter_start + n_pairs}, "
            f"but only found {len(winter_files)} frames."
        )

    database_dir = output_dir / "images" / "test" / "database"
    queries_dir = output_dir / "images" / "test" / "queries"

    database_dir.mkdir(parents=True, exist_ok=True)
    queries_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Creating Nordland-clean subset")
    print("=" * 70)
    print(f"Summer raw folder: {summer_raw}")
    print(f"Winter raw folder: {winter_raw}")
    print(f"Output folder:     {output_dir}")
    print(f"Summer start:      {summer_start}")
    print(f"Winter start:      {winter_start}")
    print(f"Number of pairs:   {n_pairs}")
    print("=" * 70)

    for i in range(n_pairs):
        summer_src = summer_files[summer_start + i]
        winter_src = winter_files[winter_start + i]

        filename = make_gmberton_filename(i)

        summer_dst = database_dir / filename
        winter_dst = queries_dir / filename

        shutil.copy2(summer_src, summer_dst)
        shutil.copy2(winter_src, winter_dst)

    print("Done.")
    print(f"Database images: {len(list_images(database_dir))}")
    print(f"Query images:    {len(list_images(queries_dir))}")
    print()
    print("Subset saved to:")
    print(output_dir)
    print()
    print("Expected B1 result after auto_VPR EigenPlaces evaluation:")
    print("R@1 / R@5 / R@10 = 52.8 / 76.2 / 87.8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summer_raw", type=Path, default=DEFAULT_SUMMER_RAW)
    parser.add_argument("--winter_raw", type=Path, default=DEFAULT_WINTER_RAW)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summer_start", type=int, default=SUMMER_START)
    parser.add_argument("--winter_start", type=int, default=WINTER_START)
    parser.add_argument("--n_pairs", type=int, default=N_PAIRS)

    args = parser.parse_args()

    create_subset(
        summer_raw=args.summer_raw,
        winter_raw=args.winter_raw,
        output_dir=args.output_dir,
        summer_start=args.summer_start,
        winter_start=args.winter_start,
        n_pairs=args.n_pairs,
    )


if __name__ == "__main__":
    main()
