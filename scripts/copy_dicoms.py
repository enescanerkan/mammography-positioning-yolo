"""
Copy all labelled DICOM files from the VinDr-Mammo dataset to the project's
raw data directory.

Source  : <VINDR_IMAGES>/<study_uid>/<sop_uid>.dicom
Target  : <PROJECT_ROOT>/data/raw/<study_uid>/<sop_uid>.dicom

All SOPInstanceUIDs that appear in mlo_labels.csv OR cc_labels.csv are
copied — regardless of split (train / val / test).  Files that already
exist at the destination are skipped without re-copying.

Usage
-----
    python scripts/copy_dicoms.py

    # Override the VinDr source directory at the command line:
    python scripts/copy_dicoms.py --vindr_dir "D:/datasets/vindr-mammo/images"
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_VINDR_IMAGES = Path(
    r"C:\Users\Monster\Desktop\vindr-dataset"
    r"\vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0"
    r"\images"
)

RAW_DIR = PROJECT_ROOT / "data" / "raw"
LABELS_DIR = PROJECT_ROOT / "data" / "labels"


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def build_sop_to_study_map(labels_dir: Path) -> dict[str, str]:
    """
    Read both annotation CSV files and return a mapping of
    SOPInstanceUID → StudyInstanceUID for every labelled image.

    Args:
        labels_dir: Directory containing ``mlo_labels.csv`` and
                    ``cc_labels.csv``.

    Returns:
        Dictionary ``{sop_uid: study_uid}``.
    """
    dfs = []
    for name in ("mlo_labels.csv", "cc_labels.csv"):
        p = labels_dir / name
        if not p.exists():
            print(f"[WARNING] Label file not found: {p}")
            continue
        dfs.append(pd.read_csv(p))

    if not dfs:
        raise FileNotFoundError(
            f"No label CSV files found in {labels_dir}. "
            "Place mlo_labels.csv and cc_labels.csv there first."
        )

    df = pd.concat(dfs, ignore_index=True)
    df["SOPInstanceUID"] = df["SOPInstanceUID"].astype(str)
    df["StudyInstanceUID"] = df["StudyInstanceUID"].astype(str)

    # Drop duplicates — a SOP can appear in both MLO and CC CSVs
    df = df[["SOPInstanceUID", "StudyInstanceUID"]].drop_duplicates(subset="SOPInstanceUID")

    mapping = dict(zip(df["SOPInstanceUID"], df["StudyInstanceUID"]))
    print(f"[INFO] Unique labelled images found in CSVs: {len(mapping)}")
    return mapping


def resolve_source(
    sop_uid: str,
    study_uid: str | None,
    vindr_images: Path,
) -> Path | None:
    """
    Locate a DICOM file in the VinDr dataset.

    Tries the direct path ``<vindr_images>/<study_uid>/<sop_uid>.dicom``
    first.  Falls back to a shallow scan of all sub-directories only when
    the direct path fails.

    Args:
        sop_uid: SOPInstanceUID (filename without extension).
        study_uid: StudyInstanceUID (parent folder name), or ``None``.
        vindr_images: Root of the VinDr images directory.

    Returns:
        :class:`pathlib.Path` to the DICOM file, or ``None`` if not found.
    """
    # Fast path — direct lookup using the CSV-provided study UID
    if study_uid:
        candidate = vindr_images / study_uid / f"{sop_uid}.dicom"
        if candidate.exists():
            return candidate

    # Slow fallback — scan all study sub-directories
    for study_dir in vindr_images.iterdir():
        if not study_dir.is_dir():
            continue
        candidate = study_dir / f"{sop_uid}.dicom"
        if candidate.exists():
            return candidate

    return None


def copy_all_dicoms(vindr_images: Path) -> None:
    """
    Copy every labelled DICOM from ``vindr_images`` into ``data/raw/``.

    Args:
        vindr_images: Path to the ``images/`` directory of the
                      VinDr-Mammo dataset.
    """
    if not vindr_images.exists():
        print(f"[ERROR] VinDr images directory not found:\n  {vindr_images}")
        print("  Pass the correct path with: --vindr_dir <path>")
        sys.exit(1)

    sop_to_study = build_sop_to_study_map(LABELS_DIR)
    all_sops = sorted(sop_to_study.keys())

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0   # already exists
    missing = 0
    missing_list: list[str] = []

    print(f"\n[INFO] Copying {len(all_sops)} DICOMs → {RAW_DIR}\n")

    for sop_uid in tqdm(all_sops, desc="Copying DICOMs", unit="file"):
        study_uid = sop_to_study.get(sop_uid)

        dst_dir = RAW_DIR / (study_uid or "unknown")
        dst = dst_dir / f"{sop_uid}.dicom"

        # Skip files that are already in place
        if dst.exists():
            skipped += 1
            continue

        src = resolve_source(sop_uid, study_uid, vindr_images)

        if src is None:
            missing += 1
            missing_list.append(sop_uid)
            continue

        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))
        copied += 1

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    total_in_raw = len(list(RAW_DIR.rglob("*.dicom")))

    print("\n" + "=" * 50)
    print("  COPY COMPLETE")
    print("=" * 50)
    print(f"  Copied          : {copied}")
    print(f"  Already existed : {skipped}")
    print(f"  Not found       : {missing}")
    print(f"  Total in data/raw/: {total_in_raw}")
    print("=" * 50)

    if missing_list:
        print(f"\n[WARNING] {missing} files were not found in the VinDr source:")
        for uid in missing_list[:20]:
            print(f"  {uid}")
        if len(missing_list) > 20:
            print(f"  ... and {len(missing_list) - 20} more.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Copy all labelled DICOMs from the VinDr-Mammo dataset "
            "into data/raw/, preserving the study_uid/sop_uid.dicom structure."
        )
    )
    parser.add_argument(
        "--vindr_dir",
        type=str,
        default=str(DEFAULT_VINDR_IMAGES),
        help=(
            "Path to the VinDr-Mammo 'images/' directory. "
            f"Default: {DEFAULT_VINDR_IMAGES}"
        ),
    )
    args = parser.parse_args()

    copy_all_dicoms(Path(args.vindr_dir))


if __name__ == "__main__":
    main()
