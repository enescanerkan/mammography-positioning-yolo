"""
Copy 146 MLO + 146 CC test DICOMs from vindr-dataset to data/raw/
Preserves vindr directory structure: data/raw/{study_uid}/{sop_uid}.dicom
"""
import pandas as pd
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
VINDR_IMAGES = Path(
    r"C:\Users\Monster\Desktop\vindr-dataset"
    r"\vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0"
    r"\images"
)
RAW_DIR = PROJECT_ROOT / "data" / "raw"

def main():
    mlo_meta = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "advanced" / "MLO" / "metadata.csv")
    cc_meta = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "advanced" / "CC" / "metadata.csv")
    
    mlo_labels = pd.read_csv(PROJECT_ROOT / "data" / "labels" / "mlo_labels.csv")
    cc_labels = pd.read_csv(PROJECT_ROOT / "data" / "labels" / "cc_labels.csv")
    all_labels = pd.concat([mlo_labels, cc_labels], ignore_index=True)
    all_labels['SOPInstanceUID'] = all_labels['SOPInstanceUID'].astype(str)
    
    sop_to_study = dict(zip(all_labels['SOPInstanceUID'], all_labels['StudyInstanceUID'].astype(str)))
    
    mlo_test = mlo_meta[mlo_meta['split'] == 'test'].copy()
    mlo_test = mlo_test[~mlo_test['sop_uid'].astype(str).str.contains('_flip', na=False)]
    
    cc_test = cc_meta[cc_meta['split'] == 'test'].copy()
    cc_test = cc_test[~cc_test['sop_uid'].astype(str).str.contains('_flip', na=False)]
    
    print(f"MLO test images: {len(mlo_test)}")
    print(f"CC  test images: {len(cc_test)}")
    
    all_sop_uids = set(mlo_test['sop_uid'].astype(str).tolist() + cc_test['sop_uid'].astype(str).tolist())
    print(f"Unique SOPInstanceUIDs to copy: {len(all_sop_uids)}")
    
    copied = 0
    not_found = 0
    already_exists = 0
    
    for sop_uid in sorted(all_sop_uids):
        study_uid = sop_to_study.get(sop_uid)
        
        if study_uid:
            src = VINDR_IMAGES / study_uid / f"{sop_uid}.dicom"
        else:
            src = None
        
        if src is None or not src.exists():
            for study_dir in VINDR_IMAGES.iterdir():
                if not study_dir.is_dir():
                    continue
                candidate = study_dir / f"{sop_uid}.dicom"
                if candidate.exists():
                    src = candidate
                    study_uid = study_dir.name
                    break
        
        if src is None or not src.exists():
            print(f"  NOT FOUND: {sop_uid}")
            not_found += 1
            continue
        
        dst_dir = RAW_DIR / study_uid
        dst = dst_dir / f"{sop_uid}.dicom"
        
        if dst.exists():
            already_exists += 1
            continue
        
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))
        copied += 1
    
    print(f"\nDone!")
    print(f"  Copied:        {copied}")
    print(f"  Already exist: {already_exists}")
    print(f"  Not found:     {not_found}")
    print(f"  Total target:  {len(all_sop_uids)}")
    
    actual_files = list(RAW_DIR.rglob("*.dicom"))
    print(f"  Files in data/raw/: {len(actual_files)}")

if __name__ == "__main__":
    main()
