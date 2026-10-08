import os
import re
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, List

SPLIT_NAMES = ("train", "val", "test")

def split_files(cfg, split: str) -> Tuple[List[str], List[str]]:
    """Sorted (image_paths, mask_paths) of all slices whose patient is in the frozen `split` list."""
    if split not in SPLIT_NAMES:
        raise ValueError(f"split must be one of {SPLIT_NAMES}, got {split!r}")
    pts = set(load_splits(cfg.SPLIT_DIR)[SPLIT_NAMES.index(split)])
    files = [f for f in sorted(os.listdir(cfg.IMG_DIR)) if patient_of(f) in pts]
    img_paths = [os.path.join(cfg.IMG_DIR, f) for f in files]
    mask_paths = [os.path.join(cfg.MASK_DIR, f) for f in files]
    return img_paths, mask_paths

# Robust Regex Pattern for filename format: {device}_{patient_id}_{slice_id}.<extension> (e.g. .png, .tiff)
FILENAME_PATTERN = re.compile(r"^(?P<device>[^_]+)_(?P<patient_id>[^_]+)_(?P<slice_id>.+)\.[a-zA-Z0-9]+$")

def get_stratified_splits(all_files: List[str], seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Groups files by patient and performs a stratified split based on the OCT device (Cirrus, Spectralis, Topcon).
    Ensures that slices from the same patient are never split between training and validation/test sets.
    
    Robustly parses filenames via regex and validates class representation before stratification.
    Maintains backward compatibility by returning patient lists expected by train.py and eval.py.
    
    Args:
        all_files: List of file names in the images directory.
        seed: Random seed for reproducibility.
        
    Returns:
        train_pts, val_pts, test_pts: Arrays of patient identifiers (e.g., 'Spectralis_TRAIN001') for each split.
    """
    patient_to_device = {}
    
    for f in all_files:
        match = FILENAME_PATTERN.match(f)
        if not match:
            continue
            
        device = match.group("device")
        patient_id = match.group("patient_id")
        # Construct full unique patient identifier consistent with train.py and eval.py
        patient = f"{device}_{patient_id}"
        
        if patient not in patient_to_device:
            patient_to_device[patient] = device
            
    if not patient_to_device:
        raise ValueError("No valid patient files found matching the expected pattern {device}_{patient_id}_{slice_id}.<extension>")
        
    patients = np.array(list(patient_to_device.keys()))
    devices = np.array(list(patient_to_device.values()))
    
    # Validate stratification capability (sklearn requires >= 2 samples per class to stratify)
    unique_devices, counts = np.unique(devices, return_counts=True)
    if np.any(counts < 2):
        raise ValueError(
            f"Stratification failed: All device classes must have >= 2 patients. "
            f"Found device counts: {dict(zip(unique_devices, counts))}"
        )
        
    # PHASE 1: Separate Training set from the rest (80% Train, 20% for Val+Test)
    train_pts, temp_pts, _, temp_devs = train_test_split(
        patients, devices, test_size=0.20, random_state=seed, stratify=devices
    )
    
    # PHASE 2: Split the remaining 20% into Validation and Test sets (10% Val, 10% Test total)
    val_pts, test_pts = train_test_split(
        temp_pts, test_size=0.50, random_state=seed, stratify=temp_devs
    )
    
    return train_pts, val_pts, test_pts

def patient_of(filename: str) -> str:
    """Patient identifier ('{device}_{patient_id}') of a slice filename."""
    return "_".join(filename.split("_")[:2])

def write_splits(split_dir, train_pts, val_pts, test_pts) -> None:
    """Writes the three patient lists as one-patient-per-line text files."""
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, pts in (("train", train_pts), ("val", val_pts), ("test", test_pts)):
        (split_dir / f"{name}_patients.txt").write_text("\n".join(str(p) for p in pts) + "\n")

def load_splits(split_dir) -> Tuple[List[str], List[str], List[str]]:
    """Reads the frozen train/val/test patient lists written by `write_splits`."""
    split_dir = Path(split_dir)
    return tuple(
        (split_dir / f"{name}_patients.txt").read_text().split()
        for name in ("train", "val", "test")
    )

if __name__ == "__main__":
    # Regenerate the frozen split files from the current data (seed 42). Only needed once.
    import os
    from octseg.config import load_config
    cfg = load_config()
    write_splits(cfg.SPLIT_DIR, *get_stratified_splits(sorted(os.listdir(cfg.IMG_DIR))))
