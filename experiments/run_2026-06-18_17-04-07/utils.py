import re
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, List

# Robust Regex Pattern for filename format: {device}_{patient_id}_{slice_id}.png
FILENAME_PATTERN = re.compile(r"^(?P<device>[^_]+)_(?P<patient_id>[^_]+)_(?P<slice_id>.+)\.png$")

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
        raise ValueError("No valid patient files found matching the expected pattern {device}_{patient_id}_{slice_id}.png")
        
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
