import os
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, List

def get_stratified_splits(all_files: List[str], seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Groups files by patient and performs a stratified split based on the OCT device (Cirrus, Spectralis, Topcon).
    Ensures that slices from the same patient are never split between training and validation/test sets.
    
    Assumes filename format: {device}_{patient_id}_{slice_id}.png
    
    Returns:
        train_pts, val_pts, test_pts: Arrays of patient identifiers for each split.
    """
    patient_to_device = {}
    for f in all_files:
        # Expected format: "Spectralis_TRAIN001_001.png" -> device="Spectralis", patient="Spectralis_TRAIN001"
        parts = f.split("_")
        device = parts[0]
        patient = "_".join(parts[:2])
        if patient not in patient_to_device:
            patient_to_device[patient] = device
            
    patients = np.array(list(patient_to_device.keys()))
    devices = np.array(list(patient_to_device.values()))
    
    # PHASE 1: Separate Training set from the rest (80% Train, 20% for Val+Test)
    # Stratification by 'device' ensures balanced representation of hardware manufacturers across splits.
    train_pts, temp_pts, _, temp_devs = train_test_split(
        patients, devices, test_size=0.20, random_state=seed, stratify=devices
    )
    
    # PHASE 2: Split the remaining 20% into Validation and Test sets (10% Val, 10% Test total)
    val_pts, test_pts = train_test_split(
        temp_pts, test_size=0.50, random_state=seed, stratify=temp_devs
    )
    
    return train_pts, val_pts, test_pts
