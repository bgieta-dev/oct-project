import os
import numpy as np
from sklearn.model_selection import train_test_split

from typing import Tuple, List

def get_stratified_splits(all_files: List[str], seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Groups files by patient and performs a stratified split based on the OCT device.
    Assumes filename format: {device}_{patient_id}_{slice_id}.png
    """
    patient_to_device = {}
    for f in all_files:
        parts = f.split("_")
        device = parts[0]
        patient = "_".join(parts[:2])
        if patient not in patient_to_device:
            patient_to_device[patient] = device
            
    patients = np.array(list(patient_to_device.keys()))
    devices = np.array(list(patient_to_device.values()))
    
    # First split: Train vs (Val + Test)
    train_pts, temp_pts, _, temp_devs = train_test_split(
        patients, devices, test_size=0.20, random_state=seed, stratify=devices
    )
    
    # Second split: Val vs Test
    val_pts, test_pts = train_test_split(
        temp_pts, test_size=0.50, random_state=seed, stratify=temp_devs
    )
    
    return train_pts, val_pts, test_pts
