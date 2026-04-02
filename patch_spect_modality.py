"""
patch_spect_modality.py
Aendert den Modality-Tag der SPECT-DICOM-Kopie in puma_input/SPECT/ von NM auf PT,
damit PUMA die Datei als PET-Datei akzeptiert.
Die Originaldaten in NM/LU-FO-004_C1D02/ bleiben unveraendert.
"""

import pydicom
from pathlib import Path

SPECT_DIR = Path(__file__).parent / "NM" / "puma_input" / "SPECT"

patched = 0
skipped = 0

for dcm_path in SPECT_DIR.glob("*.dcm"):
    ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=False)
    modality = str(ds.get("Modality", ""))
    if modality == "NM":
        ds.Modality = "PT"
        ds.save_as(str(dcm_path))
        patched += 1
    else:
        skipped += 1

print(f"Patched: {patched} NM->PT  |  Skipped (already OK): {skipped}")
