"""
convert_to_suv.py
Konvertiert das alignierte SPECT-NIfTI (PUMA-Output) zu SUVbw-Werten.

Wiederverwendet _load_dose_meta und _compute_decay aus totalseg/extract_suv.py.

Ausgabe:
  puma_output/aligned_PT/aligned_PT_SPECT_..._spect_GREEN_SUV.nii
  puma_output/aligned_PT/aligned_PT_PET_..._pet_RED_SUV.nii
"""

import sys
import os
from pathlib import Path

# extract_suv.py aus totalseg importieren
TOTALSEG_DIR = Path(__file__).parent.parent / "totalseg"
sys.path.insert(0, str(TOTALSEG_DIR))
from extract_suv import _load_dose_meta, _compute_decay

import pydicom
import numpy as np
import nibabel as nib

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

NM_BASE = Path(__file__).parent / "NM"
PUMA_OUT = NM_BASE / "puma_output" / "aligned_PT"

# Originale DICOM-Quellen (unveraendert)
SPECT_DCM = NM_BASE / "LU-FO-004_C1D02" / "1.2.752.37.54.2769.229375722830778478742950335816754324047"
PET_DCM   = NM_BASE / "BREAST-3BP-FO-004-01-TRIAL" / "1.2.276.0.7230010.3.1.3.0.2322.1771914173.412047"


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def get_calibration_slope_intercept(ds) -> tuple[float, float, str, bool]:
    """Gibt (slope, intercept, einheit, is_rwvms) aus DICOM-Tags zurueck.

    is_rwvms=True  -> RWVMS-Slope, wird von SimpleITK NICHT automatisch angewandt
                      -> muss manuell auf NIfTI-Werte angewandt werden
    is_rwvms=False -> RescaleSlope, wird von SimpleITK automatisch angewandt
                      -> NIfTI-Werte sind bereits kalibriert, NICHT nochmals anwenden
    """
    if hasattr(ds, "RealWorldValueMappingSequence"):
        rwvm = ds.RealWorldValueMappingSequence[0]
        slope     = float(getattr(rwvm, "RealWorldValueSlope", 1.0))
        intercept = float(getattr(rwvm, "RealWorldValueIntercept", 0.0))
        units_seq = getattr(rwvm, "MeasurementUnitsCodeSequence", None)
        units     = units_seq[0].CodeMeaning if units_seq else "unknown"
        print(f"  Kalibrierung via RealWorldValueMappingSequence: x{slope:.6f} + {intercept:.4f} [{units}]")
        return slope, intercept, units, True   # RWVMS -> muss noch angewandt werden
    slope     = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    units     = str(getattr(ds, "Units", "BQML"))
    print(f"  Kalibrierung via RescaleSlope: x{slope:.6f} + {intercept:.4f} [{units}]")
    return slope, intercept, units, False  # SimpleITK hat bereits angewandt


def suv_norm_from_dicom_dir(dcm_dir: Path, modality_label: str) -> tuple[float, float, float, str, bool]:
    """Laedt erstes DICOM aus dcm_dir und gibt (suv_norm_At, slope, intercept, units, is_rwvms) zurueck."""
    dcm_files = sorted(dcm_dir.glob("*.dcm"))
    if not dcm_files:
        raise FileNotFoundError(f"Kein DICOM in {dcm_dir}")

    print(f"\n[{modality_label}] Lade Metadaten aus {dcm_dir.name} ...")
    ds_full = pydicom.dcmread(str(dcm_files[0]))
    slope, intercept, units, is_rwvms = get_calibration_slope_intercept(ds_full)

    ds = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)
    meta, dose_bq, half_life_s, inj_dt_str, weight_g = _load_dose_meta(ds)
    _compute_decay(meta, dose_bq, half_life_s, inj_dt_str, weight_g)

    suv_norm = meta.get("SUV_norm_At")
    if suv_norm is None:
        print(f"  WARNUNG: SUV_norm_At nicht berechenbar (fehlende Metadaten)")
    else:
        print(f"  SUV_norm_At (A(t)/BW) = {suv_norm:.4f} Bq/g")

    return suv_norm, slope, intercept, units, is_rwvms


def find_aligned_nifti(puma_dir: Path, prefix: str) -> Path:
    """Sucht NIfTI-Datei die mit prefix beginnt."""
    matches = sorted(puma_dir.glob(f"{prefix}*.nii*"))
    if not matches:
        raise FileNotFoundError(f"Keine Datei mit Prefix '{prefix}' in {puma_dir}")
    return matches[0]


def nifti_to_suv(nifti_path: Path, slope: float, intercept: float,
                 suv_norm: float, out_path: Path, units: str, is_rwvms: bool) -> None:
    """Laedt NIfTI, rechnet Pixelwerte -> Bq/mL -> SUVbw, speichert Ergebnis.

    is_rwvms=True:  RWVMS-Slope noch nicht angewandt (SimpleITK ignoriert RWVMS)
                    -> Pixelwert * slope + intercept = Bq/mL
    is_rwvms=False: RescaleSlope bereits von SimpleITK angewandt
                    -> NIfTI-Werte sind schon Bq/mL, slope NICHT nochmal anwenden
    """
    nii = nib.load(str(nifti_path))
    data = nii.get_fdata(dtype=np.float32)

    max_val = float(np.nanmax(data))
    print(f"  NIfTI max Rohwert: {max_val:.1f}")

    # Schritt 1: -> Bq/mL
    if is_rwvms:
        print(f"  Wende RWVMS-Kalibrierung an: x{slope:.6f} + {intercept:.4f} -> Bq/mL")
        data = data * slope + intercept
    else:
        print(f"  RescaleSlope bereits von SimpleITK angewandt (NIfTI = Bq/mL)")

    # Schritt 2: Bq/mL -> SUVbw
    if suv_norm and suv_norm > 0:
        suv = (data / suv_norm).astype(np.float32)
        print(f"  SUV-Bereich: {float(np.nanmin(suv)):.2f} - {float(np.nanmax(suv)):.2f}")
    else:
        print("  WARNUNG: Kein SUV_norm - speichere unveraendert")
        suv = data

    nii_out = nib.Nifti1Image(suv, nii.affine, nii.header)
    nii_out.header.set_data_dtype(np.float32)
    nib.save(nii_out, str(out_path))
    print(f"  Gespeichert: {out_path.name}")


# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not PUMA_OUT.exists():
        raise FileNotFoundError(f"PUMA-Output nicht gefunden: {PUMA_OUT}\nBitte zuerst run_puma.py ausfuehren.")

    # -- SPECT --
    spect_suv_norm, spect_slope, spect_intercept, spect_units, spect_is_rwvms = suv_norm_from_dicom_dir(SPECT_DCM, "SPECT")
    spect_nifti = find_aligned_nifti(PUMA_OUT, "aligned_PT_SPECT_")
    spect_out   = PUMA_OUT / (spect_nifti.stem.replace("_GREEN", "").replace("_SUVbw", "") + "_SUVbw.nii")
    nifti_to_suv(spect_nifti, spect_slope, spect_intercept, spect_suv_norm, spect_out, spect_units, spect_is_rwvms)

    # -- PET --
    pet_suv_norm, pet_slope, pet_intercept, pet_units, pet_is_rwvms = suv_norm_from_dicom_dir(PET_DCM, "PET")
    pet_nifti = find_aligned_nifti(PUMA_OUT, "aligned_PT_PET_")
    pet_out   = PUMA_OUT / (pet_nifti.stem.replace("_RED", "").replace("_SUVbw", "") + "_SUVbw.nii")
    nifti_to_suv(pet_nifti, pet_slope, pet_intercept, pet_suv_norm, pet_out, pet_units, pet_is_rwvms)

    print("\nFertig. SUV-Dateien:")
    print(f"  {spect_out}")
    print(f"  {pet_out}")
