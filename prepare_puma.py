"""
prepare_puma.py
Bereitet eine reine NIfTI-Struktur fuer PUMA vor (keine DICOM-Unterordner).

PUMA erwartet pro Tracer-Ordner genau 1x CT_*.nii und 1x PT_*.nii:
  puma_input/
  +-- PET_Ga68/
  |   +-- CT_pet_ct.nii
  |   +-- PT_pet.nii
  +-- SPECT/
      +-- CT_spect_ct.nii
      +-- PT_spect.nii
"""

import os
import shutil
import tempfile
import dicom2nifti
import SimpleITK as sitk
import pydicom
from pathlib import Path

NM_BASE = Path(__file__).parent / "NM"
OUTPUT_BASE = NM_BASE / "puma_input"

SOURCES = {
    "PET_Ga68": {
        "CT": NM_BASE / "BREAST-3BP-FO-004-01-TRIAL" / "1.2.276.0.7230010.3.1.3.0.2322.1771914173.412559",
        "PT": NM_BASE / "BREAST-3BP-FO-004-01-TRIAL" / "1.2.276.0.7230010.3.1.3.0.2322.1771914173.412047",
    },
    "SPECT": {
        "CT": NM_BASE / "LU-FO-004_C1D02" / "1.2.276.0.7230010.3.1.3.0.1178.1772763304.721362",
        "PT": NM_BASE / "LU-FO-004_C1D02" / "1.2.752.37.54.2769.229375722830778478742950335816754324047",
    },
}

OUTPUT_NAMES = {
    "PET_Ga68": {"CT": "CT_pet_ct.nii", "PT": "PT_pet.nii"},
    "SPECT":    {"CT": "CT_spect_ct.nii", "PT": "PT_spect.nii"},
}


def convert_dicom_dir_to_nifti(src_dir: Path, out_path: Path) -> None:
    """Konvertiert einen DICOM-Ordner via dicom2nifti und benennt das Ergebnis um."""
    with tempfile.TemporaryDirectory() as tmp:
        dicom2nifti.convert_directory(str(src_dir), tmp, compression=False, reorient=True)
        nii_files = list(Path(tmp).glob("*.nii")) + list(Path(tmp).glob("*.nii.gz"))
        if len(nii_files) != 1:
            raise RuntimeError(f"Erwartet 1 NIfTI, gefunden {len(nii_files)} in {src_dir}")
        shutil.copy2(nii_files[0], out_path)


def convert_multiframe_dicom_to_nifti(src_dir: Path, out_path: Path) -> None:
    """Konvertiert einen Multi-Frame-DICOM-Ordner via SimpleITK."""
    dcm_files = list(src_dir.glob("*.dcm"))
    if len(dcm_files) != 1:
        raise RuntimeError(f"Erwartet 1 DICOM, gefunden {len(dcm_files)} in {src_dir}")
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(dcm_files[0]))
    img = reader.Execute()
    sitk.WriteImage(img, str(out_path))


def main():
    # puma_input komplett neu anlegen
    if OUTPUT_BASE.exists():
        print("Loesche alten puma_input-Ordner...")
        shutil.rmtree(str(OUTPUT_BASE))

    for tracer, modalities in SOURCES.items():
        tracer_dir = OUTPUT_BASE / tracer
        tracer_dir.mkdir(parents=True)
        print(f"\n[{tracer}]")

        for modality, src_dir in modalities.items():
            out_name = OUTPUT_NAMES[tracer][modality]
            out_path = tracer_dir / out_name

            dcm_files = list(src_dir.glob("*.dcm"))
            is_multiframe = False
            if len(dcm_files) == 1:
                ds = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)
                sop = str(ds.get("SOPClassUID", ""))
                # Nuclear Medicine Image Storage = multi-frame, muss mit SimpleITK konvertiert werden
                if sop == "1.2.840.10008.5.1.4.1.1.20":
                    is_multiframe = True

            print(f"  {modality}: {len(dcm_files)} DICOM(s) -> {out_name}", end=" ")
            if is_multiframe:
                print("(SimpleITK multi-frame)")
                convert_multiframe_dicom_to_nifti(src_dir, out_path)
            else:
                print("(dicom2nifti)")
                convert_dicom_dir_to_nifti(src_dir, out_path)

    print(f"\nFertig. Input-Ordner: {OUTPUT_BASE}")
    for d in sorted(OUTPUT_BASE.iterdir()):
        if d.is_dir():
            nii = [f.name for f in d.iterdir() if f.suffix == ".nii"]
            print(f"  {d.name}/: {nii}")


if __name__ == "__main__":
    main()
