# PUMA Projekt - Claude Log

## v1.0.0 | 2026-04-02

**Aufgabe:** PET/CT (Ga-68, Brustkrebs) und SPECT/CT (Lunge C1D2) mit PUMA alignen

**Ergebnis:** Erfolgreich abgeschlossen. Ergebnisse in `NM/puma_output/`

### Neue Dateien

- `prepare_puma.py` - Konvertiert DICOMs zu NIfTI und erstellt korrekte Ordnerstruktur fuer PUMA
- `run_puma.py` - PUMA-Wrapper mit Windows-Korrekturen (Monkey-Patching, Pfade)
- `patch_spect_modality.py` - (Hilfsskript, veraltet, Logik in prepare_puma.py integriert)

### Geloeste Probleme

1. Leerzeichen im Projektpfad -> greedy.exe schlaegt ohne Quotes still fehl -> Input nach `C:\pumaz_work\` kopiert
2. MOOSE-Modellpfad braucht Schreibrecht -> Monkey-Patch auf AppData
3. SPECT-DICOM (SOPClass NM) -> dicom2nifti kann ihn nicht konvertieren -> SimpleITK verwendet
4. NumPy 1.x inkompatibel mit SimpleITK 3.0 -> numpy auf 2.4.4 upgegradet
5. Windows-Multiprocessing-Guard fehlt in eigenem Skript hinzugefuegt
