"""
run_puma.py
Startet PUMA mit schreibbaren Pfaden fuer MOOSE-Modelle.
Setzt MODELS_DIRECTORY_PATH auf einen user-schreibbaren Ordner,
bevor PUMA die Pipeline startet.
"""

import os
import sys

# Schreibbare Pfade fuer PUMA und MOOSE setzen
USER_BIN = os.path.join(os.environ["USERPROFILE"], "AppData", "Local", "pumaz", "bin")
os.environ["PUMAZ_BINARY_PATH"] = USER_BIN
os.makedirs(USER_BIN, exist_ok=True)

# Schreibbaren Modellordner setzen, bevor moosez importiert wird
USER_MODELS = os.path.join(os.environ["USERPROFILE"], "AppData", "Local", "moosez", "nnunet_trained_models")
os.makedirs(USER_MODELS, exist_ok=True)

# moosez.system monkey-patchen
import moosez.system as moose_system
moose_system.MODELS_DIRECTORY_PATH = USER_MODELS
moose_system.ENVIRONMENT_ROOT_PATH = os.path.dirname(USER_MODELS)

# Auch nnunet-Umgebungsvariablen setzen
os.environ["nnUNet_results"] = USER_MODELS

# sys.argv fuer pumaz setzen
subject_dir = r"C:\pumaz_work"
sys.argv = ["pumaz", "-d", subject_dir, "-ir", "none", "-m"]

# PUMA starten (Windows braucht __main__-Guard fuer multiprocessing)
if __name__ == "__main__":
    from pumaz.pumaz import cli
    cli()
