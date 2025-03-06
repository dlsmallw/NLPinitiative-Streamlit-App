# Used for setting some constants for the project codebase

from pathlib import Path

# Root Path
ROOT = Path(__file__).resolve().parents[0]

# Model Directory
MODELS_DIR = ROOT / 'models'

# Binary Classification Model Path
BIN_MODEL_PATH = MODELS_DIR / 'binary_classification'

# Multilabel Regression Model Path
ML_MODEL_PATH = MODELS_DIR / 'multilabel_regression'

# HF Hub Repositories
BIN_REPO = 'dlsmallw/Binary-Classification-testing'
ML_REPO = 'dlsmallw/Multilabel-Regression-testing'

BIN_API_URL = f"https://api-inference.huggingface.co/models/{BIN_REPO}"
ML_API_URL = f"https://api-inference.huggingface.co/models/{ML_REPO}"