import os

BASE_DIR = os.path.abspath(os.path.dirname(os.getcwd()))
LOG_DIR = os.path.abspath(os.path.join(BASE_DIR, "logs"))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "models"))
OUTPUT_DIR = os.path.abspath(os.path.join(BASE_DIR, "out"))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "data"))
FASHION_DATA_DIR = os.path.abspath(os.path.join(DATA_DIR, "fashion"))

# Ensure necessary directories exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)