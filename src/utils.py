from globals import LOG_DIR
from config import hyperparameters

from pathlib import Path

import os
import datetime

def get_log_dir():
    base_dir = os.path.join(LOG_DIR, "runs")
    time_str = datetime.datetime.now().strftime("%d-%m-%Y--%H-%M-%S")
    
    # Create the params string
    params = "-".join(f"{key}={value}" for key, value in hyperparameters.items())
    
    # Join params and time_str
    run_id = f"{params}--{time_str}"

    return os.path.join(base_dir, run_id)

def file_exists(filepath: str) -> bool:
    return os.path.isfile(filepath)

def directory_exists(file_dir: str) -> bool:
    return os.path.isdir(file_dir)

def list_files(file_dir: str):
    path = Path(file_dir)
    
    return [str(file) for file in path.rglob('*') if file.is_file()]