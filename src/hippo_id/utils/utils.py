# src/hippo_id/utils/utils.py
import numpy as np 
import os
from uuid import uuid4

def verbose_print(text: str , verbose: bool = True) -> None:
    print(text) if verbose else None

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def ensure_directory_exists(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_identifier() -> str:
    return str(uuid4())