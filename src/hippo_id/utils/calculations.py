import numpy as np 
from uuid import uuid4

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def generate_identifier() -> str:
    return str(uuid4())