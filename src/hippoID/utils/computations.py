import numpy as np 
from nanoid import generate

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def generate_identifier(size: int = 7) -> str:
    return generate(size=size)