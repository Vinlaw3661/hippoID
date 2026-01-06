from hippoID.utils.computations import cosine_similarity
from src.hippoID.engine.constants import RecognitionState
from src.hippoID.memory.database import LocalVectorCollection
from hippoID.utils.computations import generate_identifier
from typing import Union, Tuple
from deepface import DeepFace
import numpy as np 


def import_image_as_numpy_array(image_path: str) -> np.ndarray:
    pass

def store_face_embedding(image_embedding: np.ndarray, name: str, collection = LocalVectorCollection.face) -> bool:
    
    try:
        uid = generate_identifier(size=7)
        
        collection.add(
            embeddings=image_embedding,
            metadatas=[{"name": name}],
            ids=[uid]
        )
    except Exception:
        return False

    return True

def create_face_embedding(img_path: str) -> Union[np.ndarray, None]:
    try:
        embedding = DeepFace.represent(img_path=img_path, model_name='Facenet')[0]["embedding"]
        return embedding
    except ValueError as e:
        return None


def is_known_face(face_embedding: np.ndarray, collection = LocalVectorCollection.face) -> Tuple[bool, str]:

    matches =  collection.query(
                    query_embeddings=face_embedding,
                    n_results=1,
                    include=["embeddings", "metadatas", "distances"]
                    
                )
    
    if len(matches["ids"][0]) == 0:
        return False, RecognitionState.UNKNOWN

    matched_embedding = matches["embeddings"][0][0]
    distance = 1 - cosine_similarity(face_embedding, matched_embedding)

    if distance < 0.55:
        return True, matches["metadatas"][0][0]["name"]
    
    return False, RecognitionState.UNKNOWN

