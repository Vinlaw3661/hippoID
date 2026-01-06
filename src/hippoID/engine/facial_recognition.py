from hippoID.utils.computations import cosine_similarity
from src.hippoID.engine.constants import RecognitionState
from src.hippoID.memory.database import LocalVectorCollection
from src.hippoID.io.constants import ImageCaptureFileNames
from hippoID.utils.computations import generate_identifier
from typing import Union, Tuple
from deepface import DeepFace
import os
import cv2
import numpy as np 
import mediapipe as mp
from PIL import Image

class FacialRecognitionEngine:
    def __init__(self, collection = LocalVectorCollection.face):
        self.collection = collection
    
    @staticmethod
    def import_image_as_numpy_array(image_path: str) -> np.ndarray:
        image = Image.open(image_path)
        return np.array(image)
    
    @staticmethod
    def create_face_embedding(image_path: str) -> Union[np.ndarray, None]:
        try:
            embedding = DeepFace.represent(img_path=image_path, model_name='Facenet')[0]["embedding"]
            return embedding
        except Exception:
            return None

    def store_face_embedding(self, image_embedding: np.ndarray, name: str) -> bool:
        
        try:
            uid = generate_identifier()

            self.collection.add(
                embeddings=image_embedding,
                metadatas=[{"name": name}],
                ids=[uid]
            )
        except Exception:
            return False

        return True


    def is_known_face(self, face_embedding: np.ndarray) -> Tuple[bool, str]:

        matches =  self.collection.query(
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
    
    def segment_faces(image: np.ndarray, image_save_path: str = ImageCaptureFileNames.FACE_DETECTION_SAVE_DIRECTORY.value) -> tuple[np.ndarray,str, bool]:
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    
        height, width, _ = image.shape

        results = face_detection.process(image)

        if results.detections:
            detection = results.detections[0]
            bboxC = detection.location_data.relative_bounding_box

            # Convert bounding box to pixel values
            x = int(bboxC.xmin * width)
            y = int(bboxC.ymin * height)
            w = int(bboxC.width * width)
            h = int(bboxC.height * height)

            # Ensure coordinates are within image bounds
            x = max(0, x)
            y = max(0, y)
            w = min(width - x, w)
            h = min(height - y, h)

            face_roi = image[y:y+h, x:x+w]
            ouput_path = f"{image_save_path}/{ImageCaptureFileNames.MASKED_FILE_NAME.value}"
            os.makedirs(os.path.dirname(ouput_path), exist_ok=True)
            cv2.imwrite(ouput_path, face_roi)
            return face_roi, ouput_path, True
        else:

            return np.array([]), RecognitionState.UNDETECTED, False
