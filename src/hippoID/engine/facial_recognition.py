from src.hippoID.utils.computations import cosine_similarity
from src.hippoID.engine.constants import RecognitionState
from src.hippoID.memory.database import LocalVectorCollection
from src.hippoID.io.constants import ImageCaptureFileNames
from src.hippoID.utils.computations import generate_identifier
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
    
    @staticmethod
    def segment_faces(image: np.ndarray, image_save_path: str = ImageCaptureFileNames.FACE_DETECTION_SAVE_DIRECTORY.value) -> tuple[np.ndarray,str, bool]:
        face_detector =  mp.tasks.vision.FaceDetector.create_from_options(
            mp.tasks.vision.FaceDetectorOptions(
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                min_detection_confidence=0.5,
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=os.path.abspath("src/hippoID/models/vision/blaze_face_short_range.tflite")
                    )
            )
        )
    
        # Convert BGR → RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        results = face_detector.detect(mp_image)

        if results.detections:
            detection = results.detections[0]
            bbox= detection.bounding_box

            h, w, _ = image.shape
            x, y, bw, bh = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height

            # Convert bounding box to pixel values
            x = max(0, x)
            y = max(0, y)
            bw = min(w - x, bw)
            bh = min(h - y, bh)

            face_roi = image[y:y+bh, x:x+bw]

            ouput_path = f"{image_save_path}/{ImageCaptureFileNames.MASKED_FILE_NAME.value}"
            os.makedirs(os.path.dirname(ouput_path), exist_ok=True)
            cv2.imwrite(ouput_path, face_roi)
            return face_roi, ouput_path, True
        else:
            return np.array([]), "", False
