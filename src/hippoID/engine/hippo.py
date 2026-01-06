from src.hippoID.models.speech.stt import DefaultSTT
from src.hippoID.models.speech.tts import DefaultTTS
from src.hippoID.models.language.llms import DefaultLLM
from src.hippoID.engine.constants import EngineMode
from src.hippoID.io.utils import IOUtils
from src.hippoID.engine.constants import RecognitionState
from src.hippoID.utils.processing import verbose_print
from src.hippoID.engine.facial_recognition import FacialRecognitionEngine
from src.hippoID.engine.interaction import InteractionEngine
from typing import Tuple

class Hippo:
    def __init__(
            self, 
            llm_model = DefaultLLM.chatgpt, 
            tts_model = DefaultTTS.elevenlabs, 
            stt_model = DefaultSTT.assemblyai, 
            fre_engine = FacialRecognitionEngine(),
            verbose: bool = False
            ):
        self.stt = stt_model 
        self.tts = tts_model 
        self.llm = llm_model 
        self.fre = fre_engine
        self.ire = InteractionEngine(
            tts_model= tts_model,
            llm_model= llm_model,
            stt_model= stt_model
        )
        self.print = verbose_print(verbose=verbose)

    def identify(self, image) -> Tuple[bool, str]: 
        try:
            self.print(f"{'-'*10}STARTING HIPPO IDENTIFICATION PROCESS{'-'*10}")
            self.print("\nSegmenting faces...")

            segmented_face , face_path, result = self.fre.segment_faces(image)

            if not result:
                self.print("No faces detected in the image")
                return False, RecognitionState.UNDETECTED
            
            self.print("\nSegmentation Done!")
            single_face, single_face_path = segmented_face, face_path
            self.print(f"\nIdentifying person in {single_face_path}...")

            is_known, possible_name = self.fre.is_known_face(single_face_path)

            if is_known:
                self.print(f"\nPerson already known! :{possible_name}")
                return False, possible_name
            
            self.print("\nAsking for person's name")
            name_asked = self.ire.ask_for_name(single_face_path)
            self.print("\nName asked successfully!")

            if name_asked:
                self.print(f"\nListening for name...")
                name = self.ire.listen_for_name()
                self.print(f"\nName captured as: {name}")

                self.print(f"\nSaving identity to database...")
                self.fre.store_face_embedding(single_face_path, name)
                
                self.print("\nIdentity saved!")
                self.ire.acknowledge_person(name)
                return True, name
            
            else:
                self.print("\nFailed to ask for name.")
                return False, RecognitionState.UNKNOWN
            
        except Exception as e:
            self.print(f"\nAn error occurred during identification: {e}")
            return False, RecognitionState.UNKNOWN
