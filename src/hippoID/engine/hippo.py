from src.hippoID.models.speech.stt import DefaultSTT
from src.hippoID.models.speech.tts import DefaultTTS
from src.hippoID.models.language.llms import DefaultLLM
from src.hippoID.engine.constants import EngineMode
from src.hippoID.io.utils import IOUtils
from src.hippoID.engine.facial_recognition import FacialRecognitionEngine
from src.hippoID.engine.interaction import InteractionEngine

class Hippo:
    def __init__(
            self, 
            llm_model = DefaultLLM.chatgpt, 
            tts_model = DefaultTTS.elevenlabs, 
            stt_model = DefaultSTT.assemblyai, 
            fre_engine = FacialRecognitionEngine(),
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

    def identify(self):
        pass
    