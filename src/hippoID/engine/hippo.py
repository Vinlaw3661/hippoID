from src.hippoID.models.speech.stt import DefaultSTT
from src.hippoID.models.speech.tts import DefaultTTS
from src.hippoID.models.language.llms import DefaultLLM
from src.hippoID.io.utils import IOUtils
from src.hippoID.engine.constants import EngineMode
from src.hippoID.engine.facial_recognition import FacialRecognitionEngine

class Hippo:
    def __init__(self, llm_model = None, tts_model = None, stt_model = None, fre_engine = FacialRecognitionEngine()):
        self.stt = stt_model or DefaultSTT.assemblyai
        self.tts = tts_model or DefaultTTS.elevenlabs
        self.llm = llm_model or DefaultLLM.chatgpt
        self.fre = fre_engine
        
    def run(self):
        pass
    