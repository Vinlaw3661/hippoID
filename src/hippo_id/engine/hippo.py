from hippo_id.models.speech.stt import DefaultSTT
from hippo_id.models.speech.tts import DefaultTTS
from hippo_id.models.language.llms import DefaultLLM
from hippo_id.io.utils import IOUtils
from hippo_id.engine.constants import EngineMode

class Hippo:
    def __init__(self, llm_model = None, tts_model = None, stt_model = None):
        self.stt = stt_model or DefaultSTT.assemblyai
        self.tts = tts_model or DefaultTTS.elevenlabs
        self.llm = llm_model or DefaultLLM.chatgpt

    def run(self):
        pass
    