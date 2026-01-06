from src.hippoID.models.speech.constants import APIKeys
import assemblyai as aai

aai.settings.api_key = APIKeys.ASSEMBLYAI_API_KEY.value
ASSEMBLYAI = aai.Transcriber()

class DefaultSTT:
    assemblyai: aai.Transcriber = ASSEMBLYAI