from elevenlabs.client import ElevenLabs
from src.hippoID.models.speech.constants import APIKeys

ELEVENLABS = ElevenLabs(
    api_key=APIKeys.ELEVENLABS_API_KEY.value,
)

class DefaultTTS:
    elevenlabs: ElevenLabs = ELEVENLABS