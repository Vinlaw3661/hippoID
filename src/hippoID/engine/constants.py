from enum import StrEnum, Enum

class EngineMode(StrEnum):
    stream: str = "stream"
    record: str = "record"
    capture: str = "capture"

class SimilarityThresholds(Enum):
    FACE: float = 0.7
    VOICE: float = 0.8

class RecognitionState(StrEnum):
    UNKNOWN = "Unknown"
    UNDETECTED = "Undetected"
    KNOWN = "Known"

class ElevenLabsConfig(StrEnum):
   VOICE_ID: str = "pqHfZKP75CvOlQylNhV4"
   TTS_MODEL_ID: str = "eleven_multilingual_v2"
   SIMILARITY_BOOST: float = 0.75
   STABILITY: float = 0.5
   USE_SPEAKER_BOOST: bool = True