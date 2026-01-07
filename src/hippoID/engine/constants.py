from enum import StrEnum, Enum

class EngineMode(StrEnum):
    stream = "stream"
    record = "record"
    capture = "capture"

class SimilarityThresholds(Enum):
    FACE = 0.7
    VOICE = 0.8

class RecognitionState(StrEnum):
    UNKNOWN = "Unknown"
    UNDETECTED = "Undetected"
    KNOWN = "Known"

class ElevenLabsConfig(Enum):
   VOICE_ID = "pqHfZKP75CvOlQylNhV4"
   TTS_MODEL_ID = "eleven_multilingual_v2"
   SIMILARITY_BOOST = 0.75
   STABILITY = 0.5
   USE_SPEAKER_BOOST = True