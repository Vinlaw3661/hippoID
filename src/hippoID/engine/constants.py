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