from enum import Enum 

class RecognitionState(Enum):
    UNKNOWN = "unknown"
    UNDETECTED = "undetected"
    RECOGNIZED = "recognized"