from dataclasses import dataclass

@dataclass
class EngineMode:
    stream: str = "stream"
    record: str = "record"
    capture: str = "capture"