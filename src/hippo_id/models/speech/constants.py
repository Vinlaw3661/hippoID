# src/hippo_id/models/speech/constants.py
"""
This module defines constants used in the Hippo ID speech models.
It includes default model names, parameters, and API keys.
"""
import os
from dotenv import load_dotenv
from enum import Enum
load_dotenv()

class APIKeys(Enum):
    ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

class DefaultParameters(Enum):
    VOICE_ID = "pqHfZKP75CvOlQylNhV4"
    USE_SPEAKER_BOOST = True
    SIMILARITY_BOOST = 0.75
    STABILITY = 0.5

class DefaultTTSModels(Enum):
    ELEVENLABS_MULTILINGUAL_V2 = "eleven_multilingual_v2"

class DefaultSTTModels(Enum):
    pass