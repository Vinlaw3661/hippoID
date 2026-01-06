# src/hippo_id/models/speech/stt.py
""""
This module defines the speech-to-text (STT) models used in the Hippo ID project.
"""
from hippo_id.models.speech.constants import APIKeys
import assemblyai as aai

aai.settings.api_key = APIKeys.ASSEMBLYAI_API_KEY.value
ASSEMBLYAI = aai.Transcriber()

class DefaultSTT:
    assemblyai: aai.Transcriber = ASSEMBLYAI