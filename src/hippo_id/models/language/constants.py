# src/hippo_id/models/language/constants.py
"""
This module defines constants used in the Hippo ID language models.
It includes default model names, parameters, and API keys.
"""
import os
from dotenv import load_dotenv

load_dotenv()

class ModelNames():
    CLAUDE_SONNET_3_7 = "claude-3-7-sonnet-latest"
    CLAUDE_OPUS_4 = "claude-opus-4-20250514"
    CHATGPT_4_1 = "gpt-4-1"
    CHATGPT_4_o = "gpt-4o"
    GEMINI_2_0_FLASH = "gemini-2.0-flash"
    GEMINI_PRO = "gemini-pro"

class DefaultParameters:
    TEMPERATURE = 0.4
    MAX_TOKENS = 1024
    TIMEOUT = None
    MAX_RETRIES = 3

class DefaultModels:
    CLAUDE = ModelNames.CLAUDE_SONNET_3_7
    CHATGPT = ModelNames.CHATGPT_4_1
    GEMINI = ModelNames.GEMINI_2_0_FLASH

class APIKeys:
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
