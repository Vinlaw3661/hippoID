# src/hippo_id/models/language/llms.py
"""
This module defines the language models used in the Hippo ID project.
It includes configurations for ChatGPT, Claude, and Gemini models.
"""
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from hippo_id.models.language.constants import DefaultModels, DefaultParameters, APIKeys

CHATGPT = ChatOpenAI(
    model = DefaultModels.CHATGPT,
    temperature = DefaultParameters.TEMPERATURE,
    max_tokens = DefaultParameters.MAX_TOKENS,
    timeout = DefaultParameters.TIMEOUT,
    max_retries = DefaultParameters.MAX_RETRIES,
    openai_api_key = APIKeys.OPENAI_API_KEY
)

CLAUDE = ChatAnthropic(
    model = DefaultModels.CLAUDE,
    temperature = DefaultParameters.TEMPERATURE,
    max_tokens = DefaultParameters.MAX_TOKENS,
    timeout = DefaultParameters.TIMEOUT,
    max_retries = DefaultParameters.MAX_RETRIES,
    openai_api_key = APIKeys.OPENAI_API_KEY
)

CLAUDE = ChatAnthropic(
    model = DefaultModels.CLAUDE,
    temperature = DefaultParameters.TEMPERATURE,
    max_tokens = DefaultParameters.MAX_TOKENS,
    timeout = DefaultParameters.TIMEOUT,
    max_retries = DefaultParameters.MAX_RETRIES,
    anthropic_api_key = APIKeys.ANTHROPIC_API_KEY
)

GEMINI = ChatGoogleGenerativeAI(
    model = DefaultModels.GEMINI,
    temperature = DefaultParameters.TEMPERATURE,
    max_output_tokens = DefaultParameters.MAX_TOKENS,
    timeout = DefaultParameters.TIMEOUT,
    max_retries = DefaultParameters.MAX_RETRIES,
    google_api_key = APIKeys.GOOGLE_API_KEY
)

class DefaultLLM:
    chatgpt: ChatOpenAI = CHATGPT
    claude: ChatAnthropic = CLAUDE
    gemini: ChatGoogleGenerativeAI = GEMINI