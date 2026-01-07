from langchain_anthropic import ChatAnthropic
from src.hippoID.models.language.constants import DefaultModels, DefaultParameters, APIKeys

CLAUDE = ChatAnthropic(
    model = DefaultModels.CLAUDE,
    temperature = DefaultParameters.TEMPERATURE,
    max_tokens = DefaultParameters.MAX_TOKENS,
    timeout = DefaultParameters.TIMEOUT,
    max_retries = DefaultParameters.MAX_RETRIES,
    anthropic_api_key = APIKeys.ANTHROPIC_API_KEY
)

class DefaultLLM:
    claude: ChatAnthropic = CLAUDE