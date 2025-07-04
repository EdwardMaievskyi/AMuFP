from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic

from src.utils.logger import get_logger

# Initialize logger at module level
logger = get_logger("llm_factory")


def get_llm(model_provider: str,
            API_KEY: str,
            model_name: str):
    """Factory function to get the appropriate LLM based on the provider."""
    logger.info("Creating LLM instance for provider: %s, model: %s",
                model_provider, model_name)

    if model_provider == "openai":
        # Ensure OPENAI_API_KEY is set in your environment
        return ChatOpenAI(api_key=API_KEY,
                          model=model_name, temperature=0)
    elif model_provider == "google":
        # Ensure GOOGLE_API_KEY is set in your environment
        # gemini-pro-vision is suitable for multimodal inputs (text and image)
        return ChatGoogleGenerativeAI(model=model_name, temperature=0,
                                      api_key=API_KEY)
    elif model_provider == "anthropic":
        # Ensure ANTHROPIC_API_KEY is set in your environment
        # claude-3-opus-20240229 is a powerful multimodal model
        return ChatAnthropic(api_key=API_KEY,
                             model=model_name,
                             temperature=0)
    else:
        logger.error("Unsupported model provider: %s", model_provider)
        raise ValueError(f"Unsupported model provider: {model_provider}")
