import os
import logging
from abc import ABC, abstractmethod

# Conditional imports to avoid hard failures if packages aren't installed
try:
    from langchain_ollama import ChatOllama
except ImportError:
    try:
        from langchain_community.chat_models import ChatOllama
    except ImportError:
        ChatOllama = None

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

class LLMBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str, system_instruction: str = None) -> str:
        pass

class OllamaBackend(LLMBackend):
    def __init__(self, model_name: str = "gemma", base_url: str = "http://localhost:11434"):
        if not ChatOllama:
            raise ImportError("langchain_ollama or langchain_community not installed.")
        self.llm = ChatOllama(model=model_name, base_url=base_url, temperature=0.1)
        logger.info(f"Initialized OllamaBackend with model={model_name}")

    def generate(self, prompt: str, system_instruction: str = None) -> str:
        messages = []
        if system_instruction:
            messages.append(SystemMessage(content=system_instruction))
        messages.append(HumanMessage(content=prompt))
        
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            if "Connection refused" in str(e) or "10061" in str(e):
                logger.error("CRITICAL: Could not connect to Ollama. Is 'ollama serve' running?")
                logger.error("Please run 'ollama serve' in a separate terminal.")
            logger.error(f"Ollama generation failed: {e}")
            raise e

class GoogleGenAIBackend(LLMBackend):
    def __init__(self, model_name: str = "gemini-pro", api_key: str = None):
        if not ChatGoogleGenerativeAI:
            raise ImportError("langchain_google_genai not installed.")
        
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found.")
            
        self.llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0.1)
        logger.info(f"Initialized GoogleGenAIBackend with model={model_name}")

    def generate(self, prompt: str, system_instruction: str = None) -> str:
        messages = []
        if system_instruction:
            messages.append(SystemMessage(content=system_instruction))
        messages.append(HumanMessage(content=prompt))
        
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Google GenAI generation failed: {e}")
            raise e

class LLMFactory:
    @staticmethod
    def create_client(provider: str = None) -> LLMBackend:
        # Default to Ollama if not specified, as user asked for Gemma (often local)
        provider = provider or os.getenv("LLM_PROVIDER", "ollama").lower()
        
        if provider == "ollama":
            model = os.getenv("OLLAMA_MODEL", "gemma:2b") # Default small gemma
            return OllamaBackend(model_name=model)
        elif provider == "google":
            model = os.getenv("GOOGLE_MODEL", "gemini-1.5-flash")
            return GoogleGenAIBackend(model_name=model)
        else:
            raise ValueError(f"Unknown LLM Provider: {provider}")
