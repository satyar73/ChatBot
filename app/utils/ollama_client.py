import logging
from typing import Any, List, Optional, Mapping
from langchain.llms.base import LLM
import requests
from pydantic import Field
from dotenv import load_dotenv

from langchain_community.embeddings import OllamaEmbeddings

from app.config.chat_model_config import ChatModelConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class OllamaClientManager(LLM):
    model: str = Field(..., description="The Ollama model to use")
    endpoint: str = Field(default="http://localhost:11434/api/generate")

    def __init__(self, chat_model_config: ChatModelConfig, **kwargs):
        model = chat_model_config.embedding_model
        super().__init__(model=model, **kwargs)
        logger.info(f"Initialized OllamaLLM with model: {self.model}")

    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[Any] = None, **kwargs: Any) -> str:
        try:
            response = requests.post(self.endpoint, json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            })
            response.raise_for_status()
            return response.json()["response"]
        except requests.RequestException as e:
            logger.error(f"Error invoking Ollama LLM: {str(e)}")
            raise

    @property
    def _llm_type(self) -> str:
        return "ollama"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": self.model}

    @classmethod
    def get_embeddings(
            cls,
            chat_model_config: ChatModelConfig,
            cache_key: str = None,
            enable_cache: bool = True
    ) -> OllamaEmbeddings:
        """
        Get or create an OllamaEmbeddings instance with the specified parameters.

        Args:
            chat_model_config: ChatModelConfig with details of the OpenAI config
            cache_key: Optional key to cache and reuse embedding instances
            enable_cache: Whether to enable Portkey caching

        Returns:
            OllamaEmbeddings instance
        """
        embeddings = OllamaEmbeddings(model=chat_model_config.embedding_model)
        return embeddings

class OllamaLLMForJson(OllamaClientManager):
    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[Any] = None, **kwargs: Any) -> str:
        try:
            response = super()._call(prompt, stop, run_manager, **kwargs)
            # TODO: Add logic to ensure JSON output if needed
            return response
        except Exception as e:
            logger.error(f"Error in OllamaLLMForJson: {str(e)}")
            raise

if __name__ == "__main__":
    ollama_llm = OllamaClientManager()