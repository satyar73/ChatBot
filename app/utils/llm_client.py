"""
Centralized LLM client management using Portkey with caching.
"""
from typing import Dict, Any, Optional, Union, List, Type
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import BaseMessage, SystemMessage
from pydantic import BaseModel
import os
import hashlib
import json
import logging
from portkey_ai import Portkey
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class LLMClientManager:
    """
    Centralized manager for LLM clients using Portkey with caching.
    Provides shared access to various LLM clients across the application.
    """

    # Cache for LLM instances
    _llm_instances: Dict[str, Any] = {}
    _embeddings_instances: Dict[str, Any] = {}

    # Portkey client
    _portkey_client = None

    # Local response cache to prevent duplicate requests
    _response_cache: Dict[str, Any] = {}

    @classmethod
    def _get_portkey_client(cls) -> Optional[Portkey]:
        """
        Initialize and return the Portkey client.

        Returns:
            Portkey client instance or None if not configured
        """
        if cls._portkey_client is None:
            api_key = os.getenv("PORTKEY_API_KEY")
            if not api_key:
                logger.warning("PORTKEY_API_KEY not found in environment variables. Using default LLM clients.")
                return None

            # Initialize Portkey client
            cls._portkey_client = Portkey(
                api_key=api_key,
                environment="prod" if os.getenv("ENVIRONMENT", "dev").lower() in ["prod", "production"] else "dev"
            )
            logger.info("Portkey client initialized")

        return cls._portkey_client

    @classmethod
    def _generate_cache_key(cls, messages: List[Dict[str, Any]], model: str, temperature: float) -> str:
        """
        Generate a cache key for a request.

        Args:
            messages: List of message dictionaries
            model: Model name
            temperature: Temperature setting

        Returns:
            Cache key string
        """
        # Create a dictionary of all parameters that affect the response
        cache_dict = {
            "messages": messages,
            "model": model,
            "temperature": temperature
        }

        # Convert to string and hash
        cache_str = json.dumps(cache_dict, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()

    @classmethod
    def get_chat_llm(
            cls,
            model: str = None,
            temperature: float = None,
            streaming: bool = None,
            cache_key: str = None,
            enable_cache: bool = True
    ) -> ChatOpenAI:
        """
        Get or create a ChatOpenAI instance with the specified parameters.
        Uses Portkey if configured.

        Args:
            model: Model name (defaults to config value)
            temperature: Temperature setting (defaults to config value)
            streaming: Streaming setting (defaults to config value)
            cache_key: Optional key to cache and reuse LLM instances
            enable_cache: Whether to enable Portkey caching

        Returns:
            ChatOpenAI instance
        """
        # Get default values from environment or config
        model = model or os.getenv("LLM_MODEL", "gpt-4o")
        temperature = temperature if temperature is not None else float(os.getenv("LLM_TEMPERATURE", "0"))
        streaming = streaming if streaming is not None else os.getenv("LLM_STREAMING", "True").lower() == "true"

        # Create a cache key if none provided
        if not cache_key:
            cache_key = f"chat_{model}_{temperature}_{streaming}_{enable_cache}"

        # Check if instance already exists in cache
        if cache_key in cls._llm_instances:
            return cls._llm_instances[cache_key]

        # Get Portkey client
        portkey_client = cls._get_portkey_client()

        # Create LLM instance
        if portkey_client:
            try:
                # Get cache settings
                cache_ttl = int(os.getenv("PORTKEY_CACHE_TTL", "3600"))

                # Use Portkey as a proxy for OpenAI
                portkey_base_url = "https://api.portkey.ai/v1/proxy"

                # Prepare headers with Portkey API key and cache settings
                portkey_headers = {
                    "x-portkey-api-key": portkey_client.api_key,
                    "x-portkey-mode": "proxy",
                    "x-portkey-provider": "openai",
                    # Explicit cache settings in headers
                    "x-portkey-cache": "true" if enable_cache else "false",
                    "x-portkey-cache-ttl": str(cache_ttl)
                }

                # Create ChatOpenAI instance with Portkey proxy
                llm = ChatOpenAI(
                    model_name=model,
                    temperature=temperature,
                    streaming=streaming,
                    openai_api_base=portkey_base_url,
                    default_headers=portkey_headers
                )

                logger.info(
                    f"Created LLM instance with Portkey, caching {'enabled' if enable_cache else 'disabled'}: {model}")
            except Exception as e:
                logger.error(f"Error creating LLM with Portkey: {str(e)}")
                # Fallback to standard client
                llm = ChatOpenAI(
                    model=model,
                    temperature=temperature,
                    streaming=streaming
                )
                logger.info(f"Fallback - Created standard LLM instance: {model}")
        else:
            # Use standard OpenAI client
            llm = ChatOpenAI(
                model=model,
                temperature=temperature,
                streaming=streaming
            )
            logger.info(f"Created standard LLM instance: {model}")

        # Cache the instance
        cls._llm_instances[cache_key] = llm
        return llm

    @classmethod
    def get_embeddings(
            cls,
            model: str = None,
            dimensions: int = None,
            cache_key: str = None,
            enable_cache: bool = True
    ) -> OpenAIEmbeddings:
        """
        Get or create an OpenAIEmbeddings instance with the specified parameters.

        Args:
            model: Embedding model name (defaults to config value)
            dimensions: Embedding dimensions (defaults to config value)
            cache_key: Optional key to cache and reuse embedding instances
            enable_cache: Whether to enable Portkey caching

        Returns:
            OpenAIEmbeddings instance
        """
        # Get default values from environment or config
        model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        dimensions = dimensions or int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))

        # Create a cache key if none provided
        if not cache_key:
            cache_key = f"embed_{model}_{dimensions}_{enable_cache}"

        # Check if instance already exists
        if cache_key in cls._embeddings_instances:
            return cls._embeddings_instances[cache_key]

        # Get Portkey client
        portkey_client = cls._get_portkey_client()

        # Create embeddings instance
        if portkey_client:
            try:
                # Get cache settings
                cache_ttl = int(os.getenv("PORTKEY_CACHE_TTL", "3600"))

                # Use Portkey as a proxy for OpenAI
                portkey_base_url = "https://api.portkey.ai/v1/proxy"

                # Prepare headers with Portkey API key and cache settings
                portkey_headers = {
                    "x-portkey-api-key": portkey_client.api_key,
                    "x-portkey-mode": "proxy",
                    "x-portkey-provider": "openai",
                    # Explicit cache settings in headers
                    "x-portkey-cache": "true" if enable_cache else "false",
                    "x-portkey-cache-ttl": str(cache_ttl)
                }

                # Configure embeddings with Portkey
                embeddings = OpenAIEmbeddings(
                    model=model,
                    dimensions=dimensions,
                    openai_api_base=portkey_base_url,
                    headers=portkey_headers
                )

                logger.info(
                    f"Created embeddings with Portkey, caching {'enabled' if enable_cache else 'disabled'}: {model}")
            except Exception as e:
                logger.error(f"Error creating embeddings with Portkey: {str(e)}")
                # Fallback to standard client
                embeddings = OpenAIEmbeddings(
                    model=model,
                    dimensions=dimensions
                )
                logger.info(f"Fallback - Created standard embeddings: {model}")
        else:
            # Use standard OpenAI client
            embeddings = OpenAIEmbeddings(
                model=model,
                dimensions=dimensions
            )
            logger.info(f"Created standard embeddings: {model}")

        # Cache the instance
        cls._embeddings_instances[cache_key] = embeddings
        return embeddings

    @classmethod
    def with_structured_output(cls, output_type: Any, enable_cache: bool = True, **llm_kwargs) -> ChatOpenAI:
        """
        Get a ChatOpenAI instance with structured output support.

        Args:
            output_type: The pydantic model or type specification for structured output
            enable_cache: Whether to enable Portkey caching
            **llm_kwargs: Additional keyword arguments for the LLM

        Returns:
            ChatOpenAI instance with structured output capability
        """
        llm = cls.get_chat_llm(enable_cache=enable_cache, **llm_kwargs)
        return llm.with_structured_output(output_type)

    @classmethod
    async def direct_portkey_completion(
            cls,
            prompt: str,
            model: str = None,
            temperature: float = None,
            max_tokens: int = None,
            use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Make a direct completion request using the Portkey client with caching.
        This bypasses LangChain and uses Portkey's API directly.

        Args:
            prompt: The prompt text
            model: Model name (defaults to config value)
            temperature: Temperature setting (defaults to config value)
            max_tokens: Maximum tokens to generate
            use_cache: Whether to use caching

        Returns:
            Raw completion response from Portkey
        """
        # Get base Portkey client
        base_portkey_client = cls._get_portkey_client()
        if not base_portkey_client:
            raise ValueError("Portkey client not configured")

        # Get OpenAI API key
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")

        # Get default values
        model = model or os.getenv("LLM_MODEL", "gpt-4o")
        temperature = temperature if temperature is not None else float(os.getenv("LLM_TEMPERATURE", "0"))
        max_tokens = max_tokens or int(os.getenv("LLM_MAX_TOKENS", "1000"))

        # Create messages from prompt
        messages = [{"role": "user", "content": prompt}]

        # Check local cache if enabled
        if use_cache:
            cache_key = cls._generate_cache_key(messages, model, temperature)
            cached_response = cls._response_cache.get(cache_key)
            if cached_response:
                logger.info("Using cached response")
                return cached_response

        try:
            # Get cache parameters
            cache_ttl = int(os.getenv("PORTKEY_CACHE_TTL", "3600"))

            # Configure the client with provider and cache settings directly
            # AND include the OpenAI API key
            portkey_client = base_portkey_client.with_options(
                mode="proxy",
                provider="openai",
                api_key=openai_api_key,  # Include OpenAI API key
                cache=use_cache,
                cache_ttl=cache_ttl
            ).chat.completions

            logger.info("Configured Portkey client with provider and cache settings")

            # Make request
            response = await portkey_client.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Store in local cache if enabled
            if use_cache:
                cls._response_cache[cache_key] = response

            return response
        except Exception as e:
            logger.error(f"Error in direct Portkey completion: {str(e)}")
            # Additional error information
            logger.error(f"Error type: {type(e).__name__}")
            raise

    @classmethod
    def clear_cache(cls):
        """Clear cached LLM instances and response cache to free up resources."""
        cls._llm_instances.clear()
        cls._embeddings_instances.clear()
        cls._response_cache.clear()
        logger.info("LLM and response caches cleared")