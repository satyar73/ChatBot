from enum import Enum

class LlmProxyType(Enum):
    """
    To run LLM, we can use one of the below as proxies. 
    PORTKEY is a online service that customers can subsribe
    Ollama is a opensource tool / service that will help run various models (e.g. llama) locally (without having to pay for OpenAI)
    """
    PORTKEY = 1,
    OLLAMA = 2

class LlmProxyConfig:
    """
    To run LLM, we can use one of the below as proxies. 
    PORTKEY is a online service that customers can subsribe
    Ollama is a opensource tool / service that will help run various models (e.g. llama) locally (without having to pay for OpenAI)
    """
    def __init__(self, proxy_type: LlmProxyType, url: str, api_key: str = None, cache_ttl: int = 3600):
        self._proxy_type = proxy_type
        self._url = url
        self._api_key = api_key
        self._cache_ttl = cache_ttl #cache settings; used by PORTKEY

    @property
    def proxy_type(self):
        return self._proxy_type

    @property
    def url(self):
        return self._url

    @property
    def api_key(self):
        return self.api_key

    @property
    def cache_ttl(self):
        return self.cache_ttl


