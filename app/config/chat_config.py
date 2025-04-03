import os
from typing import Dict
from dotenv import load_dotenv
from app.config.chat_model_config import ChatModelConfig, CloudProvider
from app.config.llm_proxy_config import LlmProxyConfig, LlmProxyType
from app.config.vector_store_config import PineconeConfig
from app.utils.other_utlis import load_feature_flags

class ChatConfig:
    """Class to manage all configuration settings for the application"""

    def __init__(self):
        # Load environment variables
        load_dotenv()

        # Load feature flags
        self.CHAT_FEATURE_FLAGS = load_feature_flags("chat")

        # Company information
        self.COMPANY_NAME = os.getenv("COMPANY_NAME", "SVV")

        self.OUTPUT_DIR = "data"

        # Google Drive API Settings
        self.GOOGLE_DRIVE_CREDENTIALS_FILE = os.getenv("GOOGLE_DRIVE_CREDENTIALS_FILE")
        self.GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID", None)  # Optional, can be None to use root
        self.GOOGLE_DRIVE_RECURSIVE = True  # Process subfolders recursively

        # Google Slides Enhanced Processing
        self.USE_ENHANCED_SLIDES = os.getenv("USE_ENHANCED_SLIDES", "false").lower() == "true"
        self.OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")
        self.VISION_MAX_TOKENS = int(os.getenv("VISION_MAX_TOKENS", "4000"))
        self.MAX_SLIDES_TO_PROCESS = int(os.getenv("MAX_SLIDES_TO_PROCESS", "50"))

        # Google Drive specific file paths
        self.DRIVE_DOCUMENTS_FILE = os.path.join(self.OUTPUT_DIR, "drive_documents.json")
        self.DRIVE_PROCESSED_FILE = os.path.join(self.OUTPUT_DIR, "drive_processed.json")

        # Shopify API Settings
        self.SHOPIFY_API_VERSION = "2024-04"
        self.SHOPIFY_API_KEY = os.getenv("SHOPIFY_API_KEY")
        self.SHOPIFY_STORE = os.getenv("SHOPIFY_STORE")
        self.SHOPIFY_SHOP_DOMAIN = os.getenv("SHOPIFY_SHOP_DOMAIN")
        self.SHOPIFY_SITE_BASE_URL = os.getenv("SHOPIFY_SITE_BASE_URL")

        # API Limits
        self.BLOG_FETCH_LIMIT = 250
        self.ARTICLE_FETCH_LIMIT = 250
        self.PRODUCT_FETCH_LIMIT = 250

        # Pinecone Settings
        self.PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        self.PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
        self.PINECONE_DIMENSION = 1536  # Changed from 512 to 1536 to match existing index
        self.PINECONE_CLOUD = os.getenv("PINECONE_CLOUD")
        self.PINECONE_REGION = os.getenv("PINECONE_REGION")

        # Ollama Settings
        self.OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
        self.OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
        self.OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_MODEL")
        self.OLLAMA_PINECONE_INDEX_NAME = os.getenv("OLLAMA_PINECONE_INDEX_NAME")

        # OpenAI Settings
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.OPENAI_SUMMARY_MODEL = "gpt-3.5-turbo"
        self.EMBEDDING_CONTEXT_LENGTH = 8192

        # Processing Settings
        self.SUMMARIZE_CONTENT = False  # Set to True if you want to summarize content
        self.SUMMARIZE_THRESHOLD = 8192  # Set to True if you want to summarize content
        self.SAVE_INTERMEDIATE_FILES = True  # Save JSON files during processing
        self.CHUNK_SIZE = 256  # Reduced from 600 for more precise retrieval
        self.CHUNK_OVERLAP = 40  # Adjusted to maintain ~15% overlap with new chunk size
        self.QA_SOURCE_FILE = "app/services/qagold.txt"
        self.QA_SOURCE_FILE_JSON = "app/services/qagold.json"

        # File paths

        self.BLOGS_FILE = os.path.join(self.OUTPUT_DIR, "blogs.json")
        self.BLOGS_PROCESSED_FILE = os.path.join(self.OUTPUT_DIR, "blogs_processed.json")
        self.PRODUCTS_FILE = os.path.join(self.OUTPUT_DIR, "products.json")
        self.PRODUCTS_PROCESSED_FILE = os.path.join(self.OUTPUT_DIR, "products_processed.json")
        self.COMBINED_FILE = os.path.join(self.OUTPUT_DIR, f"{self.COMPANY_NAME.lower()}_combined.json")

        # API Settings
        self.API_HOST = "0.0.0.0"
        self.API_PORT = 8000

        # Configuration settings for chat functionality
        # LLM Configuration
        self.LLM_CONFIG_4o = {
            "temperature": 0,
            "streaming": True,
            "model": "gpt-4o"
        }

        self.chat_model_configs: Dict[str, ChatModelConfig] = self._get_model_configs()

        # Retriever Configuration
        self.RETRIEVER_CONFIG = {
            "search_type": "mmr",
            "k": 5,
            "fetch_k": 20,
            "lambda_mult": 0.7
        }

        # System Prompts
        self.RAG_SYSTEM_PROMPT = f"""
            You are a chatbot answering questions about {self.COMPANY_NAME}, a community of analytics and marketing
            professionals focused on making marketing attribution accessible, affordable, and effective.

            Response Guidelines
                - Answer Directly & Clearly
                    - Begin with a direct answer before adding context.
                    - Define technical concepts before expanding on them.
                    - Cover all key terms mentioned in the question.
                - Ensure Accuracy & Completeness
                    - Use precise terminology from source documents.
                    - Preserve numerical data and statistical details exactly.
                    - Incorporate multiple perspectives if sources differ.
                    - If information is missing, acknowledge it rather than fabricate details
                - Response Structure
                    - Keep responses concise (2-3 paragraphs).
                    - Use bullet points for clarity when listing information.
                    - Prioritize essential insights over exhaustive details.
                    - Avoid repeating the same information.
                - Source Linking
                    - Always provide a relevant source link: Learn more: Title.
                    - Do not generate links that aren't in the source material.
                - Content Boundaries
                    - Share only {self.COMPANY_NAME}-specific data.
                    - Do not explain or generate code.
                    - For pricing, direct users to the product page.
                    - For time-sensitive info, direct users to the masterclass page.
                - Handling Technical & Attribution Topics
                    - Include attribution methodologies (MMM, Geo-testing, incrementality testing) when referenced.
                    - Explain technical terms in practical marketing applications.
                    - Provide step-by-step guidance for processes if available.
                    - Highlight biases in platform-specific attribution when relevant.
                - Special Cases
                    - For budget allocation, include this disclaimer:
                            "For optimal results, we recommend consulting with {self.COMPANY_NAME} experts to discuss your
                             specific needs before making allocation decisions."
                    - If a term is not in the source, provide the best explanation based on related concepts.
                - Final Principles
                    - Maintain the original meaning of source material.
                    - When in doubt, prioritize completeness of key concepts over brevity.
                    - Ensure responses are conversational, clear, and informative.
        """

        self.NON_RAG_SYSTEM_PROMPT = """
        You are a helpful website chatbot who is tasked with answering questions about marketing and attribution.
        You should answer based only on your general knowledge without using any specific document retrieval.
        Keep your answers short and accurate.
        """

        self.DATABASE_SYSTEM_PROMPT = f"""
        You are a data analysis assistant specialized in marketing analytics.
        Your primary responsibility is to help users analyze marketing data, understand metrics,
        and extract insights from the database.

        When a user asks a question about data:
        1. Analyze what metrics or KPIs they're interested in
        2. Use the query_database tool to retrieve relevant data
        3. Explain the results in a clear, concise manner
        4. Provide insights based on the data, focusing on actionable information
        5. Format tables neatly using markdown

        For marketing-specific questions that don't require database access,
        direct the user to ask the question in a way that would make use of the RAG agent,
        which has access to {self.COMPANY_NAME}'s knowledge base.

        Keep your responses focused on the data and insights, avoiding speculation
        beyond what the data shows. When appropriate, suggest further analyses that
        might be valuable.
        """

        self.GOOGLE_SLIDE_IMAGE_TO_TEXT_PROMPT= """
        Please provide a complete and detailed transcription of ALL content in this slide image, formatted as clean markdown:

        1. Use "## Title" for the slide title (exactly as it appears, without mentioning it's the slide title)
        2. Use "### Subtitle" for any subtitles (exactly as they appear)
        3. Format ALL bullet points as proper markdown lists (using - or * for each item) with proper indentation for nested lists
        4. If it's a table: render the ENTIRE table in markdown table format (|---|---|) with ALL rows, columns, and cell contents
        5. If it contains a chart/graph: create a detailed section describing the chart including:
           - Chart type and title (as a heading)
           - All axis labels and ranges
           - Each data series and its values
           - Legend information
           - Key trends or data points
        6. If it contains images: create a section describing each image in detail
        7. Include all footnotes, citations, or small text using appropriate markdown (e.g., > for quotes, *italics* for emphasis)

        Do NOT add any meta-commentary (like "This slide contains") - just transcribe the content directly using proper markdown formatting.
        Do NOT summarize or paraphrase - transcribe EVERYTHING exactly as it appears.
        Format your response as a clean, properly structured markdown document that could be used as-is.
        """

        # Retriever Tool Configuration with expanded parameters
        self.RETRIEVER_TOOL_CONFIG = {
            "name": "search_rag_docs",
            "description": f"""Searches and returns docs, products and blogs from RAG.
            You do not know anything about {self.COMPANY_NAME}, so if you are
            ever asked about {self.COMPANY_NAME}, you should use this tool.
            
            To get more relevant results, you should specify:
            1. query: The main search terms or question
            2. content_type (optional): Specify the type of content to search ("article", "blog", "product", "client_case_study")
            3. client_name (optional): If the query is about a specific client (e.g., "LaserAway"), specify the client name
            4. topic (optional): The specific topic of interest ("attribution", "incrementality", "geo_testing", "mmm", "facebook", "tiktok")
            
            Example: For "What were the results of LaserAway's Facebook geo tests?", you would provide:
            query: "LaserAway Facebook geo tests results"
            client_name: "LaserAway" 
            topic: "geo_testing"
            """
        }

        # Document prompt template
        self.DOCUMENT_PROMPT_TEMPLATE = "<DOC>\n# Title: {title}\n*Source*: ({url})\n## Content: {page_content}\n</DOC>"

    def update_setting(self, setting_name, value):
        """Update a setting value if it exists
               @param setting_name:
               @param value:
               @return:
        """
        if hasattr(self, setting_name):
            setattr(self, setting_name, value)
            return True
        return False

    def get_all_settings(self):
        """Get all settings as a dictionary (excluding private attributes)"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def validate_settings(self):
        """Validate that essential settings are present"""
        missing_settings = []

        # Check which indexer we're using
        if os.environ.get("USE_GOOGLE_DRIVE", "false").lower() == "true":
            # Google Drive validation
            if not getattr(self, "GOOGLE_DRIVE_CREDENTIALS_FILE", None):
                    missing_settings.append("GOOGLE_DRIVE_CREDENTIALS_FILE")
        else:
            # Shopify validation
            if not getattr(self, "SHOPIFY_API_KEY", None):
                missing_settings.append("SHOPIFY_API_KEY")

        # Common validations
        if not getattr(self, "PINECONE_API_KEY", None):
            missing_settings.append("PINECONE_API_KEY")

        if not getattr(self, "OPENAI_API_KEY", None):
            missing_settings.append("OPENAI_API_KEY")

        return missing_settings

    def _get_model_configs(self) -> Dict[str, ChatModelConfig]:
        llm_proxy_config = None
        portkey_api_key = os.getenv("PORTKEY_API_KEY")
        if portkey_api_key is not None:
            llm_proxy_config = LlmProxyConfig(
                proxy_type = LlmProxyType.PORTKEY,
                url = "https://api.portkey.ai/v1/proxy",
                api_key = portkey_api_key,
                cache_ttl = int(os.getenv("PORTKEY_CACHE_TTL", "3600"))
            )

        chat_model_configs: Dict[str, ChatModelConfig] = {} # user can configure more than one models
        if self.OPENAI_EMBEDDING_MODEL is not None:
            if self.PINECONE_INDEX_NAME is None:
                # we do need a vector store; if not found throw error
                raise ValueError(f"Pinecone index name not found for OpenAI model '{self.PINECONE_INDEX_NAME}'")

            # Use a namespace from environment or default based on index
            pinecone_namespace = os.getenv("PINECONE_NAMESPACE", "default")
            
            vector_store_config = PineconeConfig(
                api_key = self.PINECONE_API_KEY,
                index_name = self.PINECONE_INDEX_NAME,
                cloud = self.PINECONE_CLOUD,
                region = self.PINECONE_REGION,
                namespace = pinecone_namespace)

            chat_model_configs[self.OPENAI_EMBEDDING_MODEL] = ChatModelConfig(
                cloud_provider=CloudProvider.OpenAI,
                model = self.OPENAI_EMBEDDING_MODEL,
                vector_store_config = vector_store_config,
                cloud_api_key = self.OPENAI_API_KEY,
                llm_proxy_config = llm_proxy_config,
                embedding_context_length = self.EMBEDDING_CONTEXT_LENGTH)

        if self.OLLAMA_EMBEDDING_MODEL is not None:
            if self.OLLAMA_PINECONE_INDEX_NAME is None:
                # we do need a vector store; if not found throw error
                raise ValueError(f"Pinecone index name not found for Ollama model '{self.PINECONE_INDEX_NAME}'")

            # Use a namespace from environment or default based on index for Ollama
            ollama_namespace = os.getenv("OLLAMA_PINECONE_NAMESPACE", "ollama")
            
            vector_store_config = PineconeConfig(
                api_key = self.PINECONE_API_KEY,
                index_name = self.OLLAMA_PINECONE_INDEX_NAME,
                cloud = self.PINECONE_CLOUD,
                region = self.PINECONE_REGION,
                namespace = ollama_namespace)

            chat_model_configs["ollama_embedding"] = ChatModelConfig(
                cloud_provider=CloudProvider.Local,
                model = self.OLLAMA_EMBEDDING_MODEL,
                vector_store_config = vector_store_config,
                cloud_api_key = None,
                llm_proxy_config = llm_proxy_config,
                embedding_context_length = self.EMBEDDING_CONTEXT_LENGTH
            )
        return chat_model_configs

# Create a singleton instance
chat_config = ChatConfig()
