import os
from dotenv import load_dotenv

class ChatConfig:
    """Class to manage all configuration settings for the application"""

    def __init__(self):
        # Load environment variables
        load_dotenv()

        self.OUTPUT_DIR = "data"

        # Google Drive API Settings
        self.GOOGLE_DRIVE_CREDENTIALS_FILE = os.getenv("GOOGLE_DRIVE_CREDENTIALS_FILE")
        self.GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID", None)  # Optional, can be None to use root
        self.GOOGLE_DRIVE_RECURSIVE = True  # Process subfolders recursively

        # Google Drive specific file paths
        self.DRIVE_DOCUMENTS_FILE = os.path.join(self.OUTPUT_DIR, "drive_documents.json")
        self.DRIVE_PROCESSED_FILE = os.path.join(self.OUTPUT_DIR, "drive_processed.json")

        # Shopify API Settings
        self.SHOPIFY_API_KEY = os.getenv("SHOPIFY_API_KEY")
        self.SHOPIFY_STORE = "919904"  # MSquare shopify store id
        self.SHOPIFY_SHOP_DOMAIN = "919904.myshopify.com"  # MSquare shopify store id
        self.SHOPIFY_API_VERSION = "2024-04"
        self.SHOPIFY_SITE_BASE_URL = "https://msquared.club"

        # API Limits
        self.BLOG_FETCH_LIMIT = 250
        self.ARTICLE_FETCH_LIMIT = 250
        self.PRODUCT_FETCH_LIMIT = 250

        # Pinecone Settings
        self.PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        self.PINECONE_INDEX_NAME = "attributiongpt-23082024"
        self.PINECONE_DIMENSION = 512
        self.PINECONE_CLOUD = "aws"
        self.PINECONE_REGION = "us-west-2"

        # OpenAI Settings
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
        self.OPENAI_SUMMARY_MODEL = "gpt-3.5-turbo"

        # Processing Settings
        self.SUMMARIZE_CONTENT = False  # Set to True if you want to summarize content
        self.SAVE_INTERMEDIATE_FILES = True  # Save JSON files during processing
        self.CHUNK_SIZE = 1024
        self.CHUNK_OVERLAP = 128

        # File paths

        self.BLOGS_FILE = os.path.join(self.OUTPUT_DIR, "blogs.json")
        self.BLOGS_PROCESSED_FILE = os.path.join(self.OUTPUT_DIR, "blogs_processed.json")
        self.PRODUCTS_FILE = os.path.join(self.OUTPUT_DIR, "products.json")
        self.PRODUCTS_PROCESSED_FILE = os.path.join(self.OUTPUT_DIR, "products_processed.json")
        self.COMBINED_FILE = os.path.join(self.OUTPUT_DIR, "msquare_combined.json")

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

        # Vector Store Configuration
        self.VECTOR_STORE_CONFIG = {
            "index_name": "attributiongpt-23082024",
            "embedding_model": "text-embedding-3-small",
            "dimensions": 1536
        }

        # Retriever Configuration
        self.RETRIEVER_CONFIG = {
            "search_type": "mmr",
            "k": 5,
            "fetch_k": 20,
            "lambda_mult": 0.5
        }

        # System Prompts
        self.RAG_SYSTEM_PROMPT = """
        You are a helpful website chatbot who is tasked with answering questions about MSquared.
        MSquared is community of analytics and marketing professionals committed to making
        marketing attribution accessible, affordable, and effective for every brand.
        Unless otherwise explicitly stated, it is probably fair to assume that questions are about MSquared and marketing.
        If there is any ambiguity, you probably assume they are about MSquared. Keep you answers short and accurate.
        The key is to give the users a brief to get them interested to explore the blogs and products but dont be too pushy.
        If the user asked about the price of product, please don't answer the amount instead provide product link.
        Only provide links of products/blogs if they were mentioned as *Source*, else it won't work. If the link is not given 
        in source, then do not provide any made up links. Also avoid using the link from one doc when mentioning content 
        from another doc. Each searched doc is a combination of Title, Source and Content. Do not write or explain any code. 
        Make sure that you are only sharing the MSquared data. If user asked about the date of the masterclass then direct 
        them to upcoming masterclass page else and avoid mentioning specific dates. Do not repeat yourself in each message.
        If user asks budget allocation type questions, provide an answer if you can but give them a disclaimer that they 
        should consult with MSquared experts to discuss before taking any decision.
        """

        self.NON_RAG_SYSTEM_PROMPT = """
        You are a helpful website chatbot who is tasked with answering questions about MSquared.
        MSquared is community of analytics and marketing professionals committed to making
        marketing attribution accessible, affordable, and effective for every brand.

        This is the non-RAG version of the response, so you should answer based only on your general knowledge 
        without using any specific document retrieval. Keep you answers short and accurate.

        Make sure to clarify that your response is not based on specific MSquared documentation and may not 
        include the most up-to-date information about their products and services.
        """
        
        self.DATABASE_SYSTEM_PROMPT = """
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
        which has access to MSquared's knowledge base.
        
        Keep your responses focused on the data and insights, avoiding speculation 
        beyond what the data shows. When appropriate, suggest further analyses that 
        might be valuable.
        """

        # Retriever Tool Configuration
        self.RETRIEVER_TOOL_CONFIG = {
            "name": "search_msquared_docs",
            "description": "Searches and returns docs, products and blogs from MSquared. You do not know anything about MSquared, so if you are ever asked about MSquared, you should use this tool."
        }

        # Document prompt template
        self.DOCUMENT_PROMPT_TEMPLATE = "<DOC>\n# Title: {title}\n*Source*: ({url})\n## Content: {page_content}\n</DOC>"

    def update_setting(self, setting_name, value):
        """Update a setting value if it exists"""
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
