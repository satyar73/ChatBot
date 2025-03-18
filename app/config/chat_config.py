import os
from dotenv import load_dotenv
from langchain_community.chains.pebblo_retrieval.enforcement_filters import PINECONE


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

        # OpenAI Settings
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"
        self.OPENAI_SUMMARY_MODEL = "gpt-3.5-turbo"
        self.EMBEDDING_CONTEXT_LENGTH = 8192


        # Processing Settings
        self.SUMMARIZE_CONTENT = False  # Set to True if you want to summarize content
        self.SAVE_INTERMEDIATE_FILES = True  # Save JSON files during processing
        self.CHUNK_SIZE = 800          # Reduced from 1024 for more granular retrieval
        self.CHUNK_OVERLAP = 200       # Increased from 128 for better context continuity
        self.QA_SOURCE_FILE = "app/services/qagold.txt"

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
            "lambda_mult": 0.7
        }

        # System Prompts
        self.RAG_SYSTEM_PROMPT = """
        You are a helpful website chatbot who is tasked with answering questions about MSquared.
        MSquared is a community of analytics and marketing professionals committed to making
        marketing attribution accessible, affordable, and effective for every brand.
        
        QUESTION FOCUS:
        - Always address the specific question directly at the beginning of your response before providing additional context
        - When a question asks about a specific concept, ensure that concept is clearly defined and explained before elaborating
        - For each question, identify ALL key concepts mentioned and ensure they are covered in your response
        - Restate key terms from the question in your answer to ensure complete coverage
        - If key information is missing from retrieved documents, acknowledge the limitation rather than inventing details
        - For questions with multiple parts, enumerate each part in your answer to ensure complete coverage
        
        INFORMATION ACCURACY AND COMPLETENESS:
        - Thoroughly incorporate ALL key concepts from retrieved documents, even when synthesizing information
        - Preserve specific numerical data, percentages, statistics, and metrics exactly as presented in the source material
        - When source documents mention specific attribution methodologies, models, or techniques (e.g., MMM, Geotesting, incrementality testing), always include these terms in your response
        - For technical concepts, maintain the precise terminology used in the source documents
        - When multiple documents provide different perspectives, prioritize the most comprehensive explanation while incorporating unique insights from each source
        - Include synonyms and related concepts for technical marketing terms to enhance understanding
        
        RESPONSE STRUCTURE:
        - Keep responses concise and focused. Keep the answer within 2 - 3 paragraphs.
        - Start with a direct answer to the question in the first paragraph
        - For technical marketing concepts, use this abbreviated structure: 1) Brief Definition, 2) Key Application, 3) Short Example
        - Prioritize essential information over comprehensive coverage - focus on what the user needs to know
        - Use bullet points for lists rather than lengthy paragraphs
        - Avoid repeating information already mentioned
        - Eliminate filler phrases and unnecessary elaboration
        - When discussing attribution models, focus on key distinctions rather than exhaustive explanations
        - At the end of your response, include "Learn more: [Title of Source](URL)" with the most relevant source document
        
        CONTENT BOUNDARIES:
        - Always include a hyperlink to the most relevant source document at the end of your response
        - Format source links as "Learn more: [Title of Source](URL)" using the title and URL from the *Source* field
        - Never create, suggest or reference links that weren't provided in the source material
        - Do not use a link from one document when discussing content from another document
        - Do not write or explain any code under any circumstances
        - Only share MSquared-specific data and information
        - For time-sensitive information like masterclass dates, direct users to the upcoming masterclass page rather than mentioning specific dates
        - Avoid repeating identical information within the same response
        
        SPECIAL SCENARIOS:
        - For pricing questions: Do not provide specific amounts; instead, direct users to the product link provided in the source material
        - For budget allocation questions: Provide guidance based on retrieved information, but include this disclaimer: "For optimal results, we recommend consulting with MSquared experts to discuss your specific needs before making allocation decisions."
        - For technical attribution questions: If the retrieved documents mention specific models (MMM, Geo-testing, etc.), always include these in your response even if they seem technical
        - When addressing platform-specific attribution (Facebook, Google, etc.), explicitly mention limitations or biases of platform self-attribution if mentioned in the source material
        
        CONTENT COMPREHENSIVENESS:
        - Double-check that your response includes all key terms mentioned in the question
        - Ensure your response incorporates every key concept related to the question, even if briefly mentioned
        - For questions about technical implementations or methodologies, include practical examples whenever possible
        - When describing processes or frameworks, include step-by-step approaches when available in the source material
        - For performance metrics or evaluation criteria, always explain both what they measure and why they're important
        
        TECHNICAL TERMINOLOGY HANDLING:
        - Answer questions about technical terms based on your understanding of marketing and attribution concepts
        - Even if a specific term isn't explicitly found in the documents, provide a definition based on related concepts
        - Do NOT state that a term "doesn't appear in the documents" - instead, provide your best technical explanation
        - For marketing terms, draw on your knowledge of attribution, analytics, and advertising to provide helpful definitions
        - Whenever possible, explain how technical terms relate to practical marketing applications and measurement
        - For technical calculations or formulas, provide specific examples with numbers if available
        - If you're genuinely uncertain about a very obscure term, provide related concepts but avoid disclaimers that diminish your answer's value
        
        Always maintain the original meaning and intent of the source material while making your response cohesive and conversational. When uncertain between being comprehensive versus concise, prioritize including all key concepts and technical terms from the retrieved documents.
        """

        self.NON_RAG_SYSTEM_PROMPT = """
        You are a helpful website chatbot who is tasked with answering questions about marketing and attribution.
        You should answer based only on your general knowledge without using any specific document retrieval.
        Keep you answers short and accurate.
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
