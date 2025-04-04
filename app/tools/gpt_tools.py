"""
Tools for the agent system.
"""
import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from langchain_core.tools import tool
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain_core.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

from app.config.chat_config import chat_config
from app.config.chat_model_config import ChatModelConfig
from app.config.llm_proxy_config import LlmProxyType
from app.services.enhancement_service import enhancement_service

class ToolManager:
    """Manager for all tools used by the agent."""
    config = chat_config

    @staticmethod
    @tool
    def get_current_time() -> str:
        """Get the current time of the user"""
        now = datetime.utcnow()
        current_time = f'the current time is {now.strftime("%d-%B-%Y")}'
        return current_time
        
    @staticmethod
    @tool
    def query_database(query: str) -> str:
        """Execute a database query and return the results.
        The query should be specific about what data is needed.
        Example queries:
        - Show me top products by revenue
        - Get customer conversion rates by month
        - Calculate average order value by product category
        """
        # This is a simplified implementation - in a real system, you would:
        # 1. Parse the natural language query
        # 2. Translate to SQL or other database query language
        # 3. Execute against your actual database
        # 4. Format and return the results
        
        # For demo purposes, return sample data based on keywords in query
        if "products" in query.lower() or "revenue" in query.lower():
            return """
            | Product          | Revenue   | Units Sold |
            |------------------|-----------|------------|
            | Attribution Pro  | $125,000  | 250        |
            | Marketing Suite  | $87,500   | 350        |
            | Analytics Basic  | $45,000   | 900        |
            | Data Connector   | $32,500   | 650        |
            """
        elif "customer" in query.lower() or "conversion" in query.lower():
            return """
            | Month     | Visitors | Conversions | Rate  |
            |-----------|----------|-------------|-------|
            | January   | 25,400   | 762         | 3.0%  |
            | February  | 28,500   | 913         | 3.2%  |
            | March     | 32,100   | 1,091       | 3.4%  |
            | April     | 30,800   | 956         | 3.1%  |
            """
        elif "average" in query.lower() or "order" in query.lower():
            return """
            | Category        | Avg Order Value |
            |-----------------|-----------------|
            | Enterprise      | $3,250          |
            | Mid-market      | $1,125          |
            | Small Business  | $485            |
            """
        else:
            return "I don't have specific data for that query. Please try a more specific question about products, revenue, customers, or order values."

    @classmethod
    def configure_retriever(cls, 
                            chat_model_config: ChatModelConfig, 
                            query: Optional[str] = None,
                            content_type: Optional[str] = None,
                            client_name: Optional[str] = None, 
                            topic: Optional[str] = None):
        """
        Configure and return a vector store retriever with enhanced filtering and query enrichment.
        
        Args:
            chat_model_config: Configuration for the chat model and vector store
            query: The main search query string
            content_type: Optional content type to filter by ("article", "blog", "product", etc.)
            client_name: Optional client name to filter by (e.g., "LaserAway")
            topic: Optional topic to filter by ("attribution", "geo_testing", etc.)
        """
        # Log all parameters for debugging
        logger = logging.getLogger(__name__)
        logger.debug(f"Configuring retriever with parameters:")
        logger.debug(f"  query: {query}")
        logger.debug(f"  content_type: {content_type}")
        logger.debug(f"  client_name: {client_name}")
        logger.debug(f"  topic: {topic}")
        
        if (chat_model_config.llm_proxy_config is None 
            or chat_model_config.llm_proxy_config.proxy_type == LlmProxyType.PORTKEY):
            # should be using OpenAI
            vector_store_config = chat_model_config.vector_store_config

            embeddings = OpenAIEmbeddings(
                model=chat_model_config.model,
                dimensions=vector_store_config.get_embedding_dimensions(model_name=chat_model_config.model)
            )

            # Initialize the vectorstore with the appropriate namespace
            # If client_name is provided, use it as the namespace
            if client_name:
                namespace = client_name
                logger.debug(f"NAMESPACE TRACKING (QUERY): Using client name '{client_name}' as namespace")
            else:
                # Otherwise use the default namespace from config
                namespace = vector_store_config.namespace
                logger.debug(f"NAMESPACE TRACKING (QUERY): Using default namespace from config: '{namespace}'")
            
            # Log environment variables that might affect namespaces
            pinecone_namespace_env = os.getenv("PINECONE_NAMESPACE", "Not set")
            logger.debug(f"NAMESPACE TRACKING (ENV): PINECONE_NAMESPACE environment variable is '{pinecone_namespace_env}'")
            
            logger.debug(f"NAMESPACE TRACKING (STRATEGY): Using explicit namespace parameter with fresh index")
            
            logger.info(f"Using namespace '{namespace}' for retrieval")
            
            vectorstore = PineconeVectorStore(
                index_name=vector_store_config.index_name,
                embedding=embeddings,
                namespace=namespace  # Explicitly use client name as namespace if provided
            )

            # Set up base search parameters
            search_kwargs: Dict[str, Any] = {
                "k": cls.config.RETRIEVER_CONFIG["k"],
                "fetch_k": cls.config.RETRIEVER_CONFIG["fetch_k"],
                "lambda_mult": cls.config.RETRIEVER_CONFIG["lambda_mult"]
            }
            
            # Build a more specific filter based on provided parameters
            filter_dict = {}
            
            # Apply client filter if provided
            if client_name:
                # Use the client name for filtering
                filter_dict["client"] = client_name
                # Also set type to client for client-specific content
                filter_dict["type"] = "client"
                
            # Apply content type filter if provided
            if content_type:
                # Map content_type to source field in metadata
                if content_type in ["article", "blog", "product", "client_case_study"]:
                    filter_dict["source"] = content_type
                    
            # Apply topic filter if provided
            if topic:
                # Add topic to keywords field for filtering
                filter_dict["keywords"] = topic
                
            # Set default filter if no specific filters were applied
            if not filter_dict:
                filter_dict["type"] = "Domain Knowledge"
                
            # Apply the constructed filter
            search_kwargs["filter"] = filter_dict
            
            # Enhance the query using the enhancement service for better semantic matching
            if query:
                # Create metadata for the query based on provided parameters
                query_metadata = {}
                if client_name:
                    query_metadata["client"] = client_name
                if content_type:
                    query_metadata["type"] = content_type
                    query_metadata["source"] = content_type
                if topic:
                    query_metadata["keywords"] = [topic]
                
                # First use the enhance_query method to generate a semantically richer query
                enhanced_query_result = enhancement_service.enhance_query(query)
                enhanced_query = enhanced_query_result["enhanced_query"]
                
                logger.info(f"Enhanced query for retriever: {enhanced_query}")
                
                # Then create a comprehensive embedding prompt with rich context
                embedding_prompt = enhancement_service.create_embedding_prompt(
                    text=enhanced_query,
                    metadata=query_metadata
                )
                
                logger.debug(f"Using embedding prompt for retrieval: {embedding_prompt[:100]}...")
                
                # Store both the original query and the enhanced embedding prompt
                # This maintains backward compatibility with similarity_search_with_score
                # while also supporting the more advanced embedding approach
                search_kwargs["query_texts"] = [embedding_prompt]
                
                # Log complete search parameters for debugging
                logger.debug("Complete search parameters:")
                logger.debug(f"  filter: {search_kwargs.get('filter', {})}")
                logger.debug(f"  k: {search_kwargs.get('k')}")
                logger.debug(f"  fetch_k: {search_kwargs.get('fetch_k')}")
                logger.debug(f"  lambda_mult: {search_kwargs.get('lambda_mult')}")
                logger.debug(f"  search_type: {cls.config.RETRIEVER_CONFIG['search_type']}")
                logger.debug(f"  namespace: {vector_store_config.namespace}")
                
                # Log the full embedding prompt (truncated for readability)
                max_log_len = 200
                if len(embedding_prompt) > max_log_len:
                    logger.debug(f"Full query text (truncated): {embedding_prompt[:max_log_len]}...")
                else:
                    logger.debug(f"Full query text: {embedding_prompt}")
            
            retriever = vectorstore.as_retriever(
                search_type=cls.config.RETRIEVER_CONFIG["search_type"],
                search_kwargs=search_kwargs
            )
            return retriever
        else:
            #should be using Ollama
            return None #TODO for now return None

    @classmethod
    def get_retriever_tool(cls, chat_model_config: ChatModelConfig, query=None, content_type=None, client_name=None, topic=None):
        """
        Create and return a retriever tool with enhanced filtering capabilities and semantic query processing.
        
        Args:
            chat_model_config: Configuration for the chat model and vector store
            query: Main search query string
            content_type: Optional content type to filter by ("article", "blog", "product", etc.)
            client_name: Optional client name to filter by (e.g., "LaserAway") - this is also used as namespace
            topic: Optional topic to filter by ("attribution", "geo_testing", etc.)
        """
        # Log the client_name parameter for debugging
        logger = logging.getLogger(__name__)
        logger.info(f"Creating retriever tool with client_name='{client_name}'")
        
        # Configure retriever with all available parameters
        retriever = cls.configure_retriever(
            chat_model_config=chat_model_config, 
            query=query,
            content_type=content_type,
            client_name=client_name,
            topic=topic
        )

        # If retriever is None (e.g., when using Ollama), return None
        if retriever is None:
            return None

        return create_retriever_tool(
            retriever,
            cls.config.RETRIEVER_TOOL_CONFIG["name"],
            cls.config.RETRIEVER_TOOL_CONFIG["description"],
            document_prompt=PromptTemplate.from_template(cls.config.DOCUMENT_PROMPT_TEMPLATE)
        )

    @classmethod
    def get_rag_tools(cls, chat_model_config: ChatModelConfig):
        """Get tools for RAG-enabled agent."""
        tools = [cls.get_current_time]
        retriever_tool = cls.get_retriever_tool(chat_model_config)
        
        if retriever_tool is not None:
            tools.insert(0, retriever_tool)
            
        return tools

    @classmethod
    def get_standard_tools(cls):
        """Get tools for standard (non-RAG) agent."""
        return [cls.get_current_time]
        
    @classmethod
    def get_database_tools(cls):
        """Get tools for database operations."""
        return [cls.query_database, cls.get_current_time]
