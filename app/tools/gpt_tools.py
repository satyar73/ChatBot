"""
Tools for the agent system.
"""
from datetime import datetime
from langchain_core.tools import tool
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain_core.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from app.config.chat_config import ChatConfig

class ToolManager:
    """Manager for all tools used by the agent."""
    config = ChatConfig()

    @staticmethod
    @tool
    def get_current_time() -> str:
        """Get the current time of the user"""
        now = datetime.utcnow()
        current_time = f'the current time is {now.strftime("%d-%B-%Y")}'
        return current_time

    @classmethod
    def configure_retriever(cls):
        """Configure and return a vector store retriever."""
        embeddings = OpenAIEmbeddings(
            model=cls.config.VECTOR_STORE_CONFIG["embedding_model"],
            dimensions=cls.config.VECTOR_STORE_CONFIG["dimensions"]
        )

        vectorstore = PineconeVectorStore(
            index_name=cls.config.VECTOR_STORE_CONFIG["index_name"],
            embedding=embeddings
        )

        retriever = vectorstore.as_retriever(
            search_type=cls.config.RETRIEVER_CONFIG["search_type"],
            search_kwargs={
                "k": cls.config.RETRIEVER_CONFIG["k"],
                "fetch_k": cls.config.RETRIEVER_CONFIG["fetch_k"],
                "lambda_mult": cls.config.RETRIEVER_CONFIG["lambda_mult"]
            }
        )
        return retriever

    @classmethod
    def get_retriever_tool(cls):
        """Create and return a retriever tool."""
        retriever = cls.configure_retriever()

        return create_retriever_tool(
            retriever,
            cls.config.RETRIEVER_TOOL_CONFIG["name"],
            cls.config.RETRIEVER_TOOL_CONFIG["description"],
            document_prompt=PromptTemplate.from_template(cls.config.DOCUMENT_PROMPT_TEMPLATE)
        )

    @classmethod
    def get_rag_tools(cls):
        """Get tools for RAG-enabled agent."""
        return [cls.get_retriever_tool(), cls.get_current_time]

    @classmethod
    def get_standard_tools(cls):
        """Get tools for standard (non-RAG) agent."""
        return [cls.get_current_time]
