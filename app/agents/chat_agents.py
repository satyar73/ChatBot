"""
Agent configuration and execution logic.
"""
from textwrap import dedent
from langchain_openai import ChatOpenAI
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor, create_openai_functions_agent
from langchain.schema import SystemMessage
from langchain_core.prompts import MessagesPlaceholder
from app.tools.gpt_tools import ToolManager
from app.config.chat_config import ChatConfig
from app.utils.llm_client import LLMClientManager

class AgentFactory:
    """Factory for creating different types of agents."""
    config = ChatConfig()

    @classmethod
    def create_llm(cls):
        """Create and return the LLM instance."""
        return LLMClientManager.get_chat_llm(
            model=cls.config.LLM_CONFIG_4o["model"],
            temperature=cls.config.LLM_CONFIG_4o["temperature"],
            streaming=cls.config.LLM_CONFIG_4o["streaming"]
        )

    @classmethod
    def create_agent_prompt(cls, system_content):
        """Create agent prompt with system message and history placeholder."""
        system_message = SystemMessage(content=dedent(system_content))
        return OpenAIFunctionsAgent.create_prompt(
            system_message=system_message,
            extra_prompt_messages=[MessagesPlaceholder(variable_name="history")]
        )

    @classmethod
    def create_agent_executor(cls, agent, tools):
        """Create and return an agent executor with the specified agent and tools."""
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            return_intermediate_steps=True,
            handle_parsing_errors=True
        )

class AgentManager:
    """Manager for creating and accessing different types of agents."""

    def __init__(self):
        self._rag_agent = None
        self._standard_agent = None
        self.config = ChatConfig()

    @property
    def rag_agent(self):
        """Get or lazy-initialize the RAG-enabled agent."""
        if self._rag_agent is None:
            self._rag_agent = self._configure_rag_agent()
        return self._rag_agent

    @property
    def standard_agent(self):
        """Get or lazy-initialize the standard (non-RAG) agent."""
        if self._standard_agent is None:
            self._standard_agent = self._configure_standard_agent()
        return self._standard_agent

    def _configure_rag_agent(self):
        """Configure and return a RAG-enabled agent."""
        llm = AgentFactory.create_llm()
        tools = ToolManager.get_rag_tools()
        prompt = AgentFactory.create_agent_prompt(self.config.RAG_SYSTEM_PROMPT)

        agent = create_openai_functions_agent(
            llm=llm,
            tools=tools,
            prompt=prompt
        )

        return AgentFactory.create_agent_executor(agent, tools)

    def _configure_standard_agent(self):
        """Configure and return a standard (non-RAG) agent."""
        llm = AgentFactory.create_llm()
        tools = ToolManager.get_standard_tools()
        prompt = AgentFactory.create_agent_prompt(self.config.NON_RAG_SYSTEM_PROMPT)

        agent = create_openai_functions_agent(
            llm=llm,
            tools=tools,
            prompt=prompt
        )

        return AgentFactory.create_agent_executor(agent, tools)

# Create a singleton instance
agent_manager = AgentManager()
