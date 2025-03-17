"""
Agent configuration and execution logic.
"""
from textwrap import dedent
from langchain_openai import ChatOpenAI
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor, create_openai_functions_agent
from langchain.schema import SystemMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain.callbacks.base import BaseCallbackHandler
from app.tools.gpt_tools import ToolManager
from app.config.chat_config import ChatConfig
from app.utils.llm_client import LLMClientManager
from app.utils.logging_utils import get_logger
import json
import logging
import os
from datetime import datetime


class PromptCaptureCallback(BaseCallbackHandler):
    """Callback handler to capture complete prompts and responses."""

    def __init__(self):
        # Create a logger specifically for prompt captures
        self.logger = get_logger("prompt_capture", "DEBUG")

        # Create a directory for prompt logs with absolute path
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.log_dir = os.path.join(base_dir, "prompt_logs")
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            self.logger.debug(f"Created prompt logs directory at: {self.log_dir}")
        except Exception as e:
            self.logger.debug(f"Error creating prompt logs directory: {e}")
            # Fallback to a location that should be writable
            self.log_dir = "/tmp/prompt_logs"
            os.makedirs(self.log_dir, exist_ok=True)

        # Create a timestamped log file name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"llm_prompts_responses_{timestamp}.jsonl")
        self.logger.debug(f"Will log prompts to: {self.log_file}")

        # Add file handler with more detailed error handling
        try:
            file_handler = logging.FileHandler(self.log_file)
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

            # Test write to the log file
            with open(self.log_file, "a") as f:
                f.write(json.dumps({"event": "init", "timestamp": datetime.now().isoformat()}) + "\n")

            self.logger.debug(f"Successfully initialized prompt capture logging to {self.log_file}")
        except Exception as e:
            self.logger.debug(f"ERROR setting up log file: {e}")

    def on_llm_start(self, serialized, prompts, **kwargs):
        """Log the complete prompt when an LLM call starts."""
        self.logger.info(f"LLM start event triggered with {len(prompts)} prompts")
        try:
            # Extract and log the prompt
            prompt_data = {
                "event": "llm_start",
                "timestamp": datetime.now().isoformat(),
                "model": serialized.get("name", "unknown_model") if serialized else "unknown_model",
                "prompts": prompts
            }

            self.logger.info(f"Processing LLM start with prompts of total length: {sum(len(p) for p in prompts)}")

            # Write to the log file directly with explicit error reporting
            try:
                with open(self.log_file, "a") as f:
                    json_data = json.dumps(prompt_data)
                    f.write(json_data + "\n")
                    print(f"Successfully wrote {len(json_data)} bytes to log file")
            except Exception as e:
                self.logger.error(f"ERROR writing to log file: {e}")

        except Exception as e:
            print(f"ERROR in on_llm_start: {e}")
            self.logger.error(f"Error logging prompt: {e}")

    def on_llm_end(self, response, **kwargs):
        """Log the complete response when an LLM call ends."""
        self.logger.info(f"LLM END event triggered")
        try:
            # Extract and log the response
            response_data = {
                "event": "llm_end",
                "timestamp": datetime.now().isoformat(),
                "response": self._serialize_response(response)
            }

            self.logger.debug(f"RESPONSE: {json.dumps(response_data)}")

            # Also write to the JSONL file directly
            with open(self.log_file, "a") as f:
                f.write(json.dumps(response_data) + "\n")

        except Exception as e:
            self.logger.error(f"Error logging response: {e}")

    def on_chain_start(self, serialized, inputs, **kwargs):
        """Log when a chain starts."""

        try:
            self.logger.info(f"ON CHAIN START event triggered with inputs: {inputs} and serialized: {serialized}")
            chain_data = {
                "event": "chain_start",
                "timestamp": datetime.now().isoformat(),
                "chain": serialized.get("name", "unknown_chain") if serialized is not None else "unknown_chain",
                "inputs": self._clean_inputs(inputs)
            }

            with open(self.log_file, "a") as f:
                f.write(json.dumps(chain_data) + "\n")

        except Exception as e:
            self.logger.error(f"Error logging chain start: {e}")

    def on_tool_start(self, serialized, input_str, **kwargs):
        """Log when a tool is called."""
        try:
            self.logger.info(f"ON TOOL START event triggered with inputs: {input_str} and serialized: {serialized}")
            tool_data = {
                "event": "tool_start",
                "timestamp": datetime.now().isoformat(),
                "tool": serialized.get("name", "unknown_tool"),
                "input": input_str
            }

            with open(self.log_file, "a") as f:
                f.write(json.dumps(tool_data) + "\n")

        except Exception as e:
            self.logger.error(f"Error logging tool start: {e}")

    def on_tool_end(self, output, **kwargs):
        """Log when a tool returns."""
        try:
            self.logger.info(f"ON TOOL END event triggered with inputs: {output}")
            tool_data = {
                "event": "tool_end",
                "timestamp": datetime.now().isoformat(),
                "output": output
            }

            with open(self.log_file, "a") as f:
                f.write(json.dumps(tool_data) + "\n")

        except Exception as e:
            self.logger.error(f"Error logging tool end: {e}")

    def _serialize_response(self, response):
        """Serialize LLM response to a JSON-friendly format."""
        try:
            # Handle different response structures
            if hasattr(response, "generations"):
                # Extract text from generations
                generations = []
                for gen_list in response.generations:
                    for gen in gen_list:
                        gen_dict = {
                            "text": gen.text
                        }
                        # Include function call info if present
                        if hasattr(gen, "message") and hasattr(gen.message, "additional_kwargs"):
                            if "function_call" in gen.message.additional_kwargs:
                                gen_dict["function_call"] = gen.message.additional_kwargs["function_call"]
                        generations.append(gen_dict)
                return {"generations": generations}
            else:
                # Fall back to simple string representation
                return {"raw_response": str(response)}
        except Exception as e:
            return {"error": f"Failed to serialize response: {str(e)}"}

    def _clean_inputs(self, inputs):
        """Clean inputs to ensure they're JSON serializable."""
        try:
            # Simple strategy: convert any complex objects to strings
            cleaned = {}
            for key, value in inputs.items():
                if isinstance(value, (str, int, float, bool, type(None))):
                    cleaned[key] = value
                elif isinstance(value, (list, tuple)):
                    # Handle lists of messages
                    cleaned_list = []
                    for item in value:
                        if hasattr(item, 'content') and hasattr(item, 'type'):
                            # This is likely a LangChain message
                            cleaned_list.append({
                                'type': getattr(item, 'type', 'unknown'),
                                'content': getattr(item, 'content', str(item))
                            })
                        else:
                            # Regular list item
                            cleaned_list.append(str(item))
                    cleaned[key] = cleaned_list
                elif isinstance(value, dict):
                    # Handle nested dictionaries
                    cleaned[key] = {k: str(v) for k, v in value.items()}
                elif hasattr(value, 'content') and hasattr(value, 'type'):
                    # This is likely a LangChain message
                    cleaned[key] = {
                        'type': getattr(value, 'type', 'unknown'),
                        'content': getattr(value, 'content', str(value))
                    }
                else:
                    # Fall back to string representation
                    cleaned[key] = str(value)
            return cleaned
        except Exception as e:
            self.logger.error(f"Error in _clean_inputs: {e}")
            return {"error": f"Failed to clean inputs: {str(e)}"}

class AgentFactory:
    """Factory for creating different types of agents."""
    config = ChatConfig()

    # Initialize the prompt capture callback
    prompt_capture = PromptCaptureCallback()

    @classmethod
    def create_llm(cls):
        """Create and return the LLM instance."""
        llm = LLMClientManager.get_chat_llm(
            model=cls.config.LLM_CONFIG_4o["model"],
            temperature=cls.config.LLM_CONFIG_4o["temperature"],
            streaming=cls.config.LLM_CONFIG_4o["streaming"]
        )

        # Add the callback to the LLM
        if not hasattr(llm, "callbacks") or llm.callbacks is None:
            llm.callbacks = []
        llm.callbacks.append(cls.prompt_capture)

        return llm

    @classmethod
    def create_agent_prompt(cls, system_content):
        """Create agent prompt with system message and history placeholder."""
        system_message = SystemMessage(content=dedent(system_content))
        prompt = OpenAIFunctionsAgent.create_prompt(
            system_message=system_message,
            extra_prompt_messages=[MessagesPlaceholder(variable_name="history")]
        )

        # Log the created prompt template
        cls.prompt_capture.logger.debug(f"Created prompt template with system content: {system_content[:100]}...")

        return prompt

    @classmethod
    def create_agent_executor(cls, agent, tools):
        """Create and return an agent executor with the specified agent and tools."""
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            return_intermediate_steps=True,
            handle_parsing_errors=True,
            callbacks=[cls.prompt_capture]  # Add the callback here
        )


class AgentManager:
    """Manager for creating and accessing different types of agents."""

    def __init__(self):
        self._rag_agent = None
        self._standard_agent = None
        self._database_agent = None
        self.config = ChatConfig()
        self.logger = get_logger(f"{__name__}.AgentManager", "DEBUG")
        self.logger.info("AgentManager initialized")

    @property
    def rag_agent(self):
        """Get or lazy-initialize the RAG-enabled agent."""
        if self._rag_agent is None:
            self.logger.debug("Initializing RAG agent")
            self._rag_agent = self._configure_rag_agent()
        return self._rag_agent

    @property
    def standard_agent(self):
        """Get or lazy-initialize the standard (non-RAG) agent."""
        if self._standard_agent is None:
            self.logger.debug("Initializing standard agent")
            self._standard_agent = self._configure_standard_agent()
        return self._standard_agent

    @property
    def database_agent(self):
        """Get or lazy-initialize the database agent."""
        if self._database_agent is None:
            self.logger.debug("Initializing database agent")
            self._database_agent = self._configure_database_agent()
        return self._database_agent

    def _configure_rag_agent(self):
        """Configure and return a RAG-enabled agent."""
        self.logger.debug("Configuring RAG agent with prompt")
        llm = AgentFactory.create_llm()
        tools = ToolManager.get_rag_tools()
        for tool in tools:
            if not hasattr(tool, "callbacks") or tool.callbacks is None:
                tool.callbacks = []
            if AgentFactory.prompt_capture not in tool.callbacks:
                tool.callbacks.append(AgentFactory.prompt_capture)
                self.logger.debug(f"Added callback to tool: {tool.name}")

        prompt = AgentFactory.create_agent_prompt(self.config.RAG_SYSTEM_PROMPT)

        self.logger.debug(f"Creating RAG agent with {len(tools)} tools")
        agent = create_openai_functions_agent(
            llm=llm,
            tools=tools,
            prompt=prompt
        )

        self.logger.debug("Creating RAG agent executor")
        return AgentFactory.create_agent_executor(agent, tools)

    def _configure_standard_agent(self):
        """Configure and return a standard (non-RAG) agent."""
        self.logger.debug("Configuring standard agent with prompt")
        llm = AgentFactory.create_llm()
        tools = ToolManager.get_standard_tools()
        prompt = AgentFactory.create_agent_prompt(self.config.NON_RAG_SYSTEM_PROMPT)

        self.logger.debug(f"Creating standard agent with {len(tools)} tools")
        agent = create_openai_functions_agent(
            llm=llm,
            tools=tools,
            prompt=prompt
        )

        self.logger.debug("Creating standard agent executor")
        return AgentFactory.create_agent_executor(agent, tools)

    def _configure_database_agent(self):
        """Configure and return a database-enabled agent."""
        self.logger.debug("Configuring database agent with prompt")
        llm = AgentFactory.create_llm()
        tools = ToolManager.get_database_tools()
        prompt = AgentFactory.create_agent_prompt(self.config.DATABASE_SYSTEM_PROMPT)

        self.logger.debug(f"Creating database agent with {len(tools)} tools")
        agent = create_openai_functions_agent(
            llm=llm,
            tools=tools,
            prompt=prompt
        )

        self.logger.debug("Creating database agent executor")
        return AgentFactory.create_agent_executor(agent, tools)


# Create a singleton instance
agent_manager = AgentManager()