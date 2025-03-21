"""
Agent configuration and execution logic.
"""
from textwrap import dedent
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor, create_openai_functions_agent
from langchain.schema import SystemMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain.callbacks.base import BaseCallbackHandler
from typing_extensions import Optional

from app.tools.gpt_tools import ToolManager
from app.config.chat_config import ChatConfig
from app.utils.llm_client import LLMClientManager
from app.utils.logging_utils import get_logger
from app.utils.other_utlis import write_data_logfile
import json
import os
from datetime import datetime


class PromptCaptureCallback(BaseCallbackHandler):
    """Callback handler to capture complete prompts and responses with rotating log files."""

    def __init__(self):
        # Get the project root directory for constructing paths
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.log_dir = os.path.join(project_root, "logs")
        
        # Make sure the logs directory exists
        try:
            os.makedirs(self.log_dir, exist_ok=True)
        except Exception as e:
            print(f"Error creating logs directory: {e}")
            # Fallback to a location that should be writable
            self.log_dir = "/tmp/chatbot_logs"
            os.makedirs(self.log_dir, exist_ok=True)
        
        # Define specific log file for prompt captures with .jsonl extension
        self.log_file = os.path.join(self.log_dir, "llm_prompts_responses.jsonl")
        
        # Create a logger specifically for prompt captures with a rotating file handler
        self.logger = get_logger(
            "prompt_capture", 
            "DEBUG",
            use_rotating_file=True,
            log_file=self.log_file
        )
        
        # Log initialization event
        init_data = {
            "event": "init", 
            "timestamp": datetime.now().isoformat(),
            "message": "PromptCaptureCallback initialized with rotating log files"
        }
        
        # Log the initialization data
        self.logger.info(json.dumps(init_data))
        self.logger.debug(f"Successfully initialized prompt capture with rotating logs at {self.log_file}")

    def on_llm_start(self, serialized, prompts, **kwargs):
        """Log the complete prompt when an LLM call starts."""
        try:
            # Create a simpler version of prompts for JSON serialization
            serializable_prompts = []
            for p in prompts:
                try:
                    if hasattr(p, 'to_json'):
                        serializable_prompts.append(p.to_json())
                    elif hasattr(p, 'to_string'):
                        serializable_prompts.append(p.to_string())
                    elif hasattr(p, '__str__'):
                        serializable_prompts.append(str(p))
                    else:
                        serializable_prompts.append(f"Unparseable prompt: {type(p)}")
                except Exception as inner_e:
                    serializable_prompts.append(f"Error serializing prompt: {str(inner_e)}")
            
            # Extract and log the prompt with simplified prompts
            prompt_data = {
                "event": "llm_start",
                "timestamp": datetime.now().isoformat(),
                "model": serialized.get("name", "unknown_model") if serialized else "unknown_model",
                "prompts": serializable_prompts
            }

            # Log the data through the logger which will handle rotation
            try:
                json_data = write_data_logfile("start", prompt_data, self.log_file)

                self.logger.info(json_data)
            except Exception as e:
                error_msg = f"ERROR serializing prompt data: {e}"
                print(error_msg)
                self.logger.error(error_msg)

        except Exception as e:
            error_msg = f"Error in on_llm_start: {e}"
            print(error_msg)
            self.logger.error(error_msg)

    def on_llm_end(self, response, **kwargs):
        """Log the complete response when an LLM call ends."""
        try:
            # Extract and log the response
            response_data = {
                "event": "llm_end",
                "timestamp": datetime.now().isoformat(),
                "response": self._serialize_response(response)
            }

            # Log through the logger which will handle rotation
            try:
                json_data = write_data_logfile("end", response_data, self.log_file)

                # Also log through the logger
                self.logger.info(json_data)
            except Exception as e:
                error_msg = f"ERROR serializing response data: {e}"
                print(error_msg)
                self.logger.error(error_msg)

        except Exception as e:
            error_msg = f"Error in on_llm_end: {e}"
            print(error_msg)
            self.logger.error(error_msg)

    def on_chain_start(self, serialized, inputs, **kwargs):
        """Log when a chain starts."""
        try:
            chain_data = {
                "event": "chain_start",
                "timestamp": datetime.now().isoformat(),
                "chain": serialized.get("name", "unknown_chain") if serialized is not None else "unknown_chain",
                "inputs": self._clean_inputs(inputs)
            }

            # Log through the logger which will handle rotation
            try:
                json_data = write_data_logfile("chain start", chain_data, self.log_file)

                # Also log through the logger
                self.logger.info(json_data)
            except Exception as e:
                error_msg = f"ERROR serializing chain data: {e}"
                print(error_msg)
                self.logger.error(error_msg)

        except Exception as e:
            error_msg = f"Error in on_chain_start: {e}"
            print(error_msg)
            self.logger.error(error_msg)

    def on_tool_start(self, serialized, input_str, **kwargs):
        """Log when a tool is called."""
        try:
            tool_data = {
                "event": "tool_start",
                "timestamp": datetime.now().isoformat(),
                "tool": serialized.get("name", "unknown_tool"),
                "input": input_str
            }

            # Log through the logger which will handle rotation
            try:
                json_data = write_data_logfile("tool start", tool_data, self.log_file)

                # Also log through the logger
                self.logger.info(json_data)
            except Exception as e:
                error_msg = f"ERROR serializing tool start data: {e}"
                print(error_msg)
                self.logger.error(error_msg)

        except Exception as e:
            error_msg = f"Error in on_tool_start: {e}"
            print(error_msg)
            self.logger.error(error_msg)

    def on_tool_end(self, output, **kwargs):
        """Log when a tool returns."""
        try:
            # Make sure output is serializable
            if hasattr(output, 'to_json'):
                serializable_output = output.to_json()
            elif hasattr(output, 'to_dict'):
                serializable_output = output.to_dict()
            elif hasattr(output, '__str__'):
                serializable_output = str(output)
            else:
                serializable_output = f"Unparseable output: {type(output)}"
            
            tool_data = {
                "event": "tool_end",
                "timestamp": datetime.now().isoformat(),
                "output": serializable_output
            }

            # Log through the logger which will handle rotation
            try:
                json_data = write_data_logfile("tool end", tool_data, self.log_file)

                # Also log through the logger
                self.logger.info(json_data)
            except Exception as e:
                error_msg = f"ERROR serializing tool end data: {e}"
                print(error_msg)
                self.logger.error(error_msg)

        except Exception as e:
            error_msg = f"Error in on_tool_end: {e}"
            print(error_msg)
            self.logger.error(error_msg)

    def _serialize_response(self, response):
        """Serialize LLM response to a JSON-friendly format."""
        try:
            # Handle different response structures
            if hasattr(response, "generations"):
                # Extract text from generations
                generations = []
                for gen_list in response.generations:
                    for gen in gen_list:
                        try:
                            gen_dict = {
                                "text": gen.text if hasattr(gen, "text") else str(gen)
                            }
                            # Include function call info if present
                            if hasattr(gen, "message") and hasattr(gen.message, "additional_kwargs"):
                                if "function_call" in gen.message.additional_kwargs:
                                    try:
                                        function_call = gen.message.additional_kwargs["function_call"]
                                        # If it's a complex object, convert to string
                                        if not isinstance(function_call, (dict, str)):
                                            function_call = str(function_call)
                                        gen_dict["function_call"] = function_call
                                    except Exception as fc_err:
                                        gen_dict["function_call_error"] = str(fc_err)
                            generations.append(gen_dict)
                        except Exception as gen_err:
                            generations.append({"error": f"Failed to serialize generation: {str(gen_err)}"})
                return {"generations": generations}
            elif hasattr(response, "to_dict"):
                # Use to_dict method if available
                try:
                    return response.to_dict()
                except Exception as dict_err:
                    return {"error_to_dict": str(dict_err), "raw_response": str(response)}
            elif hasattr(response, "to_json"):
                # Use to_json method if available
                try:
                    return response.to_json()
                except Exception as json_err:
                    return {"error_to_json": str(json_err), "raw_response": str(response)}
            else:
                # Fall back to simple string representation
                return {"raw_response": str(response)}
        except Exception as e:
            error_msg = f"Failed to serialize response: {str(e)}"
            print(f"ERROR: {error_msg}")
            return {"error": error_msg}

    def _clean_inputs(self, inputs):
        """Clean inputs to ensure they're JSON serializable."""
        try:
            # Simple strategy: convert any complex objects to strings
            cleaned = {}
            for key, value in inputs.items():
                try:
                    if isinstance(value, (str, int, float, bool, type(None))):
                        cleaned[key] = value
                    elif isinstance(value, (list, tuple)):
                        # Handle lists of messages
                        cleaned_list = []
                        for i, item in enumerate(value):
                            try:
                                if hasattr(item, 'to_dict'):
                                    # If there's a to_dict method, use it
                                    cleaned_list.append(item.to_dict())
                                elif hasattr(item, 'to_json'):
                                    # If there's a to_json method, use it
                                    cleaned_list.append(item.to_json())
                                elif hasattr(item, 'content') and hasattr(item, 'type'):
                                    # This is likely a LangChain message
                                    cleaned_list.append({
                                        'type': getattr(item, 'type', 'unknown'),
                                        'content': getattr(item, 'content', str(item))
                                    })
                                elif isinstance(item, (dict, list)):
                                    # For nested dictionaries or lists, recursively clean
                                    cleaned_list.append(self._clean_nested_structure(item))
                                else:
                                    # Regular list item
                                    cleaned_list.append(str(item))
                            except Exception as item_ex:
                                cleaned_list.append(f"Error serializing item {i}: {str(item_ex)}")
                        cleaned[key] = cleaned_list
                    elif isinstance(value, dict):
                        # Handle nested dictionaries
                        cleaned[key] = self._clean_nested_structure(value)
                    elif hasattr(value, 'to_dict'):
                        # Use to_dict method if available
                        cleaned[key] = value.to_dict()
                    elif hasattr(value, 'to_json'):
                        # Use to_json method if available
                        cleaned[key] = value.to_json()
                    elif hasattr(value, 'content') and hasattr(value, 'type'):
                        # This is likely a LangChain message
                        cleaned[key] = {
                            'type': getattr(value, 'type', 'unknown'),
                            'content': getattr(value, 'content', str(value))
                        }
                    else:
                        # Fall back to string representation
                        cleaned[key] = str(value)
                except Exception as key_ex:
                    cleaned[key] = f"Error serializing key {key}: {str(key_ex)}"
            return cleaned
        except Exception as e:
            error_msg = f"Error in _clean_inputs: {e}"
            print(f"ERROR: {error_msg}")
            self.logger.error(error_msg)
            return {"error": f"Failed to clean inputs: {str(e)}"}
            
    def _clean_nested_structure(self, structure):
        """Recursively clean a nested structure (dict or list) to ensure JSON serializability."""
        try:
            if isinstance(structure, dict):
                return {k: self._clean_nested_structure(v) for k, v in structure.items()}
            elif isinstance(structure, list):
                return [self._clean_nested_structure(item) for item in structure]
            elif isinstance(structure, (str, int, float, bool, type(None))):
                return structure
            elif hasattr(structure, 'to_dict'):
                return structure.to_dict()
            elif hasattr(structure, 'to_json'):
                return structure.to_json()
            else:
                return str(structure)
        except Exception as e:
            return f"Error in nested structure: {str(e)}"

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

    def get_rag_agent(self, custom_system_prompt=None, expected_answer=None) -> AgentExecutor:
        """
        Get or create a RAG-enabled agent with optional custom system prompt.
        
        Args:
            custom_system_prompt: Optional custom system prompt to override default
            expected_answer: Optional expected answer to include as a hint
            
        Returns:
            Agent executor configured for RAG
        """
        # If we have an expected answer, incorporate it into the system prompt
        if expected_answer:
            self.logger.debug("Adding expected answer to RAG agent system prompt with strong anti-verbatim guidance")

            # Start with either the custom prompt or the default
            base_prompt = custom_system_prompt or self.config.RAG_SYSTEM_PROMPT

            # Extract key concepts from the expected answer without including the full text
            # This helps prevent verbatim copying while still providing guidance

            # Analyze the expected answer to extract key points
            # First, break it into sentences
            sentences = expected_answer.split('.')

            # Filter out empty sentences
            sentences = [s.strip() + '.' for s in sentences if s.strip()]

            # Handle short expected answers differently
            if len(sentences) <= 2:
                # For very short answers, extract key concepts instead of using full sentences
                key_concepts = expected_answer.replace('.', ',').split(',')
                key_concepts = [c.strip() for c in key_concepts if c.strip()]

                # Format as bullet points of key ideas
                key_points = "\n".join([f"- Concept: {concept}" for concept in key_concepts if len(concept) > 5])
            else:
                # For longer answers, use a summary approach
                # Take alternate sentences or first and last, depending on length
                if len(sentences) <= 4:
                    selected_sentences = [sentences[0]] + [sentences[-1]]
                else:
                    # Take first, one from middle, and last sentence
                    selected_sentences = [sentences[0], sentences[len(sentences) // 2], sentences[-1]]

                # Create bullet points with key facts, not complete sentences
                key_points = ""
                for sentence in selected_sentences:
                    # Extract key phrases from the sentence
                    words = sentence.split()
                    if len(words) > 8:
                        # For longer sentences, take phrases rather than whole sentence
                        chunks = [' '.join(words[i:i + 4]) for i in range(0, len(words), 4)]
                        for chunk in chunks:
                            if len(chunk) > 10:  # Only meaningful chunks
                                key_points += f"- Key fact: {chunk}\n"
                    else:
                        # For shorter sentences, use core concept
                        key_points += f"- Main point: {' '.join(words[:min(5, len(words))])}\n"

            # Add the expected answer with instructions to avoid direct copying
            enhanced_prompt = f"""{base_prompt}

IMPORTANT - RESPONSE GUIDANCE:
I'm providing you with key concepts about this topic. Your task is to:

1. STRICTLY AVOID any phrasing that matches the reference material
2. Use only the concepts and facts to inform your response
3. Write a COMPLETELY ORIGINAL answer in your own words and structure
4. Include source links from the retrieved context, not these concepts
5. If these concepts contradict retrieved information, prioritize information from retrieval

KEY CONCEPTS FROM REFERENCE:
{key_points}

CRITICAL INSTRUCTION: Your response must NOT contain any exact phrases from the reference concepts. Rephrase everything completely while preserving the meaning.
"""

            custom_system_prompt = enhanced_prompt
            self.logger.info("Enhanced system prompt with expected answer")

        if custom_system_prompt:
            self.logger.debug("Creating RAG agent with custom system prompt")
            return self._configure_rag_agent(custom_system_prompt)
        
        if self._rag_agent is None:
            self.logger.debug("Initializing RAG agent with default system prompt")
            self._rag_agent = self._configure_rag_agent()
        return self._rag_agent
        
    @property
    def rag_agent(self):
        """Get or lazy-initialize the default RAG-enabled agent."""
        if self._rag_agent is None:
            self.logger.debug("Initializing RAG agent")
            self._rag_agent = self._configure_rag_agent()
        return self._rag_agent

    @property
    def standard_agent(self) -> AgentExecutor:
        """Get or lazy-initialize the standard (non-RAG) agent."""
        if self._standard_agent is None:
            self.logger.debug("Initializing standard agent")
            self._standard_agent = self._configure_standard_agent()
        return self._standard_agent

    @property
    def database_agent(self) -> AgentExecutor:
        """Get or lazy-initialize the database agent."""
        if self._database_agent is None:
            self.logger.debug("Initializing database agent")
            self._database_agent = self._configure_database_agent()
        return self._database_agent

    def _configure_rag_agent(self, custom_system_prompt=None) -> AgentExecutor:
        """
        Configure and return a RAG-enabled agent.
        
        Args:
            custom_system_prompt: Optional custom system prompt to override default
            
        Returns:
            Agent executor configured for RAG
        """
        self.logger.debug("Configuring RAG agent with prompt")
        llm = AgentFactory.create_llm()
        tools = ToolManager.get_rag_tools()
        for tool in tools:
            if not hasattr(tool, "callbacks") or tool.callbacks is None:
                tool.callbacks = []
            if AgentFactory.prompt_capture not in tool.callbacks:
                tool.callbacks.append(AgentFactory.prompt_capture)
                self.logger.debug(f"Added callback to tool: {tool.name}")

        # Use custom system prompt if provided, otherwise use default
        system_prompt = custom_system_prompt or self.config.RAG_SYSTEM_PROMPT
        if custom_system_prompt:
            self.logger.info("Using custom system prompt for RAG agent")
        
        prompt = AgentFactory.create_agent_prompt(system_prompt)

        self.logger.debug(f"Creating RAG agent with {len(tools)} tools")
        agent = create_openai_functions_agent(
            llm=llm,
            tools=tools,
            prompt=prompt
        )

        self.logger.debug("Creating RAG agent executor")
        return AgentFactory.create_agent_executor(agent, tools)

    def _configure_standard_agent(self) -> AgentExecutor:
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

    def _configure_database_agent(self) -> AgentExecutor:
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

    def get_agent(self, agent_type: str) -> Optional[AgentExecutor]:
        agent_choice = {
            "rag": self.get_rag_agent,
            "standard": self.standard_agent,
            "database": self.database_agent
        }

        agent = agent_choice.get(agent_type, None)()
        return agent() if agent else None

# Create a singleton instance
agent_manager = AgentManager()