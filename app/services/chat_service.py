"""
Service layer for handling agent queries and responses.
"""
from typing import Dict, List, Optional, Any
import logging
import os
import sys
from app.agents.chat_agents import agent_manager
from app.models.chat_models import ChatHistory, ResponseContent, ResponseMessage, Source, Message
from app.utils.logging_utils import get_logger, diagnose_logger, ensure_debug_logging

# Determine the environment
environment = os.getenv("ENVIRONMENT", "development").lower()  # Default to "development"
# Set the logging level based on the environment
log_level = logging.INFO if environment == "production" else logging.DEBUG
print(f"Setting log level to {log_level} for {__name__}", file=sys.stderr)

class ChatService:
    """Service for managing chat interactions with agents."""

    def __init__(self):
        # Create a dictionary to store chat histories for different sessions
        self.chat_histories = {}
        self.logger = get_logger(f"{__name__}.ChatService", "DEBUG")
        self.logger.debug("ChatService initialized")
        # Explicit print to check if output is working at all
        print("ChatService initialized", file=sys.stderr)

    async def chat(self, data: Message) -> ResponseMessage:
        """
        Process chat message and return response with both RAG and non-RAG outputs.

        Args:
            data: Message object containing user input and session ID

        Returns:
            ResponseMessage with RAG and non-RAG responses and sources
        """
        user_input = data.message
        session_id = data.session_id

        # Get or create chat history for the session
        if session_id not in self.chat_histories:
            self.chat_histories[session_id] = ChatHistory()
        chat_history = self.chat_histories[session_id]
        chat_history.add_user_message(user_input)

        self.logger.debug(f"chat call with input: {user_input[:50]}...")
        self.logger.debug(f"chat history length: {len(chat_history.get_messages())}")
        # Generate response using agent executor with RAG
        rag_response = await agent_manager.rag_agent.ainvoke(
            {"input": user_input, "history": chat_history.get_messages()},
            include_run_info=True
        )

        self.logger.debug(f"RAG response received, length: {len(str(rag_response))}")
        # Generate response using agent executor without RAG
        no_rag_response = await agent_manager.standard_agent.ainvoke(
            {"input": user_input, "history": chat_history.get_messages()},
            include_run_info=True
        )
        self.logger.debug(f"NO RAG response received, length: {len(str(no_rag_response))}")

        # Extract sources from the RAG response
        sources = self._format_sources(rag_response)

        # Save AI's message to chat history (using the RAG response as primary)
        chat_history.add_ai_message(rag_response['output'])

        # Format message history for response
        formatted_history = self._format_history(chat_history.get_messages())

        # Create the response content
        response_content = ResponseContent(
            input=user_input,
            history=formatted_history,
            output=rag_response['output'],
            no_rag_output=no_rag_response['output'],
            intermediate_steps=rag_response.get('intermediate_steps', [])
        )

        return ResponseMessage(
            response=response_content,
            sources=sources
        )

    def delete_chat(self, session_id: str) -> bool:
        """
        Delete chat history for a session.

        Args:
            session_id: The session ID to delete, or "ALL_CHATS" to clear all sessions

        Returns:
            True if deletion was successful, False otherwise
        """
        self.logger.debug(f"delete_chat called with {session_id}")
        if session_id == "ALL_CHATS":
            self.chat_histories = {}
            return True
        elif session_id in self.chat_histories:
            del self.chat_histories[session_id]
            return True
        return False

    def get_chat(self, session_id: str) -> Dict:
        """
        Get chat history for a session.

        Args:
            session_id: The session ID to retrieve, or "ALL_CHATS" to get all sessions

        Returns:
            Dictionary containing chat history or None if not found
        """
        self.logger.debug(f"get_chat called with {session_id}")
        if session_id == "ALL_CHATS":
            return self.chat_histories
        if session_id in self.chat_histories:
            return {session_id: self.chat_histories[session_id]}
        return None

    def _format_sources(self, rag_response: Dict) -> List[Source]:
        """
        Format sources from the RAG response.

        Args:
            rag_response: Response from the RAG agent

        Returns:
            List of formatted Source objects
        """
        raw_sources = rag_response.get("sources", [])
        formatted_sources = []

        for source in raw_sources:
            if isinstance(source, dict):
                formatted_sources.append(Source(
                    title=source.get("title", ""),
                    url=source.get("url", ""),
                    content=source.get("content", "")
                ))
            else:
                # Handle string or other non-dict sources
                formatted_sources.append(Source(content=str(source)))

        return formatted_sources

    def _format_history(self, messages: List) -> List:
        """
        Format message history for response.

        Args:
            messages: List of message objects

        Returns:
            List of formatted message dictionaries
        """
        return [
            msg.dict() if hasattr(msg, 'dict') else {
                "role": getattr(msg, "type", "unknown"),
                "content": getattr(msg, "content", str(msg))
            }
            for msg in messages
        ]


class AgentService:
    """
    Static service for direct agent interactions without chat history management.
    Use ChatService for most applications that require session management.
    """

    @staticmethod
    async def process_query(
            query: str,
            history: List = None,
            use_rag: bool = True,
            use_dual_response: bool = False
    ) -> ResponseMessage:
        """
        Process a user query through the appropriate agent(s).

        Args:
            query: The user's query text
            history: Optional chat history
            use_rag: Whether to use the RAG-enabled agent
            use_dual_response: Whether to generate both RAG and non-RAG responses

        Returns:
            A ResponseMessage object containing the response(s) and sources
        """
        history = history or []
        sources = []

        # Get RAG response if requested
        rag_output = None
        rag_steps = None
        if use_rag or use_dual_response:
            rag_result = await agent_manager.rag_agent.ainvoke({
                "input": query,
                "history": history
            })
            rag_output = rag_result.get("output", "")
            rag_steps = rag_result.get("intermediate_steps", [])

            # Extract sources from RAG response
            sources = AgentService._extract_sources(rag_steps)

        # Get non-RAG response if requested
        non_rag_output = None
        if not use_rag or use_dual_response:
            non_rag_result = await agent_manager.standard_agent.ainvoke({
                "input": query,
                "history": history
            })
            non_rag_output = non_rag_result.get("output", "")

        # Determine primary output
        primary_output = rag_output if use_rag else non_rag_output
        secondary_output = non_rag_output if use_rag and use_dual_response else None

        # Create response content
        response_content = ResponseContent(
            input=query,
            history=history,
            output=primary_output,
            no_rag_output=secondary_output,
            intermediate_steps=rag_steps if use_rag else []
        )

        return ResponseMessage(
            response=response_content,
            sources=sources
        )

    @staticmethod
    def _extract_sources(steps: List) -> List[Source]:
        """
        Extract source information from agent intermediate steps.

        Args:
            steps: List of intermediate steps from agent execution

        Returns:
            List of Source objects
        """
        sources = []

        for step in steps:
            if (len(step) > 1 and step[0].tool == "search_msquared_docs"
                    and isinstance(step[1], dict)):
                docs = step[1].get("documents", [])
                for doc in docs:
                    metadata = doc.metadata or {}
                    source = Source(
                        title=metadata.get("title", None),
                        url=metadata.get("url", None),
                        content=doc.page_content
                    )
                    sources.append(source)

        return sources