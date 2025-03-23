"""
Client for communicating with the ChatBot API.
"""
import aiohttp
from typing import Tuple

class ChatBotClient:
    """Client for interacting with the ChatBot API."""

    def __init__(self, api_url: str = "http://localhost:8005"):
        """
        Initialize the ChatBot client.

        Args:
            api_url: URL of the ChatBot API
        """
        self.api_url = api_url
        self.session = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an HTTP session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def get_response(self, prompt: str, session_id: str = "test") -> Tuple[str, str]:
        """
        Get both RAG and non-RAG responses from the ChatBot API.

        Args:
            prompt: User prompt
            session_id: Session ID for the chat

        Returns:
            Tuple of (rag_response, no_rag_response)
        """
        session = await self._get_session()

        # Create the request payload
        payload = {
            "message": prompt,
            "session_id": session_id,
            "mode" : "compare"
        }

        try:
            async with session.post(f"{self.api_url}/chat/", json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API request failed with status {response.status}: {error_text}")

                data = await response.json()

                # Extract responses from the response data
                rag_response = data.get("response", {}).get("output", "No RAG response received")
                no_rag_response = data.get("response", {}).get("no_rag_output", "No non-RAG response received")

                return rag_response, no_rag_response
        except aiohttp.ClientError as e:
            raise Exception(f"Connection error: {str(e)}")
        except Exception as e:
            raise Exception(f"Error getting response: {str(e)}")

    async def delete_chat(self, session_id: str) -> bool:
        """
        Delete a chat session.

        Args:
            session_id: Session ID to delete

        Returns:
            True if successful, False otherwise
        """
        session = await self._get_session()

        try:
            async with session.delete(f"{self.api_url}/chat/{session_id}") as response:
                return response.status == 204
        except Exception as e:
            print(f"Error deleting chat: {str(e)}")
            return False

    async def cleanup(self):
        """Close the HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None