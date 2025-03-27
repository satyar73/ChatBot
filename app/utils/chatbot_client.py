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
        
    async def __aenter__(self):
        """Support for async context manager."""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup when used as async context manager."""
        await self.cleanup()

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an HTTP session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                # Add timeout to prevent session from hanging indefinitely
                timeout=aiohttp.ClientTimeout(total=30)
            )
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
            "mode" : "both"  # Use "both" to get both RAG and non-RAG responses
        }

        try:
            async with session.post(f"{self.api_url}/chat/", json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API request failed with status {response.status}: {error_text}")

                data = await response.json()

                # Extract responses from the response data
                response_obj = data.get("response", {})
                if not response_obj:
                    return "No response data received", "No response data received"
                    
                rag_response = response_obj.get("output")
                no_rag_response = response_obj.get("no_rag_output")
                
                # Ensure neither response is None
                rag_response = rag_response or "No RAG response received"
                no_rag_response = no_rag_response or "No non-RAG response received"

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
            try:
                await self.session.close()
            except Exception as e:
                print(f"Error closing session: {str(e)}")
            finally:
                self.session = None