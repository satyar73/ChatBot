"""
Base service layer for document generation using RAG content.
This handles common functionality shared between Slides and Docs services.
"""
import csv
import uuid
from typing import List, Dict, Any, Tuple, Literal
from googleapiclient.discovery import build
from google.oauth2 import service_account
from app.config.chat_config import ChatConfig
from app.utils.logging_utils import get_logger
from app.services.chat.chat_service import ChatService
from app.models.chat_models import Message

class DocumentService:
    """Base service for generating content and handling document operations."""
    
    def __init__(self, credentials_path: str = None):
        """
        Initialize the base document service.
        
        Args:
            credentials_path: Path to the Google service account credentials JSON file.
                          If None, will use the path from ChatConfig.
        """
        self.config = ChatConfig()
        self.credentials_path = credentials_path or self.config.GOOGLE_DRIVE_CREDENTIALS_FILE
        self.chat_service = ChatService()
        self.logger = get_logger(__name__)
    
    def _get_drive_service(self):
        """Get an authenticated Google Drive service for sharing."""
        creds = service_account.Credentials.from_service_account_file(
            self.credentials_path,
            scopes=['https://www.googleapis.com/auth/drive']
        )
        return build('drive', 'v3', credentials=creds)
    
    def read_csv(self, csv_path: str) -> List[Tuple[str, str]]:
        """
        Read questions and format templates from a CSV file.
        The CSV should contain a 'question' column and optionally a 'format' column.
        
        Expected CSV format:
        question,format
        "What is...","Format template here"
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            List of tuples containing (question, format_template) pairs
        """
        items = []
        
        # Try different encodings if utf-8 fails
        encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings_to_try:
            try:
                self.logger.info(f"Trying to read CSV with encoding: {encoding}")
                with open(csv_path, 'r', encoding=encoding, errors='replace') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if 'question' in row:
                            # Get the format template, or empty string if not available
                            format_template = row.get('format', '')
                            # Clean up any potential problematic characters
                            format_template = format_template.replace('\ufffd', '-')
                            question = row['question']
                            items.append((question, format_template))
                
                # If we got here without an exception, we succeeded
                self.logger.info(f"Successfully read CSV with encoding: {encoding}")
                break
            except UnicodeDecodeError as e:
                self.logger.warning(f"Failed to read CSV with encoding {encoding}: {str(e)}")
                # If all encodings fail, we'll return an empty list
                continue
            
        return items
    
    async def get_formatted_content(self, question: str, format_template: str, document_type: Literal["slides", "docs"]) -> str:
        """
        Get content from RAG with the specified formatting instructions.
        First checks if the question has a predefined answer in qagold data.
        If found, uses that answer to enhance the prompt.
        
        Args:
            question: The question to ask
            format_template: Format template to structure the response
            document_type: Type of document ("slides" or "docs")
            
        Returns:
            Formatted content from RAG
        """
        try:
            # Create a unique session ID for this interaction
            session_id = f"{document_type}_gen_{uuid.uuid4().hex[:8]}"
            
            # Let the chat service handle expected answers and system prompts
            self.logger.info(f"Using RAG mode - chat service will handle system prompt and expected answers")
            
            # Create formatting instructions for the user message
            if document_type == "slides":
                format_instructions = """
                Format your response according to the provided template.
                Aim to make each slide as concise as possible - essential information only.
                If content requires multiple slides, mark continuation slides with 'Title: [Original Title] (Contd.)'
                Use "===SLIDE X===" markers to separate slides.
                Each slide should have a "Title:" section followed by content.
                Use bullet points rather than paragraphs whenever possible.
                Do not include prefixes like "Question:", "Answer:", or "Body:" in the final output.
                Avoid complete sentences when bullet points would be clearer.
                """
            elif document_type == "docs":
                format_instructions = """
                Format your response according to the provided template.
                Follow the structure exactly, using the template sections as a guide.
                Use markdown formatting for headings (# for main headings, ## for subheadings).
                For sections that require bullet points, use markdown list syntax.
                Ensure the document flows naturally between sections.
                Do not include prefixes like "Question:", "Answer:", or "Body:" in the final output.
                """
            else:
                format_instructions = """
                Format your response according to the provided template.
                Follow the structure exactly, using the template sections as a guide.
                """
            
            # Process placeholders in the format template
            if format_template and question:
                # Replace any {question} placeholders in the format_template
                format_template = format_template.replace("{question}", question)
            
            # Create a user message that contains only the formatting instructions and template
            # For document service, we want to avoid putting the question twice in the prompt
            # The question will be handled separately through metadata for expected answers
            user_message = f"""{format_instructions}

Format template to follow:
{format_template}
"""
            
            # If we have a question, and it's not already in the format template, add it to the format template directly
            if question and question not in format_template:
                # Insert the question at the beginning of the message as metadata rather than in the format template
                # This helps prevent duplication while keeping the question context available
                user_message = f"Creating slides for question: {question}\n\n{user_message}"
            
            # Add the document question identifier to a metadata field that the chat_service will pass through
            # This will allow enhancement_service to find the question without it being duplicated in the message
            message_metadata = {}
            if question:
                message_metadata["document_question"] = question
            
            # Create a message without specifying system prompt (let chat service use default)
            message = Message(
                message=user_message,     # Formatting instructions and question
                session_id=session_id,
                mode="rag",              # Use RAG mode to get the best context-aware response
                system_prompt=None,      # Use None to get the default system prompt
                prompt_style="detailed", # Use detailed style for document generation
                metadata=message_metadata  # Pass the document question in metadata
            )
            
            # Get response from the chat service
            response = await self.chat_service.chat(message)
            
            # Extract and return the response text
            return response.response.output
            
        except Exception as e:
            self.logger.error(f"Error getting formatted content for question: {question}. Error: {str(e)}")
            return f"Error generating content: {str(e)}"
    
    def share_document(self, document_id: str, email: str = None, document_type: str = "document") -> Dict[str, Any]:
        """
        Share a document with a user and make it accessible via link.
        
        Args:
            document_id: ID of the document to share
            email: Optional email address to share the document with
            document_type: Type of document ("document" or "presentation") for logging purposes
            
        Returns:
            Dictionary with sharing status and details
        """
        drive_service = self._get_drive_service()
        
        # Make the document accessible with a link
        anyone_permission = {
            'type': 'anyone',
            'role': 'writer',
            'allowFileDiscovery': False
        }
        
        # Add the public permission
        drive_service.permissions().create(
            fileId=document_id,
            body=anyone_permission,
            fields='id'
        ).execute()
        
        self.logger.info(f"Made {document_type} {document_id} accessible via link")
        
        # Share with editor access if email is provided
        if email:
            try:
                editor_permission = {
                    'type': 'user',
                    'role': 'writer',  # 'writer' provides edit access
                    'emailAddress': email
                }
                
                # Add the permission to the file
                drive_service.permissions().create(
                    fileId=document_id,
                    body=editor_permission,
                    fields='id',
                    sendNotificationEmail=True
                ).execute()
                
                self.logger.info(f"Granted editor access to {email} for {document_type} {document_id}")
                return {"status": "shared", "email": email}
            except Exception as e:
                self.logger.error(f"Error sharing {document_type} with {email}: {str(e)}")
                return {"status": "public_only", "error": str(e)}
        
        return {"status": "public_only"}