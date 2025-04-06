"""
Service layer for creating and managing Google Docs documents with RAG-generated content.
"""
import re
from googleapiclient.discovery import build
from google.oauth2 import service_account
from app.services.output.document_service import DocumentService

class DocsService(DocumentService):
    """Service for creating and managing Google Docs documents with RAG-generated content."""
    
    def __init__(self, credentials_path: str = None):
        """
        Initialize the Docs service.
        
        Args:
            credentials_path: Path to the Google service account credentials JSON file.
                          If None, will use the path from ChatConfig.
        """
        super().__init__(credentials_path)
        self._service = self._get_docs_service()
        
    def _get_docs_service(self):
        """Get an authenticated Google Docs service."""
        creds = service_account.Credentials.from_service_account_file(
            self.credentials_path,
            scopes=['https://www.googleapis.com/auth/documents', 
                   'https://www.googleapis.com/auth/drive']
        )
        return build('docs', 'v1', credentials=creds)
    
    async def create_document_from_csv(self, csv_path: str, title: str = "RAG Document", owner_email: str = None) -> str:
        """
        Create a Google Docs document from a CSV file containing questions and format templates.
        Content is generated using RAG with the specified format templates.
        
        Args:
            csv_path: Path to the CSV file containing questions and format templates
            title: Title for the document
            owner_email: Email address to share the document with
            
        Returns:
            The ID of the created document
        """
        # Read the CSV file
        question_format_pairs = self.read_csv(csv_path)
        
        # Create a new document
        doc = self._service.documents().create(
            body={'title': title}
        ).execute()
        document_id = doc.get('documentId')
        
        # Track the current insertion index
        current_index = 1  # Start after the automatically created title
        
        # Process each question and generate content
        for i, (question, format_template) in enumerate(question_format_pairs):
            try:
                self.logger.info(f"Generating formatted RAG content for question: {question}")
                self.logger.info(f"Using format template: {format_template}")
                
                # Get formatted content from RAG
                formatted_content = await self.get_formatted_content(question, format_template, "docs")
                
                # If this isn't the first section, add a page break
                if i > 0:
                    self._add_page_break(document_id, current_index)
                    current_index += 1
                
                # Insert the formatted content
                self._insert_text(document_id, current_index, formatted_content)
                
                # Update current index (approximate, as we don't know exactly how many characters were added)
                current_index += len(formatted_content)
                
            except Exception as e:
                self.logger.error(f"Error processing question {i}: {str(e)}")
                # Insert error message
                error_message = f"Error generating content for question: {question}\n\nError: {str(e)}"
                self._insert_text(document_id, current_index, error_message)
                current_index += len(error_message)
        
        # Apply formatting for markdown elements
        self._apply_markdown_formatting(document_id)
        
        # Share the document if an email is provided
        if owner_email:
            self.share_document(document_id, owner_email)
        
        return document_id
    
    def _insert_text(self, document_id: str, index: int, text: str):
        """
        Insert text into the document at the specified index.
        
        Args:
            document_id: ID of the document
            index: Index at which to insert the text
            text: Text to insert
        """
        self._service.documents().batchUpdate(
            documentId=document_id,
            body={
                'requests': [
                    {
                        'insertText': {
                            'location': {
                                'index': index
                            },
                            'text': text
                        }
                    }
                ]
            }
        ).execute()
    
    def _add_page_break(self, document_id: str, index: int):
        """
        Add a page break at the specified index.
        
        Args:
            document_id: ID of the document
            index: Index at which to add the page break
        """
        self._service.documents().batchUpdate(
            documentId=document_id,
            body={
                'requests': [
                    {
                        'insertPageBreak': {
                            'location': {
                                'index': index
                            }
                        }
                    }
                ]
            }
        ).execute()
    
    def _apply_markdown_formatting(self, document_id: str):
        """
        Apply formatting for markdown elements in the document.
        
        This method identifies and applies formatting to elements like:
        - Headings (# Title, ## Subtitle, etc.)
        - Bullet points
        - Numbered lists
        - Bold/italic text
        
        Args:
            document_id: ID of the document
        """
        # First, get the document content
        doc = self._service.documents().get(documentId=document_id).execute()
        
        # Prepare requests for formatting
        requests = []
        
        # Get document content
        content = doc.get('body', {}).get('content', [])
        
        # Process each paragraph to find markdown elements
        for element in content:
            if 'paragraph' in element:
                paragraph = element['paragraph']
                elements = paragraph.get('elements', [])
                start_index = elements[0].get('startIndex', 0) if elements else 0
                end_index = elements[-1].get('endIndex', 0) if elements else 0
                
                # Skip empty paragraphs
                if start_index == end_index:
                    continue
                
                # Get the text of the paragraph
                text = ''.join(e.get('textRun', {}).get('content', '') for e in elements)
                text = text.strip()
                
                # Process headings
                if text.startswith('# '):
                    # Heading 1
                    requests.append({
                        'updateParagraphStyle': {
                            'range': {
                                'startIndex': start_index,
                                'endIndex': end_index
                            },
                            'paragraphStyle': {
                                'namedStyleType': 'HEADING_1'
                            },
                            'fields': 'namedStyleType'
                        }
                    })
                    
                    # Remove the '# ' prefix
                    requests.append({
                        'deleteText': {
                            'range': {
                                'startIndex': start_index,
                                'endIndex': start_index + 2  # Length of '# '
                            }
                        }
                    })
                    
                elif text.startswith('## '):
                    # Heading 2
                    requests.append({
                        'updateParagraphStyle': {
                            'range': {
                                'startIndex': start_index,
                                'endIndex': end_index
                            },
                            'paragraphStyle': {
                                'namedStyleType': 'HEADING_2'
                            },
                            'fields': 'namedStyleType'
                        }
                    })
                    
                    # Remove the '## ' prefix
                    requests.append({
                        'deleteText': {
                            'range': {
                                'startIndex': start_index,
                                'endIndex': start_index + 3  # Length of '## '
                            }
                        }
                    })
                    
                elif text.startswith('### '):
                    # Heading 3
                    requests.append({
                        'updateParagraphStyle': {
                            'range': {
                                'startIndex': start_index,
                                'endIndex': end_index
                            },
                            'paragraphStyle': {
                                'namedStyleType': 'HEADING_3'
                            },
                            'fields': 'namedStyleType'
                        }
                    })
                    
                    # Remove the '### ' prefix
                    requests.append({
                        'deleteText': {
                            'range': {
                                'startIndex': start_index,
                                'endIndex': start_index + 4  # Length of '### '
                            }
                        }
                    })
                    
                # Process bullet points
                elif text.startswith('- ') or text.startswith('* '):
                    # Bullet point
                    requests.append({
                        'createParagraphBullets': {
                            'range': {
                                'startIndex': start_index,
                                'endIndex': end_index
                            },
                            'bulletPreset': 'BULLET_DISC_CIRCLE_SQUARE'
                        }
                    })
                    
                    # Remove the '- ' or '* ' prefix
                    prefix_length = 2  # Length of '- ' or '* '
                    requests.append({
                        'deleteText': {
                            'range': {
                                'startIndex': start_index,
                                'endIndex': start_index + prefix_length
                            }
                        }
                    })
                    
                # Process numbered lists
                elif re.match(r'^\d+\. ', text):
                    # Numbered list
                    requests.append({
                        'createParagraphBullets': {
                            'range': {
                                'startIndex': start_index,
                                'endIndex': end_index
                            },
                            'bulletPreset': 'NUMBERED_DECIMAL'
                        }
                    })
                    
                    # Remove the '1. ' prefix (and similar)
                    match = re.match(r'^\d+\. ', text)
                    if match:
                        prefix_length = len(match.group(0))
                        requests.append({
                            'deleteText': {
                                'range': {
                                    'startIndex': start_index,
                                    'endIndex': start_index + prefix_length
                                }
                            }
                        })
        
        # Apply all formatting requests
        if requests:
            self._service.documents().batchUpdate(
                documentId=document_id,
                body={'requests': requests}
            ).execute()