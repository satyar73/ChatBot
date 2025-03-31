import csv
import re
import asyncio
import uuid
import os
from typing import List, Dict, Any, Optional, Tuple
from googleapiclient.discovery import build
from google.oauth2 import service_account
from app.config.chat_config import ChatConfig
from app.utils.logging_utils import get_logger
from app.services.chat_service import ChatService
from app.models.chat_models import Message

class SlidesService:
    """Service for creating and managing Google Slides presentations."""
    
    def __init__(self, credentials_path: str = None):
        """
        Initialize the Slides service.
        
        Args:
            credentials_path: Path to the Google service account credentials JSON file.
                            If None, will use the path from ChatConfig.
        """
        self.config = ChatConfig()
        self.credentials_path = credentials_path or self.config.GOOGLE_DRIVE_CREDENTIALS_FILE
        self._service = self._get_slides_service()
        self.chat_service = ChatService()
        self.logger = get_logger(__name__)
        
    def _get_slides_service(self):
        """Get an authenticated Google Slides service."""
        creds = service_account.Credentials.from_service_account_file(
            self.credentials_path,
            scopes=['https://www.googleapis.com/auth/presentations', 
                    'https://www.googleapis.com/auth/drive']
        )
        return build('slides', 'v1', credentials=creds)
        
    def _get_drive_service(self):
        """Get an authenticated Google Drive service for sharing."""
        creds = service_account.Credentials.from_service_account_file(
            self.credentials_path,
            scopes=['https://www.googleapis.com/auth/drive']
        )
        return build('drive', 'v3', credentials=creds)
    
    async def create_presentation_from_csv(self, csv_path: str, title: str = "Q&A Presentation", author_name: str = None, use_rag: bool = True) -> str:
        """
        Create a Google Slides presentation from a CSV file containing questions.
        Answers can be generated using RAG or taken from the CSV.
        
        Args:
            csv_path: Path to the CSV file containing questions (and optional answers)
            title: Title for the presentation
            author_name: Name of the author to include on the title slide
            use_rag: Whether to use RAG to generate answers (True) or use answers from CSV (False)
            
        Returns:
            The ID of the created presentation
        """
        # Read the CSV file
        qa_pairs = self._read_csv(csv_path)
        
        # Create a new presentation
        presentation = self._service.presentations().create(
            body={'title': title}
        ).execute()
        presentation_id = presentation.get('presentationId')
        
        # Create a title slide
        self._create_title_slide(presentation_id, title, author_name)
        
        # Create slides for each question
        for i, (question, fallback_answer) in enumerate(qa_pairs, start=1):
            # Get answer from RAG if enabled, otherwise use the provided answer
            if use_rag:
                try:
                    self.logger.info(f"Generating RAG answer for question: {question}")
                    answer = await self._get_answer_from_rag(question)
                    # If RAG fails or returns empty, use fallback
                    if not answer or answer.startswith("Error generating answer"):
                        self.logger.warning(f"Using fallback answer for question: {question}")
                        answer = fallback_answer
                except Exception as e:
                    self.logger.error(f"Error in RAG processing: {str(e)}")
                    answer = fallback_answer
            else:
                answer = fallback_answer
                
            # Create the slide with the question and answer
            self._create_qa_slide(
                    presentation_id=presentation_id,
                    question=question,
                    answer=answer,
                    slide_number=i)
            
        return presentation_id
        
    def _create_title_slide(self, presentation_id: str, title: str, author_name: str = None):
        """
        Create a title slide with the presentation title and optional author name.
        
        Args:
            presentation_id: ID of the presentation
            title: Title for the presentation
            author_name: Optional name of the author
        """
        # Step 1: Create the title slide
        create_slide_request = [{
            'createSlide': {
                'objectId': 'title_slide',
                'insertionIndex': 0,
                'slideLayoutReference': {
                    'predefinedLayout': 'TITLE'
                }
            }
        }]
        
        # Execute the slide creation
        self._service.presentations().batchUpdate(
            presentationId=presentation_id,
            body={'requests': create_slide_request}
        ).execute()
        
        # Step 2: Get the slide details to find existing shape IDs
        slide = self._service.presentations().get(
            presentationId=presentation_id,
            fields='slides'
        ).execute()
        
        # Get the current slide
        current_slide = None
        for s in slide.get('slides', []):
            if s.get('objectId') == 'title_slide':
                current_slide = s
                break
        
        # If we couldn't find the slide, use a different approach
        if not current_slide:
            # Alternative approach: Create text boxes directly
            title_box_id = 'title_box'
            subtitle_box_id = 'subtitle_box'
            
            text_boxes_request = [
                # Create a text box for the title
                {
                    'createShape': {
                        'objectId': title_box_id,
                        'shapeType': 'TEXT_BOX',
                        'elementProperties': {
                            'pageObjectId': 'title_slide',
                            'size': {
                                'width': {'magnitude': 500, 'unit': 'PT'},
                                'height': {'magnitude': 100, 'unit': 'PT'}
                            },
                            'transform': {
                                'scaleX': 1,
                                'scaleY': 1,
                                'translateX': 50,
                                'translateY': 100,
                                'unit': 'PT'
                            }
                        }
                    }
                },
                # Insert title text
                {
                    'insertText': {
                        'objectId': title_box_id,
                        'insertionIndex': 0,
                        'text': title
                    }
                },
                # Format the title text
                {
                    'updateTextStyle': {
                        'objectId': title_box_id,
                        'textRange': {
                            'type': 'ALL'
                        },
                        'style': {
                            'fontSize': {
                                'magnitude': 28,
                                'unit': 'PT'
                            },
                            'bold': True
                        },
                        'fields': 'fontSize,bold'
                    }
                }
            ]
            
            # Add author name if provided
            if author_name:
                author_requests = [
                    # Create a text box for the author
                    {
                        'createShape': {
                            'objectId': subtitle_box_id,
                            'shapeType': 'TEXT_BOX',
                            'elementProperties': {
                                'pageObjectId': 'title_slide',
                                'size': {
                                    'width': {'magnitude': 400, 'unit': 'PT'},
                                    'height': {'magnitude': 50, 'unit': 'PT'}
                                },
                                'transform': {
                                    'scaleX': 1,
                                    'scaleY': 1,
                                    'translateX': 50,
                                    'translateY': 200,
                                    'unit': 'PT'
                                }
                            }
                        }
                    },
                    # Insert author text
                    {
                        'insertText': {
                            'objectId': subtitle_box_id,
                            'insertionIndex': 0,
                            'text': f"By: {author_name}"
                        }
                    },
                    # Format the author text
                    {
                        'updateTextStyle': {
                            'objectId': subtitle_box_id,
                            'textRange': {
                                'type': 'ALL'
                            },
                            'style': {
                                'fontSize': {
                                    'magnitude': 16,
                                    'unit': 'PT'
                                },
                                'italic': True
                            },
                            'fields': 'fontSize,italic'
                        }
                    }
                ]
                text_boxes_request.extend(author_requests)
            
            # Execute the text box creation and text insertion
            self._service.presentations().batchUpdate(
                presentationId=presentation_id,
                body={'requests': text_boxes_request}
            ).execute()
            return
        
        # Find title and subtitle placeholder IDs from the slide
        title_id = None
        subtitle_id = None
        
        for element in current_slide.get('pageElements', []):
            if 'shape' in element and 'placeholder' in element['shape']:
                placeholder_type = element['shape']['placeholder']['type']
                if placeholder_type == 'TITLE':
                    title_id = element['objectId']
                elif placeholder_type == 'SUBTITLE':
                    subtitle_id = element['objectId']
        
        # Prepare text insertion requests
        text_requests = []
        
        if title_id:
            text_requests.append({
                'insertText': {
                    'objectId': title_id,
                    'insertionIndex': 0,
                    'text': title
                }
            })
            
            # Apply formatting to the title text
            text_requests.append({
                'updateTextStyle': {
                    'objectId': title_id,
                    'textRange': {
                        'type': 'ALL'
                    },
                    'style': {
                        'bold': True,
                        'fontSize': {
                            'magnitude': 28,
                            'unit': 'PT'
                        }
                    },
                    'fields': 'bold,fontSize'
                }
            })
        
        if subtitle_id and author_name:
            text_requests.append({
                'insertText': {
                    'objectId': subtitle_id,
                    'insertionIndex': 0,
                    'text': f"By: {author_name}"
                }
            })
            
            # Apply formatting to the subtitle text
            text_requests.append({
                'updateTextStyle': {
                    'objectId': subtitle_id,
                    'textRange': {
                        'type': 'ALL'
                    },
                    'style': {
                        'italic': True,
                        'fontSize': {
                            'magnitude': 16,
                            'unit': 'PT'
                        }
                    },
                    'fields': 'italic,fontSize'
                }
            })
        
        # Execute the text insertion
        if text_requests:
            self._service.presentations().batchUpdate(
                presentationId=presentation_id,
                body={'requests': text_requests}
            ).execute()
    
    async def _get_answer_from_rag(self, question: str) -> str:
        """
        Get an answer to a question using the RAG chatbot.
        
        Args:
            question: The question to ask
            
        Returns:
            The answer generated by the RAG system
        """
        try:
            # Create a unique session ID for this interaction
            session_id = f"slides_gen_{uuid.uuid4().hex[:8]}"
            
            # Create a message object for the chat service
            message = Message(
                text=question,
                session_id=session_id,
                mode="rag"  # Use RAG mode to get the best context-aware response
            )
            
            # Get response from the chat service
            response = await self.chat_service.chat(message)
            
            # Extract and return just the RAG response text
            return response.rag_response
            
        except Exception as e:
            self.logger.error(f"Error getting RAG answer for question: {question}. Error: {str(e)}")
            return f"Error generating answer: {str(e)}"

    def _read_csv(self, csv_path: str) -> list[tuple[str | Any, str | Any]]:
        """
        Read questions from a CSV file. The CSV should contain at least a 'question' column.
        An optional 'answer' column can be included as a fallback.
        
        Expected CSV format:
        question,answer
        "What is...","The answer is..."
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            List of tuples containing (question, fallback_answer) pairs
        """
        qa_pairs = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'question' in row:
                    # Use the provided answer as a fallback, or empty string if not available
                    fallback_answer = row.get('answer', '')
                    qa_pairs.append((row['question'], fallback_answer))
        return qa_pairs


    def _parse_markdown(self, text: str) -> List[Dict]:
        """
        Parse Markdown text into structured content for slides.

        Supported markdown features:
        - Bullet lists (starting with '- ' or '* ')
        - Numbered lists (starting with '1. ', '2. ', etc.)
        - Bold text using **bold** syntax
        - Italic text using *italic* syntax
        - Links using [text](url) syntax

        Args:
            text: Markdown formatted text

        Returns:
            List of dictionaries with content and formatting info
        """
        self.logger.info(f"Starting to parse markdown text: {text[:50]}...")

        result = []

        # Handle newlines consistently
        text = text.replace('\r\n', '\n')

        # Split the text into paragraphs (blocks separated by blank lines)
        paragraphs = re.split(r'\n\s*\n', text)
        self.logger.info(f"Split into {len(paragraphs)} paragraphs")

        for paragraph_idx, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            self.logger.info(f"Processing paragraph {paragraph_idx}: {paragraph[:50]}...")

            # Detect paragraph type based on its first line
            lines = paragraph.split('\n')
            first_line = lines[0].strip() if lines else ""

            # Check if it's a bullet point list (starts with - or *)
            if first_line.startswith('- ') or first_line.startswith('* '):
                self.logger.info(f"Paragraph {paragraph_idx} is a bullet list")
                items = []
                current_item = ""
                indent_level = 0

                for line_idx, line in enumerate(lines):
                    # Capture indentation before stripping
                    indentation = len(line) - len(line.lstrip())
                    line_content = line.strip()

                    # Skip empty lines
                    if not line_content:
                        continue

                    # Check if this is a new bullet point
                    if line_content.startswith('- ') or line_content.startswith('* '):
                        self.logger.info(f"Line {line_idx} is a bullet point: {line_content[:30]}...")

                        # Determine nesting level (0, 1, 2, etc.)
                        nest_level = indentation // 2  # Assuming 2 spaces per level
                        self.logger.info(f"Detected nesting level: {nest_level}")

                        # If we already have content, save the previous item
                        if current_item:
                            self.logger.info(f"Processing previous bullet item: {current_item[:30]}...")
                            # Process markdown in bullet points
                            processed_item = self._process_inline_formatting_v2(current_item)
                            items.append({
                                'text': processed_item['text'],
                                'level': indent_level,
                                'formats': processed_item.get('formats', []),
                                'links': processed_item.get('links', [])
                            })
                            current_item = ""

                        # Update the current indentation level
                        indent_level = nest_level

                        # Start a new item (removing the bullet marker)
                        if line_content.startswith('- '):
                            current_item = line_content[2:]
                        else:  # starts with *
                            current_item = line_content[2:]
                    else:
                        # This is a continuation of the current item
                        self.logger.info(f"Line {line_idx} is continuation of bullet: {line_content[:30]}...")
                        current_item += " " + line_content

                # Add the last item if there is one
                if current_item:
                    self.logger.info(f"Processing final bullet item: {current_item[:30]}...")
                    # Process markdown in bullet points
                    processed_item = self._process_inline_formatting_v2(current_item)
                    items.append({
                        'text': processed_item['text'],
                        'level': indent_level,
                        'formats': processed_item.get('formats', []),
                        'links': processed_item.get('links', [])
                    })

                # Add the bullet list to the result
                if items:
                    self.logger.info(f"Adding bullet list with {len(items)} items to result")
                    result.append({
                        'type': 'bullet_list',
                        'items': items
                    })
            # Regular paragraph
            else:
                self.logger.info(f"Paragraph {paragraph_idx} is a regular paragraph")
                # Process inline formatting (bold, italic, links) for later application in slides
                processed_paragraph = self._process_inline_formatting_v2(paragraph)

                self.logger.info(f"Processed paragraph: text={processed_paragraph['text'][:50]}...")
                self.logger.info(f"Found {len(processed_paragraph.get('formats', []))} format elements")
                self.logger.info(f"Found {len(processed_paragraph.get('links', []))} links")

                result.append({
                    'type': 'paragraph',
                    'text': processed_paragraph['text'],
                    'formats': processed_paragraph.get('formats', []),
                    'links': processed_paragraph.get('links', [])
                })

        self.logger.info(f"Finished parsing markdown, found {len(result)} content blocks")
        return result


    def _process_inline_formatting_v2(self, text: str) -> Dict:
        """
        Process inline Markdown formatting including bold, italic, and links.
        This is a completely rewritten version that correctly handles positioning.

        Args:
            text: Text with Markdown formatting

        Returns:
            Dictionary with processed text and formatting instructions
        """
        self.logger.info(f"Processing inline formatting for: {text[:50]}...")

        # We'll use a two-pass approach:
        # 1. First scan: identify all markdown patterns and their positions
        # 2. Second pass: remove markdown syntax and track position changes

        # Track all formatting elements we find
        formats = []

        # Find bold patterns: **text**
        bold_pattern = re.compile(r'\*\*(.*?)\*\*')
        for match in bold_pattern.finditer(text):
            self.logger.info(f"Found bold text: '{match.group(1)}' at positions {match.start()}-{match.end()}")
            formats.append({
                'type': 'bold',
                'start': match.start(),
                'end': match.end(),
                'inner_start': match.start() + 2,  # Skip the opening **
                'inner_end': match.end() - 2,  # Skip the closing **
                'inner_text': match.group(1)
            })

        # Find italic patterns: *text* (but not part of bold **)
        italic_pattern = re.compile(r'(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)')
        for match in italic_pattern.finditer(text):
            self.logger.info(f"Found italic text: '{match.group(1)}' at "
                              f"positions {match.start()}-{match.end()}")
            formats.append({
                'type': 'italic',
                'start': match.start(),
                'end': match.end(),
                'inner_start': match.start() + 1,  # Skip the opening *
                'inner_end': match.end() - 1,  # Skip the closing *
                'inner_text': match.group(1)
            })

        # Find link patterns: [text](url)
        link_pattern = re.compile(r'\[(.*?)\]\((.*?)\)')
        for match in link_pattern.finditer(text):
            self.logger.info(
                f"Found link: '{match.group(1)}' with URL '{match.group(2)}' "
                f"at positions {match.start()}-{match.end()}")
            formats.append({
                'type': 'link',
                'start': match.start(),
                'end': match.end(),
                'inner_start': match.start() + 1,  # Skip the opening [
                'inner_end': match.start() + 1 + len(match.group(1)),  # End of link text
                'inner_text': match.group(1),
                'url': match.group(2)
            })

        # Sort all formatting elements by their start position (descending)
        # We process from the end of the string backward to avoid position shifts
        formats.sort(key=lambda x: x['start'], reverse=True)
        self.logger.info(f"Sorted {len(formats)} formatting elements for processing")

        # Now process the text, removing markdown syntax
        clean_text = text

        # These will track the final formatting positions after markup removal
        final_formats = []
        final_links = []

        # Keep track of how much text we've removed at each position
        offset_map = {}  # position -> offset

        # Process each format, from the end of the string backward
        for fmt_idx, fmt in enumerate(formats):
            self.logger.info(f"Processing format {fmt_idx}: type={fmt['type']}, text='{fmt['inner_text']}'")

            # Calculate the offsets at the start and end positions
            offset_at_start = sum(offset_map.get(pos, 0) for pos in range(fmt['start'] + 1))
            offset_at_inner_start = sum(offset_map.get(pos, 0) for pos in range(fmt['inner_start'] + 1))
            offset_at_inner_end = sum(offset_map.get(pos, 0) for pos in range(fmt['inner_end'] + 1))
            offset_at_end = sum(offset_map.get(pos, 0) for pos in range(fmt['end'] + 1))

            self.logger.info(
                f"Calculated offsets: start={offset_at_start},"
                f" inner_start={offset_at_inner_start}, inner_end={offset_at_inner_end}, end={offset_at_end}")

            # Calculate new positions
            new_start = fmt['start'] - offset_at_start
            new_inner_start = fmt['inner_start'] - offset_at_inner_start
            new_inner_end = fmt['inner_end'] - offset_at_inner_end
            new_end = fmt['end'] - offset_at_end

            self.logger.info(
                f"New positions: start={new_start}, inner_start={new_inner_start}, inner_end={new_inner_end}, end={new_end}")

            # Amount of markup to remove
            markup_len = (fmt['inner_start'] - fmt['start']) + (fmt['end'] - fmt['inner_end'])
            self.logger.info(f"Markup length to remove: {markup_len}")

            # Text before modification
            self.logger.info(f"Text before modification: '{clean_text[:50]}...'")

            # Remove the markdown syntax
            if fmt['type'] == 'link':
                # For links, replace [text](url) with just text
                link_markup_before = fmt['inner_start'] - fmt['start']  # Length of "["
                link_markup_middle = (fmt['end'] - fmt['inner_end']) - 1  # Length of "](url)"

                self.logger.info(f"Link markup: before={link_markup_before}, middle={link_markup_middle}")
                self.logger.info(
                    f"Removing closing part:"
                    f" '{clean_text[new_inner_end:new_inner_end + link_markup_middle + 1]}'")

                # Remove the closing part first (to avoid position shifts)
                clean_text = clean_text[:new_inner_end] + clean_text[new_inner_end + link_markup_middle + 1:]

                self.logger.info(f"Text after removing closing part: '{clean_text[:50]}...'")
                self.logger.info(f"Removing opening part:"
                                  f" '{clean_text[new_start:new_start + link_markup_before]}'")

                # Then remove the opening part
                clean_text = clean_text[:new_start] + clean_text[new_start + link_markup_before:]

                self.logger.info(f"Text after removing opening part: '{clean_text[:50]}...'")

                # Update offset map for future formats
                for i in range(fmt['start'], fmt['end'] + 1):
                    offset_map[i] = offset_map.get(i, 0) + markup_len

                # Store link for later application
                link_info = {
                    'url': fmt['url'],
                    'start_pos': new_start,
                    'end_pos': new_inner_end - link_markup_before
                }
                self.logger.info(f"Adding link: {link_info}")
                final_links.append(link_info)
            else:
                # For bold/italic, remove the markers
                marker_len = 2 if fmt['type'] == 'bold' else 1

                self.logger.info(f"Marker length: {marker_len}")
                self.logger.info(f"Removing closing marker:"
                                  f" '{clean_text[new_inner_end:new_inner_end + marker_len]}'")

                # Remove the closing marker first
                clean_text = clean_text[:new_inner_end] + clean_text[new_inner_end + marker_len:]

                self.logger.info(f"Text after removing closing marker: '{clean_text[:50]}...'")
                self.logger.info(f"Removing opening marker:"
                                  f" '{clean_text[new_start:new_start + marker_len]}'")

                # Then remove the opening marker
                clean_text = clean_text[:new_start] + clean_text[new_start + marker_len:]

                self.logger.info(f"Text after removing opening marker: '{clean_text[:50]}...'")

                # Update offset map for future formats
                for i in range(fmt['start'], fmt['end'] + 1):
                    offset_map[i] = offset_map.get(i, 0) + markup_len

                # Store formatting for later application
                format_info = {
                    'type': fmt['type'],
                    'start_pos': new_start,
                    'end_pos': new_inner_end - marker_len
                }
                self.logger.info(f"Adding format: {format_info}")
                final_formats.append(format_info)

        result = {
            'text': clean_text,
            'formats': final_formats,
            'links': final_links
        }

        self.logger.info(f"Final clean text: '{clean_text[:50]}...'")
        self.logger.info(f"Final formats: {final_formats}")
        self.logger.info(f"Final links: {final_links}")

        return result


    def _create_qa_slide(self, presentation_id: str, question: str, answer: str, slide_number: int):
        """
        Create a slide with a question and answer.

        Args:
            presentation_id: ID of the presentation
            question: The question text
            answer: The answer text
            slide_number: The slide number
        """
        self.logger.info(f"Creating slide {slide_number}")
        self.logger.info(f"Question: {question[:50]}...")
        self.logger.info(f"Answer: {answer[:50]}...")

        # Parse the answer as markdown
        parsed_answer = self._parse_markdown(answer)
        self.logger.info(f"Parsed answer contains {len(parsed_answer)} blocks")

        # Step 1: Create the slide
        create_slide_request = [{
            'createSlide': {
                'objectId': f'qa_slide_{slide_number}',
                'insertionIndex': slide_number,
                'slideLayoutReference': {
                    'predefinedLayout': 'TITLE_AND_BODY'
                }
            }
        }]

        # Execute the slide creation
        self._service.presentations().batchUpdate(
            presentationId=presentation_id,
            body={'requests': create_slide_request}
        ).execute()
        self.logger.info(f"Slide {slide_number} created")

        # Step 2: Get the slide details to find existing shape IDs
        slide = self._service.presentations().get(
            presentationId=presentation_id,
            fields='slides'
        ).execute()

        # Get the current slide
        current_slide = None
        for s in slide.get('slides', []):
            if s.get('objectId') == f'qa_slide_{slide_number}':
                current_slide = s
                break

        # If we couldn't find the slide, use a different approach
        if not current_slide:
            self.logger.info(f"Could not find slide {slide_number}, using fallback approach")
            # Alternative approach: Create text boxes directly
            question_box_id = f'question_box_{slide_number}'
            answer_box_id = f'answer_box_{slide_number}'

            text_boxes_request = [
                # Create a text box for the question at the top
                {
                    'createShape': {
                        'objectId': question_box_id,
                        'shapeType': 'TEXT_BOX',
                        'elementProperties': {
                            'pageObjectId': f'qa_slide_{slide_number}',
                            'size': {
                                'width': {'magnitude': 350, 'unit': 'PT'},
                                'height': {'magnitude': 100, 'unit': 'PT'}
                            },
                            'transform': {
                                'scaleX': 1,
                                'scaleY': 1,
                                'translateX': 50,
                                'translateY': 50,
                                'unit': 'PT'
                            }
                        }
                    }
                },
                # Insert question text into the text box
                {
                    'insertText': {
                        'objectId': question_box_id,
                        'insertionIndex': 0,
                        'text': f'Question {slide_number}:\n{question}'
                    }
                },
                # Create a text box for the answer below
                {
                    'createShape': {
                        'objectId': answer_box_id,
                        'shapeType': 'TEXT_BOX',
                        'elementProperties': {
                            'pageObjectId': f'qa_slide_{slide_number}',
                            'size': {
                                'width': {'magnitude': 350, 'unit': 'PT'},
                                'height': {'magnitude': 200, 'unit': 'PT'}
                            },
                            'transform': {
                                'scaleX': 1,
                                'scaleY': 1,
                                'translateX': 50,
                                'translateY': 170,
                                'unit': 'PT'
                            }
                        }
                    }
                },
                # Insert answer text into the text box - simple rendering without markdown parsing
                # In this fallback case, we'll just insert the raw text without Markdown formatting
                {
                    'insertText': {
                        'objectId': answer_box_id,
                        'insertionIndex': 0,
                        'text': f'Answer:\n{answer.replace("**", "").replace("*", "")}'  # Simple markdown stripping
                    }
                },
                # Format the question text as title
                {
                    'updateTextStyle': {
                        'objectId': question_box_id,
                        'textRange': {
                            'type': 'ALL'
                        },
                        'style': {
                            'fontSize': {
                                'magnitude': 16,
                                'unit': 'PT'
                            },
                            'bold': True
                        },
                        'fields': 'fontSize,bold'
                    }
                },
                # Format the answer text as gray
                {
                    'updateTextStyle': {
                        'objectId': answer_box_id,
                        'textRange': {
                            'type': 'ALL'
                        },
                        'style': {
                            'fontSize': {
                                'magnitude': 16,
                                'unit': 'PT'
                            },
                            'foregroundColor': {
                                'opaqueColor': {
                                    'rgbColor': {
                                        'red': 0.5,
                                        'green': 0.5,
                                        'blue': 0.5
                                    }
                                }
                            }
                        },
                        'fields': 'fontSize,foregroundColor'
                    }
                }
            ]

            # Execute the text box creation and text insertion
            self._service.presentations().batchUpdate(
                presentationId=presentation_id,
                body={'requests': text_boxes_request}
            ).execute()
            self.logger.info(f"Created slide {slide_number} using fallback approach")
            return

        # Find title and body placeholder IDs from the slide
        title_id = None
        body_id = None

        for element in current_slide.get('pageElements', []):
            if 'shape' in element and 'placeholder' in element['shape']:
                placeholder_type = element['shape']['placeholder']['type']
                if placeholder_type == 'TITLE':
                    title_id = element['objectId']
                elif placeholder_type == 'BODY':
                    body_id = element['objectId']

        self.logger.info(f"Found title_id={title_id}, body_id={body_id}")

        # Prepare text insertion requests
        text_requests = []

        if title_id:
            text_requests.append({
                'insertText': {
                    'objectId': title_id,
                    'insertionIndex': 0,
                    'text': f'Question {slide_number}: {question}'
                }
            })

            # Apply bold and size 16pt to the title text
            text_requests.append({
                'updateTextStyle': {
                    'objectId': title_id,
                    'textRange': {
                        'type': 'ALL'
                    },
                    'style': {
                        'bold': True,
                        'fontSize': {
                            'magnitude': 16,
                            'unit': 'PT'
                        }
                    },
                    'fields': 'bold,fontSize'
                }
            })

        if body_id:
            # First insert the "Answer:" label
            text_requests.append({
                'insertText': {
                    'objectId': body_id,
                    'insertionIndex': 0,
                    'text': 'Answer:\n'
                }
            })

            insertion_index = len('Answer:\n')

            # Process the markdown content
            for block_idx, content in enumerate(parsed_answer):
                self.logger.info(f"Processing content block {block_idx}, type={content['type']}")

                if content['type'] == 'paragraph':
                    # Add the paragraph text
                    text_requests.append({
                        'insertText': {
                            'objectId': body_id,
                            'insertionIndex': insertion_index,
                            'text': content['text'] + '\n\n'
                        }
                    })

                    self.logger.info(f"Added paragraph text: '{content['text'][:50]}...'")
                    self.logger.info(f"Insertion index now at: {insertion_index}")

                    # Apply formatting for formatted segments (bold, italic)
                    current_text_length = len(content['text']) + 2  # +2 for the newlines

                    if 'formats' in content and content['formats']:
                        for fmt_idx, format_info in enumerate(content['formats']):
                            # Get the adjusted positions
                            start_pos = format_info.get('start_pos', 0)
                            end_pos = format_info.get('end_pos', 0)

                            self.logger.info(
                                f"Applying format {fmt_idx}: type={format_info['type']}, start={start_pos}, end={end_pos}")

                            # Apply formatting
                            if format_info['type'] == 'bold':
                                text_requests.append({
                                    'updateTextStyle': {
                                        'objectId': body_id,
                                        'textRange': {
                                            'type': 'FIXED_RANGE',
                                            'startIndex': insertion_index + start_pos,
                                            'endIndex': insertion_index + end_pos
                                        },
                                        'style': {
                                            'bold': True
                                        },
                                        'fields': 'bold'
                                    }
                                })
                                self.logger.info(
                                    f"Applied bold formatting from {insertion_index + start_pos} to {insertion_index + end_pos}")
                            elif format_info['type'] == 'italic':
                                text_requests.append({
                                    'updateTextStyle': {
                                        'objectId': body_id,
                                        'textRange': {
                                            'type': 'FIXED_RANGE',
                                            'startIndex': insertion_index + start_pos,
                                            'endIndex': insertion_index + end_pos
                                        },
                                        'style': {
                                            'italic': True
                                        },
                                        'fields': 'italic'
                                    }
                                })
                                self.logger.info(
                                    f"Applied italic formatting from {insertion_index + start_pos} to {insertion_index + end_pos}")

                    # Apply links
                    if 'links' in content and content['links']:
                        for link_idx, link_info in enumerate(content['links']):
                            # Get the adjusted positions
                            start_pos = link_info.get('start_pos', 0)
                            end_pos = link_info.get('end_pos', 0)

                            self.logger.info(
                                f"Applying link {link_idx}: url={link_info['url']}, start={start_pos}, end={end_pos}")

                            # Create hyperlink in the slide
                            text_requests.append({
                                'updateTextStyle': {
                                    'objectId': body_id,
                                    'textRange': {
                                        'type': 'FIXED_RANGE',
                                        'startIndex': insertion_index + start_pos,
                                        'endIndex': insertion_index + end_pos
                                    },
                                    'style': {
                                        'link': {
                                            'url': link_info['url']
                                        },
                                        'foregroundColor': {
                                            'opaqueColor': {
                                                'rgbColor': {
                                                    'red': 0.0,
                                                    'green': 0.0,
                                                    'blue': 0.8
                                                }
                                            }
                                        },
                                        'underline': True
                                    },
                                    'fields': 'link,foregroundColor,underline'
                                }
                            })
                            self.logger.info(f"Applied link from {insertion_index + start_pos} to {insertion_index + end_pos}")

                    # Advance insertion index
                    insertion_index += len(content['text']) + 2
                    self.logger.info(f"Advanced insertion index to {insertion_index}")

                elif content['type'] in ['bullet_list', 'numbered_list']:
                    self.logger.info(f"Processing {content['type']} with {len(content['items'])} items")

                    # For bullet lists, we need to transform them to Google Slides format
                    for i, item in enumerate(content['items']):
                        # Handle both string items and dict items
                        if isinstance(item, dict):
                            item_text = item.get('text', '')
                            item_level = item.get('level', 0)
                            item_formats = item.get('formats', [])
                            item_links = item.get('links', [])
                        else:
                            item_text = item
                            item_level = 0
                            item_formats = []
                            item_links = []

                        if not isinstance(item_text, str) or not item_text.strip():
                            continue

                        self.logger.info(f"Processing list item {i}: '{item_text[:30]}...', level={item_level}")

                        # For bullet lists, we need to add proper bullet formatting
                        text_requests.append({
                            'insertText': {
                                'objectId': body_id,
                                'insertionIndex': insertion_index,
                                'text': item_text + '\n'
                            }
                        })

                        self.logger.info(f"Added list item text at position {insertion_index}")

                        # Calculate ranges for applying bullet style
                        start_index = insertion_index
                        end_index = insertion_index + len(item_text)

                        # Apply formatting for bullet items (bold, italic)
                        for fmt_idx, format_info in enumerate(item_formats):
                            # Get the adjusted positions
                            start_pos = format_info.get('start_pos', 0)
                            end_pos = format_info.get('end_pos', 0)

                            self.logger.info(
                                f"Applying format {fmt_idx} to list item: type={format_info['type']}, start={start_pos}, end={end_pos}")

                            # Apply formatting
                            if format_info['type'] == 'bold':
                                text_requests.append({
                                    'updateTextStyle': {
                                        'objectId': body_id,
                                        'textRange': {
                                            'type': 'FIXED_RANGE',
                                            'startIndex': insertion_index + start_pos,
                                            'endIndex': insertion_index + end_pos
                                        },
                                        'style': {
                                            'bold': True
                                        },
                                        'fields': 'bold'
                                    }
                                })
                                self.logger.info(
                                    f"Applied bold formatting to list item from {insertion_index + start_pos} to {insertion_index + end_pos}")
                            elif format_info['type'] == 'italic':
                                text_requests.append({
                                    'updateTextStyle': {
                                        'objectId': body_id,
                                        'textRange': {
                                            'type': 'FIXED_RANGE',
                                            'startIndex': insertion_index + start_pos,
                                            'endIndex': insertion_index + end_pos
                                        },
                                        'style': {
                                            'italic': True
                                        },
                                        'fields': 'italic'
                                    }
                                })
                                self.logger.info(
                                    f"Applied italic formatting to list item from {insertion_index + start_pos} to {insertion_index + end_pos}")

                        # Apply links in bullet items
                        for link_idx, link_info in enumerate(item_links):
                            # Get the adjusted positions
                            start_pos = link_info.get('start_pos', 0)
                            end_pos = link_info.get('end_pos', 0)

                            self.logger.info(
                                f"Applying link {link_idx} to list item: url={link_info['url']}, start={start_pos}, end={end_pos}")

                            # Create hyperlink in the slide
                            text_requests.append({
                                'updateTextStyle': {
                                    'objectId': body_id,
                                    'textRange': {
                                        'type': 'FIXED_RANGE',
                                        'startIndex': insertion_index + start_pos,
                                        'endIndex': insertion_index + end_pos
                                    },
                                    'style': {
                                        'link': {
                                            'url': link_info['url']
                                        },
                                        'foregroundColor': {
                                            'opaqueColor': {
                                                'rgbColor': {
                                                    'red': 0.0,
                                                    'green': 0.0,
                                                    'blue': 0.8
                                                }
                                            }
                                        },
                                        'underline': True
                                    },
                                    'fields': 'link,foregroundColor,underline'
                                }
                            })
                            self.logger.info(
                                f"Applied link to list item from {insertion_index + start_pos} to {insertion_index + end_pos}")

                        # Update insertion index after adding the text
                        insertion_index += len(item_text) + 1
                        self.logger.info(f"Advanced insertion index to {insertion_index}")

                        # Apply bullet formatting for all list types
                        # We'll use standard bullet formatting for all items
                        # and prefix numbered items with "1. ", "2. ", etc. manually
                        bullet_request = {
                            'createParagraphBullets': {
                                'objectId': body_id,
                                'textRange': {
                                    'type': 'FIXED_RANGE',
                                    'startIndex': start_index,
                                    'endIndex': end_index
                                },
                                'bulletPreset': 'BULLET_DISC_CIRCLE_SQUARE'
                            }
                        }

                        # Add nesting level
                        if item_level > 0:
                            bullet_request['createParagraphBullets']['nestingLevel'] = item_level
                            self.logger.info(f"Set nesting level {item_level} for bullet")

                        text_requests.append(bullet_request)
                        self.logger.info(f"Added bullet formatting request for range {start_index}-{end_index}")

                    # Add extra newline after lists
                    text_requests.append({
                        'insertText': {
                            'objectId': body_id,
                            'insertionIndex': insertion_index,
                            'text': '\n'
                        }
                    })
                    insertion_index += 1
                    self.logger.info(f"Added extra newline after list, insertion index now at {insertion_index}")

                elif content['type'] == 'mixed_list':
                    self.logger.info(f"Processing mixed list with {len(content['items'])} items")

                    # Handle mixed lists (combination of bullet and numbered items)
                    for i, item in enumerate(content['items']):
                        if not item.get('text', '').strip():
                            continue

                        # Get the item's nesting level
                        item_level = item.get('level', 0)
                        item_formats = item.get('formats', [])
                        item_links = item.get('links', [])

                        self.logger.info(f"Processing mixed list item {i}: '{item['text'][:30]}...', level={item_level}")

                        # Insert the text for this item
                        text_requests.append({
                            'insertText': {
                                'objectId': body_id,
                                'insertionIndex': insertion_index,
                                'text': item['text'] + '\n'
                            }
                        })

                        self.logger.info(f"Added mixed list item text at position {insertion_index}")

                        # Calculate ranges for applying bullet style
                        start_index = insertion_index
                        end_index = insertion_index + len(item['text'])

                        # Apply formatting for mixed list items (bold, italic)
                        for fmt_idx, format_info in enumerate(item_formats):
                            # Get the adjusted positions
                            start_pos = format_info.get('start_pos', 0)
                            end_pos = format_info.get('end_pos', 0)

                            self.logger.info(
                                f"Applying format {fmt_idx} to mixed list item: type={format_info['type']}, start={start_pos}, end={end_pos}")

                            # Apply formatting
                            if format_info['type'] == 'bold':
                                text_requests.append({
                                    'updateTextStyle': {
                                        'objectId': body_id,
                                        'textRange': {
                                            'type': 'FIXED_RANGE',
                                            'startIndex': insertion_index + start_pos,
                                            'endIndex': insertion_index + end_pos
                                        },
                                        'style': {
                                            'bold': True
                                        },
                                        'fields': 'bold'
                                    }
                                })
                                self.logger.info(
                                    f"Applied bold formatting to mixed list item from {insertion_index + start_pos} to {insertion_index + end_pos}")
                            elif format_info['type'] == 'italic':
                                text_requests.append({
                                    'updateTextStyle': {
                                        'objectId': body_id,
                                        'textRange': {
                                            'type': 'FIXED_RANGE',
                                            'startIndex': insertion_index + start_pos,
                                            'endIndex': insertion_index + end_pos
                                        },
                                        'style': {
                                            'italic': True
                                        },
                                        'fields': 'italic'
                                    }
                                })
                                self.logger.info(
                                    f"Applied italic formatting to mixed list item from {insertion_index + start_pos} to {insertion_index + end_pos}")

                        # Apply links in mixed list items
                        for link_idx, link_info in enumerate(item_links):
                            # Get the adjusted positions
                            start_pos = link_info.get('start_pos', 0)
                            end_pos = link_info.get('end_pos', 0)

                            self.logger.info(
                                f"Applying link {link_idx} to mixed list item: url={link_info['url']}, start={start_pos}, end={end_pos}")

                            # Create hyperlink in the slide
                            text_requests.append({
                                'updateTextStyle': {
                                    'objectId': body_id,
                                    'textRange': {
                                        'type': 'FIXED_RANGE',
                                        'startIndex': insertion_index + start_pos,
                                        'endIndex': insertion_index + end_pos
                                    },
                                    'style': {
                                        'link': {
                                            'url': link_info['url']
                                        },
                                        'foregroundColor': {
                                            'opaqueColor': {
                                                'rgbColor': {
                                                    'red': 0.0,
                                                    'green': 0.0,
                                                    'blue': 0.8
                                                }
                                            }
                                        },
                                        'underline': True
                                    },
                                    'fields': 'link,foregroundColor,underline'
                                }
                            })
                            self.logger.info(
                                f"Applied link to mixed list item from {insertion_index + start_pos} to {insertion_index + end_pos}")

                        # Update insertion index after adding the text
                        insertion_index += len(item['text']) + 1
                        self.logger.info(f"Advanced insertion index to {insertion_index}")

                        # Apply bullet formatting for all list types
                        # We'll use standard bullet formatting for all items
                        bullet_request = {
                            'createParagraphBullets': {
                                'objectId': body_id,
                                'textRange': {
                                    'type': 'FIXED_RANGE',
                                    'startIndex': start_index,
                                    'endIndex': end_index
                                },
                                'bulletPreset': 'BULLET_DISC_CIRCLE_SQUARE'
                            }
                        }

                        # Add nesting level if specified
                        if item_level > 0:
                            bullet_request['createParagraphBullets']['nestingLevel'] = item_level
                            self.logger.info(f"Set nesting level {item_level} for mixed list bullet")

                        text_requests.append(bullet_request)
                        self.logger.info(
                            f"Added bullet formatting request for mixed list item range {start_index}-{end_index}")

                    # Add extra newline after mixed lists
                    text_requests.append({
                        'insertText': {
                            'objectId': body_id,
                            'insertionIndex': insertion_index,
                            'text': '\n'
                        }
                    })
                    insertion_index += 1
                    self.logger.info(f"Added extra newline after mixed list, insertion index now at {insertion_index}")

            # Apply gray color and size 16pt to the body text
            text_requests.append({
                'updateTextStyle': {
                    'objectId': body_id,
                    'textRange': {
                        'type': 'ALL'
                    },
                    'style': {
                        'fontSize': {
                            'magnitude': 16,
                            'unit': 'PT'
                        },
                        'foregroundColor': {
                            'opaqueColor': {
                                'rgbColor': {
                                    'red': 0.5,
                                    'green': 0.5,
                                    'blue': 0.5
                                }
                            }
                        }
                    },
                    'fields': 'fontSize,foregroundColor'
                }
            })
            self.logger.info(f"Applied global text styling")

        # Execute the text insertion
        if text_requests:
            self.logger.info(f"Executing {len(text_requests)} text requests")
            self._service.presentations().batchUpdate(
                presentationId=presentation_id,
                body={'requests': text_requests}
            ).execute()
            self.logger.info(f"Text requests executed successfully")
        else:
            self.logger.info(f"No text requests to execute")