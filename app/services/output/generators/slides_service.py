import re
from typing import List, Dict
from googleapiclient.discovery import build
from google.oauth2 import service_account
from app.services.output.document_service import DocumentService

class SlidesService(DocumentService):
    """Service for creating and managing Google Slides presentations."""
    
    def __init__(self, credentials_path: str = None):
        """
        Initialize the Slides service.
        
        Args:
            credentials_path: Path to the Google service account credentials JSON file.
                            If None, will use the path from ChatConfig.
        """
        super().__init__(credentials_path)
        self._service = self._get_slides_service()
        
    def _get_slides_service(self):
        """Get an authenticated Google Slides service."""
        creds = service_account.Credentials.from_service_account_file(
            self.credentials_path,
            scopes=['https://www.googleapis.com/auth/presentations', 
                    'https://www.googleapis.com/auth/drive']
        )
        return build('slides', 'v1', credentials=creds)
    
    async def create_presentation_from_csv(self, csv_path: str, title: str = "Q&A Presentation", author_name: str = None, owner_email: str = None) -> str:
        """
        Create a Google Slides presentation from a CSV file containing questions and format templates.
        Answers are generated using RAG with the specified format templates.
        
        Args:
            csv_path: Path to the CSV file containing questions and format templates
            title: Title for the presentation
            author_name: Name of the author to include on the title slide
            owner_email: Email address to share the presentation with
            
        Returns:
            The ID of the created presentation
        """
        import time
        
        # Read the CSV file
        question_format_pairs = self.read_csv(csv_path)
        
        # Create a new presentation with retry logic
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                presentation = self._service.presentations().create(
                    body={'title': title}
                ).execute()
                presentation_id = presentation.get('presentationId')
                self.logger.info(f"Created presentation with ID: {presentation_id}")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Error creating presentation (attempt {attempt+1}/{max_retries}): {str(e)}")
                    time.sleep(retry_delay)
                    # Increase delay for next attempt (exponential backoff)
                    retry_delay *= 2
                else:
                    self.logger.error(f"Failed to create presentation after {max_retries} attempts: {str(e)}")
                    raise
        
        # Create a title slide with retry logic
        for attempt in range(max_retries):
            try:
                self._create_title_slide(presentation_id, title, author_name)
                self.logger.info("Created title slide successfully")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Error creating title slide (attempt {attempt+1}/{max_retries}): {str(e)}")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    self.logger.error(f"Failed to create title slide after {max_retries} attempts: {str(e)}")
                    # Continue without title slide rather than failing the whole process
        
        # Track the current slide number
        current_slide_number = 1  # Start after title slide
        
        # Create slides for each question
        for i, (question, format_template) in enumerate(question_format_pairs, start=1):
            try:
                self.logger.info(f"Generating formatted RAG answer for question: {question}")
                self.logger.info(f"Using format template: {format_template}")
                
                # Get answer from RAG with format template
                formatted_answer = await self.get_formatted_content(question, format_template, "slides")
                
                # Check if the answer contains slide delimiters
                if "===SLIDE" in formatted_answer:
                    # Process multi-slide content
                    slide_contents = self._parse_slides_content(formatted_answer)
                    
                    # Create each slide
                    for slide_content in slide_contents:
                        slide_title = slide_content.get('title', f"Question {i}")
                        slide_body = slide_content.get('content', "")
                        
                        # Add retry logic for slide creation
                        for attempt in range(max_retries):
                            try:
                                self._create_qa_slide(
                                    presentation_id=presentation_id,
                                    question=slide_title,  # Use the slide title from formatted content
                                    answer=slide_body,     # Use the slide body from formatted content
                                    slide_number=current_slide_number
                                )
                                current_slide_number += 1
                                break
                            except Exception as e:
                                if attempt < max_retries - 1:
                                    self.logger.warning(f"Error creating slide {current_slide_number} (attempt {attempt+1}/{max_retries}): {str(e)}")
                                    time.sleep(retry_delay)
                                    retry_delay *= 2
                                else:
                                    self.logger.error(f"Failed to create slide {current_slide_number} after {max_retries} attempts: {str(e)}")
                                    raise  # Let the outer exception handler catch this
                else:
                    # Create a single slide with the formatted answer
                    for attempt in range(max_retries):
                        try:
                            self._create_qa_slide(
                                presentation_id=presentation_id,
                                question=question,
                                answer=formatted_answer,
                                slide_number=current_slide_number
                            )
                            current_slide_number += 1
                            break
                        except Exception as e:
                            if attempt < max_retries - 1:
                                self.logger.warning(f"Error creating slide {current_slide_number} (attempt {attempt+1}/{max_retries}): {str(e)}")
                                time.sleep(retry_delay)
                                retry_delay *= 2
                            else:
                                self.logger.error(f"Failed to create slide {current_slide_number} after {max_retries} attempts: {str(e)}")
                                raise  # Let the outer exception handler catch this
                    
            except Exception as e:
                self.logger.error(f"Error processing question {i}: {str(e)}")
                # Create a slide with error message
                try:
                    self._create_qa_slide(
                        presentation_id=presentation_id,
                        question=question,
                        answer=f"Error generating content: {str(e)}",
                        slide_number=current_slide_number
                    )
                    current_slide_number += 1
                except Exception as inner_e:
                    self.logger.error(f"Error creating error slide for question {i}: {str(inner_e)}")
                    # Continue without the error slide
        
        # Share the presentation if an email was provided
        if owner_email:
            max_share_retries = 3
            share_retry_delay = 2  # seconds
            
            for attempt in range(max_share_retries):
                try:
                    self.share_document(presentation_id, owner_email, "presentation")
                    self.logger.info(f"Successfully shared presentation with {owner_email}")
                    break
                except Exception as e:
                    if attempt < max_share_retries - 1:
                        self.logger.warning(f"Error sharing presentation (attempt {attempt+1}/{max_share_retries}): {str(e)}")
                        time.sleep(share_retry_delay)
                        share_retry_delay *= 2
                    else:
                        self.logger.error(f"Failed to share presentation after {max_share_retries} attempts: {str(e)}")
                        # Continue without sharing rather than failing the process
            
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
    
    def _parse_slides_content(self, text: str) -> List[Dict]:
        """
        Parse multi-slide content with slide delimiters.
        
        Expected format:
        ===SLIDE 1===
        Title: Slide Title
        Slide content here
        
        ===SLIDE 2===
        Title: Another Title
        More content
        
        Args:
            text: Text with slide delimiters
            
        Returns:
            List of dictionaries with slide title and content
        """
        slides = []
        
        # Split by slide delimiter
        if "===SLIDE" in text:
            parts = text.split("===SLIDE")
            
            # Process each part (skip the first part if it's empty)
            for part in parts:
                if not part.strip():
                    continue
                    
                # Split the part into lines
                lines = part.strip().split('\n')
                
                # Skip any slide number in the first line
                start_index = 0
                if lines[0].strip().endswith("==="):
                    start_index = 1
                
                # Find the title line
                title = ""
                title_index = -1
                
                for i, line in enumerate(lines[start_index:], start=start_index):
                    if line.startswith("Title:"):
                        title = line[6:].strip()
                        title_index = i
                        break
                
                # Get the content (everything after the title)
                if title_index != -1 and title_index < len(lines) - 1:
                    content_lines = lines[title_index + 1:]
                    # Check if first content line starts with "Body:" and remove it
                    if content_lines and content_lines[0].strip().startswith("Body:"):
                        content_lines[0] = content_lines[0].strip()[5:].strip()
                    content = '\n'.join(content_lines)
                else:
                    # If no title found or no content after title, use a default
                    if title == "":
                        title = "Slide"
                    content = '\n'.join(lines[start_index:])
                
                # Remove "Body:" prefix from content if present
                content = content.strip()
                if content.startswith("Body:"):
                    content = content[5:].strip()
                
                slides.append({
                    'title': title,
                    'content': content
                })
        else:
            # Handle the case where there's no slide delimiter but still formatted text
            lines = text.strip().split('\n')
            
            # Check if the first line is a title
            if lines and lines[0].startswith("Title:"):
                title = lines[0][6:].strip()
                content_lines = lines[1:]
                # Check if first content line starts with "Body:" and remove it
                if content_lines and content_lines[0].strip().startswith("Body:"):
                    content_lines[0] = content_lines[0].strip()[5:].strip()
                content = '\n'.join(content_lines).strip()
            else:
                title = "Slide"
                content = text.strip()
                
            # Remove "Body:" prefix from content if present
            if content.startswith("Body:"):
                content = content[5:].strip()
                
            slides.append({
                'title': title,
                'content': content
            })
        
        return slides
        
    def _parse_markdown(self, text: str) -> List[Dict]:
        """
        Parse Markdown text into structured content for slides.

        Supported markdown features:
        - Bullet lists (starting with '- ' or '* ')
        - Numbered lists (starting with '1. ', '2. ', etc.)
        - Bold text using **bold** syntax
        - Italic text using *italic* syntax
        - Links using [text](url) syntax
        - Headings using # syntax

        Args:
            text: Markdown formatted text

        Returns:
            List of dictionaries with content and formatting info
        """
        self.logger.info(f"Starting to parse markdown text: {text[:50]}...")

        result = []

        # Handle newlines consistently
        text = text.replace('\r\n', '\n')

        # First identify and extract headings from other content
        lines = text.split('\n')
        current_section = []
        processed_sections = []
        
        for line in lines:
            # If line is a heading, start a new section
            if line.strip().startswith('#'):
                # If we have content in the current section, save it
                if current_section:
                    processed_sections.append('\n'.join(current_section))
                    current_section = []
                
                # Add the heading as its own section
                processed_sections.append(line)
                current_section = []
            else:
                # Add line to current section
                current_section.append(line)
        
        # Add the last section if it has content
        if current_section:
            processed_sections.append('\n'.join(current_section))
        
        self.logger.info(f"Split content into {len(processed_sections)} sections (headings + content blocks)")
        
        # Process each section to handle different content types
        for section_idx, section in enumerate(processed_sections):
            # If this is a heading, add it directly as a paragraph
            if section.strip().startswith('#'):
                self.logger.info(f"HEADING SECTION {section_idx}: '{section.strip()}'")
                processed_content = self._process_inline_formatting_v2(section)
                
                result.append({
                    'type': 'paragraph',
                    'text': section.strip(),  # Keep the # for detection during rendering
                    'formats': processed_content.get('formats', []),
                    'links': processed_content.get('links', [])
                })
                continue
                
            # For non-heading sections, split into paragraphs and process normally
            section_paragraphs = re.split(r'\n\s*\n', section)
            self.logger.info(f"CONTENT SECTION {section_idx}: Split into {len(section_paragraphs)} paragraphs")
            
            # Process each paragraph in this section
            for paragraph_idx, paragraph in enumerate(section_paragraphs):
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                    
                self.logger.info(f"Processing paragraph {paragraph_idx}: {paragraph[:50]}...")
    
                # Detect paragraph type based on its first line
                lines = paragraph.split('\n')
                first_line = lines[0].strip() if lines else ""

                # Check if it's a heading (starts with #)
                if first_line.startswith('#'):
                    self.logger.info(f"Paragraph {paragraph_idx} is a heading")
                    # Process as a regular paragraph - we'll detect the heading when rendering
                    processed_paragraph = self._process_inline_formatting_v2(paragraph)
                    
                    result.append({
                        'type': 'paragraph',  # Still treat as paragraph but we'll process it as heading later
                        'text': paragraph,    # Keep the # markers for later detection
                        'formats': processed_paragraph.get('formats', []),
                        'links': processed_paragraph.get('links', [])
                    })
                # Check if it's a bullet point list (starts with - or *)
                elif first_line.startswith('- ') or first_line.startswith('* '):
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

        # Log the raw answer to see what's coming in
        self.logger.info(f"RAW ANSWER: '{answer[:200]}...'")
        
        # Check for headings in raw text
        for line in answer.split('\n'):
            if line.strip().startswith('#'):
                self.logger.info(f"FOUND HEADING in raw text: '{line.strip()}'")
        
        # Parse the answer as markdown
        parsed_answer = self._parse_markdown(answer)
        self.logger.info(f"Parsed answer contains {len(parsed_answer)} blocks")
        
        # Log each parsed block for debugging
        for idx, block in enumerate(parsed_answer):
            block_type = block.get('type', 'unknown')
            self.logger.info(f"PARSED BLOCK {idx}: Type={block_type}")
            if 'text' in block:
                self.logger.info(f"PARSED BLOCK {idx} TEXT: '{block['text'][:50]}...'")
                # Check specifically for headings in parsed content
                if block['text'].strip().startswith('#'):
                    self.logger.info(f"HEADING FOUND IN PARSED BLOCK {idx}: '{block['text'][:50]}...'")
                    
            if block_type == 'bullet_list' and 'items' in block:
                self.logger.info(f"PARSED BLOCK {idx}: Contains {len(block['items'])} bullet items")
                # Check first few bullet items
                for i, item in enumerate(block['items'][:3]):
                    if isinstance(item, dict) and 'text' in item:
                        self.logger.info(f"  BULLET ITEM {i}: '{item['text'][:30]}...'")
                        if item['text'].strip().startswith('#'):
                            self.logger.info(f"  HEADING FOUND IN BULLET ITEM {i}: '{item['text'][:30]}...'")
                    elif isinstance(item, str):
                        self.logger.info(f"  BULLET ITEM {i}: '{item[:30]}...'")
                        if item.strip().startswith('#'):
                            self.logger.info(f"  HEADING FOUND IN BULLET ITEM {i}: '{item[:30]}...'")
                    else:
                        self.logger.info(f"  BULLET ITEM {i}: Unknown format")

        # Generate a unique slide ID using both number and timestamp
        import time
        unique_slide_id = f'qa_slide_{slide_number}_{int(time.time() * 1000)}'
        self.logger.info(f"Using unique slide ID: {unique_slide_id}")

        # Step 1: Create the slide
        create_slide_request = [{
            'createSlide': {
                'objectId': unique_slide_id,
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
            if s.get('objectId') == unique_slide_id:
                current_slide = s
                break

        # If we couldn't find the slide, use a different approach
        if not current_slide:
            self.logger.info(f"Could not find slide {slide_number}, using fallback approach")
            # Alternative approach: Create text boxes directly with unique IDs
            timestamp_suffix = int(time.time() * 1000)
            question_box_id = f'question_box_{slide_number}_{timestamp_suffix}'
            answer_box_id = f'answer_box_{slide_number}_{timestamp_suffix}'

            text_boxes_request = [
                # Create a text box for the question at the top
                {
                    'createShape': {
                        'objectId': question_box_id,
                        'shapeType': 'TEXT_BOX',
                        'elementProperties': {
                            'pageObjectId': unique_slide_id,
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
                        'text': question
                    }
                },
                # Create a text box for the answer below
                {
                    'createShape': {
                        'objectId': answer_box_id,
                        'shapeType': 'TEXT_BOX',
                        'elementProperties': {
                            'pageObjectId': unique_slide_id,
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
                        'text': answer.replace("**", "").replace("*", "")  # Simple markdown stripping
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
                                'magnitude': 14,
                                'unit': 'PT'
                            },
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
                    'text': question
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
            # Start with empty body
            insertion_index = 0

            # Process the markdown content
            for block_idx, content in enumerate(parsed_answer):
                self.logger.info(f"Processing content block {block_idx}, type={content['type']}")

                if content['type'] == 'paragraph':
                    # Determine if this paragraph should be treated as a heading
                    is_heading = False
                    heading_level = 0
                    text_content = content['text']
                    
                    # Add more detailed logging
                    self.logger.info(f"HEADING DETECTION - Examining text: '{text_content[:40]}...'")
                    
                    # PART 1: Check for explicit Markdown heading patterns (# Heading or ## Subheading)
                    if text_content.lstrip().startswith('#'):
                        # Get the stripped version for more accurate detection
                        stripped_content = text_content.lstrip()
                        self.logger.info(f"POTENTIAL HEADING - Stripped text starts with #: '{stripped_content[:40]}...'")
                        
                        # Count the number of # symbols at the start
                        heading_chars = 0
                        for char in stripped_content:
                            if char == '#':
                                heading_chars += 1
                            else:
                                break
                                
                        if heading_chars > 0 and heading_chars <= 3:  # Support up to ### (h3)
                            # Extract the heading text without the # symbols
                            heading_text = stripped_content[heading_chars:].strip()
                            text_content = heading_text  # Remove the # symbols from displayed text
                            is_heading = True
                            heading_level = heading_chars
                            self.logger.info(f"CONFIRMED HEADING - Level {heading_level}: '{heading_text[:40]}...'")
                            self.logger.info(f"HEADING DISPLAY - Will display as: '{text_content[:40]}...'")
                        else:
                            self.logger.info(f"NOT A MARKDOWN HEADING - Invalid # symbols: {heading_chars}")
                    
                    # PART 2: Infer heading status from structure and formatting
                    # If the text isn't already identified as a heading by explicit Markdown
                    elif (
                        # 1. Short lines that are likely headers/section names
                        (len(text_content.strip().split()) <= 4) or
                        
                        # 2. Lines ending with colon (often introducing lists/sections)
                        text_content.strip().endswith(':') or
                        
                        # 3. Lines without terminal punctuation
                        (not any(text_content.strip().endswith(p) for p in ['.', '!', '?', ',', ';'])) or
                        
                        # 4. Short self-contained statements
                        (len(text_content.strip()) < 40 and ',' not in text_content)
                    ):
                        is_heading = True
                        heading_level = 1  # Treat as h1 by default 
                        self.logger.info(f"INFERRED HEADING - Based on structure: '{text_content[:40]}...'")
                    else:
                        self.logger.info(f"NOT A HEADING - No heading patterns detected")
                    
                    # Add the paragraph text
                    text_requests.append({
                        'insertText': {
                            'objectId': body_id,
                            'insertionIndex': insertion_index,
                            'text': text_content + '\n'
                        }
                    })

                    self.logger.info(f"TEXT INSERTION - Added text: '{text_content[:50]}...'")
                    self.logger.info(f"INSERTION INDEX - Now at: {insertion_index}")
                    
                    if is_heading:
                        self.logger.info(f"HEADING STYLE - Will apply heading level {heading_level} formatting next")
                    else:
                        self.logger.info(f"PARAGRAPH STYLE - Will apply normal paragraph formatting")

                    # Apply special formatting for headings
                    if is_heading:
                        # Make headings bold always
                        text_requests.append({
                            'updateTextStyle': {
                                'objectId': body_id,
                                'textRange': {
                                    'type': 'FIXED_RANGE',
                                    'startIndex': insertion_index,
                                    'endIndex': insertion_index + len(text_content)
                                },
                                'style': {
                                    'bold': True,
                                    'fontSize': {
                                        # Size based on heading level
                                        'magnitude': 18 - (heading_level * 1),  # H1=18pt, H2=17pt, H3=16pt
                                        'unit': 'PT'
                                    }
                                },
                                'fields': 'bold,fontSize'
                            }
                        })
                        
                        # Add some space after headings
                        text_requests.append({
                            'updateParagraphStyle': {
                                'objectId': body_id,
                                'textRange': {
                                    'type': 'FIXED_RANGE',
                                    'startIndex': insertion_index,
                                    'endIndex': insertion_index + len(text_content)
                                },
                                'style': {
                                    'spaceBelow': {
                                        'magnitude': 10,
                                        'unit': 'PT'
                                    }
                                },
                                'fields': 'spaceBelow'
                            }
                        })
                        
                        self.logger.info(f"Applied heading style to: {text_content}")
                    
                    # Apply formatting for formatted segments (bold, italic)
                    current_text_length = len(text_content) + 1  # +1 for the newline

                    if 'formats' in content and content['formats']:
                        for fmt_idx, format_info in enumerate(content['formats']):
                            # Get the adjusted positions
                            start_pos = format_info.get('start_pos', 0)
                            end_pos = format_info.get('end_pos', 0)

                            self.logger.info(
                                f"Applying format {fmt_idx}: type={format_info['type']}, start={start_pos}, end={end_pos}")

                            # Get current text length - we'll use this to validate ranges
                            # This is needed because format positions might get misaligned due to content changes
                            current_text_length = insertion_index + len(content['text']) + 1  # +1 for the newline
                            
                            # Apply formatting with safety checks
                            start_idx = min(insertion_index + start_pos, current_text_length - 1)
                            end_idx = min(insertion_index + end_pos, current_text_length)
                            
                            # Only apply formatting if range is valid
                            if start_idx < end_idx:
                                if format_info['type'] == 'bold':
                                    text_requests.append({
                                        'updateTextStyle': {
                                            'objectId': body_id,
                                            'textRange': {
                                                'type': 'FIXED_RANGE',
                                                'startIndex': start_idx,
                                                'endIndex': end_idx
                                            },
                                            'style': {
                                                'bold': True
                                            },
                                            'fields': 'bold'
                                        }
                                    })
                                    self.logger.info(
                                        f"Applied bold formatting from {start_idx} to {end_idx}")
                                elif format_info['type'] == 'italic':
                                    text_requests.append({
                                        'updateTextStyle': {
                                            'objectId': body_id,
                                            'textRange': {
                                                'type': 'FIXED_RANGE',
                                                'startIndex': start_idx,
                                                'endIndex': end_idx
                                            },
                                            'style': {
                                                'italic': True
                                            },
                                            'fields': 'italic'
                                        }
                                    })
                                    self.logger.info(
                                        f"Applied italic formatting from {start_idx} to {end_idx}")
                            else:
                                self.logger.warning(
                                    f"Skipping invalid {format_info['type']} formatting range: {start_idx}-{end_idx}")

                    # Apply links
                    if 'links' in content and content['links']:
                        for link_idx, link_info in enumerate(content['links']):
                            # Get the adjusted positions
                            start_pos = link_info.get('start_pos', 0)
                            end_pos = link_info.get('end_pos', 0)

                            self.logger.info(
                                f"Applying link {link_idx}: url={link_info['url']}, start={start_pos}, end={end_pos}")

                            # Get current text length for validation
                            current_text_length = insertion_index + len(content['text']) + 1  # +1 for the newline
                            
                            # Apply safety checks to link positions
                            start_idx = min(insertion_index + start_pos, current_text_length - 1)
                            end_idx = min(insertion_index + end_pos, current_text_length)
                            
                            # Only apply link if range is valid
                            if start_idx < end_idx:
                                # Create hyperlink in the slide
                                text_requests.append({
                                    'updateTextStyle': {
                                        'objectId': body_id,
                                        'textRange': {
                                            'type': 'FIXED_RANGE',
                                            'startIndex': start_idx,
                                            'endIndex': end_idx
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
                                self.logger.info(f"Applied link from {start_idx} to {end_idx}")
                            else:
                                self.logger.warning(f"Skipping invalid link range: {start_idx}-{end_idx}")

                    # Advance insertion index
                    insertion_index += len(content['text']) + 1
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

                            # Get list item text length for validation
                            item_text_length = insertion_index + len(item_text) + 1  # +1 for newline
                            
                            # Apply safety checks to formatting positions
                            start_idx = min(insertion_index + start_pos, item_text_length - 1)
                            end_idx = min(insertion_index + end_pos, item_text_length)
                            
                            # Only apply formatting if range is valid
                            if start_idx < end_idx:
                                # Apply formatting
                                if format_info['type'] == 'bold':
                                    text_requests.append({
                                        'updateTextStyle': {
                                            'objectId': body_id,
                                            'textRange': {
                                                'type': 'FIXED_RANGE',
                                                'startIndex': start_idx,
                                                'endIndex': end_idx
                                            },
                                            'style': {
                                                'bold': True
                                            },
                                            'fields': 'bold'
                                        }
                                    })
                                    self.logger.info(
                                        f"Applied bold formatting to list item from {start_idx} to {end_idx}")
                                elif format_info['type'] == 'italic':
                                    text_requests.append({
                                        'updateTextStyle': {
                                            'objectId': body_id,
                                            'textRange': {
                                                'type': 'FIXED_RANGE',
                                                'startIndex': start_idx,
                                                'endIndex': end_idx
                                            },
                                            'style': {
                                                'italic': True
                                            },
                                            'fields': 'italic'
                                        }
                                    })
                                    self.logger.info(
                                        f"Applied italic formatting to list item from {start_idx} to {end_idx}")
                            else:
                                self.logger.warning(
                                    f"Skipping invalid {format_info['type']} formatting for list item: {start_idx}-{end_idx}")

                        # Apply links in bullet items
                        for link_idx, link_info in enumerate(item_links):
                            # Get the adjusted positions
                            start_pos = link_info.get('start_pos', 0)
                            end_pos = link_info.get('end_pos', 0)

                            self.logger.info(
                                f"Applying link {link_idx} to list item: url={link_info['url']}, start={start_pos}, end={end_pos}")

                            # Get list item text length for link validation
                            item_text_length = insertion_index + len(item_text) + 1  # +1 for newline
                            
                            # Apply safety checks to link positions
                            start_idx = min(insertion_index + start_pos, item_text_length - 1)
                            end_idx = min(insertion_index + end_pos, item_text_length)
                            
                            # Only apply link if range is valid
                            if start_idx < end_idx:
                                # Create hyperlink in the slide
                                text_requests.append({
                                    'updateTextStyle': {
                                        'objectId': body_id,
                                        'textRange': {
                                            'type': 'FIXED_RANGE',
                                            'startIndex': start_idx,
                                            'endIndex': end_idx
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
                                self.logger.info(f"Applied link to list item from {start_idx} to {end_idx}")
                            else:
                                self.logger.warning(f"Skipping invalid link range for list item: {start_idx}-{end_idx}")

                        # Update insertion index after adding the text
                        insertion_index += len(item_text) + 1
                        self.logger.info(f"Advanced insertion index to {insertion_index}")

                        # Apply bullet formatting for all list types
                        # We'll use standard bullet formatting for all items
                        # and prefix numbered items with "1. ", "2. ", etc. manually
                        
                        # Determine if this item should be formatted as a heading or as a bullet point
                        is_heading = False
                        original_item_text = item_text
                        
                        # Add more verbose logging
                        self.logger.info(f"HEADING CHECK - Item text: '{item_text[:40]}...'")
                        
                        # Check if the text is a string
                        if isinstance(item_text, str):
                            stripped_text = item_text.lstrip()
                            
                            # First check: Explicit markdown heading pattern with # symbols
                            if stripped_text.startswith('#') or '\n#' in item_text or '\r\n#' in item_text:
                                is_heading = True
                                self.logger.info(f"EXPLICIT HEADING FOUND - Text contains # pattern: '{stripped_text[:40]}...'")
                                
                                # For text that starts with #, remove the # symbols for display
                                if stripped_text.startswith('#'):
                                    # Count the heading level and remove the # symbols
                                    heading_chars = 0
                                    for char in stripped_text:
                                        if char == '#':
                                            heading_chars += 1
                                        else:
                                            break
                                    
                                    self.logger.info(f"HEADING LEVEL - Found {heading_chars} # symbols")
                                    
                                    if heading_chars > 0:
                                        # Remove the # symbols from the displayed text
                                        item_text = stripped_text[heading_chars:].strip()
                                        self.logger.info(f"HEADING CLEANED - Text after removing #: '{item_text[:40]}...'")
                            
                            # Second check: Infer heading status based on structural position and format
                            elif item_level == 0 and (
                                # 1. Top-level items that are short (likely section headers)
                                (len(item_text.strip().split()) <= 4) or
                                
                                # 2. Items ending with colon (often headers introducing a section)
                                item_text.strip().endswith(':') or
                                
                                # 3. Items without ending punctuation (likely headings not sentences)
                                (not any(item_text.strip().endswith(p) for p in ['.', '!', '?', ',', ';'])) or
                                
                                # 4. Items that are self-contained (no subordinate clauses/phrases)
                                (len(item_text.strip()) < 40 and ',' not in item_text)
                            ):
                                is_heading = True
                                self.logger.info(f"INFERRED HEADING - Based on structure and format: '{item_text[:40]}...'")
                            else:
                                self.logger.info(f"NOT A HEADING - Text does not match heading patterns")
                                
                            # Apply specific treatment for headings
                            if is_heading:
                                self.logger.info(f"HEADING PROCESSING - Will NOT apply bullet formatting to this heading")
                        else:
                            self.logger.info(f"NOT A HEADING - Item text is not a string type")
                        
                        if not is_heading:
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
                        else:
                            # For headings, don't create a bullet request
                            bullet_request = None

                        # Apply indentation using text styling instead of bullet nesting
                        if item_level > 0 and not is_heading:
                            # Calculate the indentation based on level (20pts per level)
                            indent_size = item_level * 20
                            
                            # Add indentation to the bullet through paragraph style
                            text_requests.append({
                                'updateParagraphStyle': {
                                    'objectId': body_id,
                                    'textRange': {
                                        'type': 'FIXED_RANGE',
                                        'startIndex': start_index,
                                        'endIndex': end_index
                                    },
                                    'style': {
                                        'indentStart': {
                                            'magnitude': indent_size,
                                            'unit': 'PT'
                                        }
                                    },
                                    'fields': 'indentStart'
                                }
                            })
                            
                            self.logger.info(f"Applied indentation of {indent_size}pt for level {item_level} bullet")

                        # Apply heading formatting if this is a heading
                        if is_heading:
                            # Make headings bold always
                            heading_level = 1  # Default to H1
                            
                            # Count # symbols to determine heading level
                            if isinstance(item_text, str):
                                heading_chars = 0
                                for char in item_text.lstrip():
                                    if char == '#':
                                        heading_chars += 1
                                    else:
                                        break
                                        
                                if heading_chars > 0 and heading_chars <= 3:
                                    heading_level = heading_chars
                            
                            # Apply heading style (bold + size based on level)
                            text_requests.append({
                                'updateTextStyle': {
                                    'objectId': body_id,
                                    'textRange': {
                                        'type': 'FIXED_RANGE',
                                        'startIndex': start_index,
                                        'endIndex': end_index
                                    },
                                    'style': {
                                        'bold': True,
                                        'fontSize': {
                                            'magnitude': 18 - (heading_level * 1),  # H1=18pt, H2=17pt, H3=16pt
                                            'unit': 'PT'
                                        }
                                    },
                                    'fields': 'bold,fontSize'
                                }
                            })
                            
                            self.logger.info(f"Applied heading style to item: {item_text[:30]}...")
                        
                        # Only add bullet request if it's not a heading and the request exists
                        if bullet_request is not None:
                            text_requests.append(bullet_request)
                            self.logger.info(f"BULLET ADDED - Added bullet formatting request for range {start_index}-{end_index}")
                        else:
                            if is_heading:
                                self.logger.info(f"BULLET SKIPPED - Not applying bullet format to heading: '{item_text[:40]}...'")
                            else:
                                self.logger.info(f"BULLET SKIPPED - Request was None (unexpected)")

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

                            # Get mixed list item text length for validation
                            item_text_length = insertion_index + len(item['text']) + 1  # +1 for newline
                            
                            # Apply safety checks to formatting positions
                            start_idx = min(insertion_index + start_pos, item_text_length - 1)
                            end_idx = min(insertion_index + end_pos, item_text_length)
                            
                            # Only apply formatting if range is valid
                            if start_idx < end_idx:
                                # Apply formatting
                                if format_info['type'] == 'bold':
                                    text_requests.append({
                                        'updateTextStyle': {
                                            'objectId': body_id,
                                            'textRange': {
                                                'type': 'FIXED_RANGE',
                                                'startIndex': start_idx,
                                                'endIndex': end_idx
                                            },
                                            'style': {
                                                'bold': True
                                            },
                                            'fields': 'bold'
                                        }
                                    })
                                    self.logger.info(
                                        f"Applied bold formatting to mixed list item from {start_idx} to {end_idx}")
                                elif format_info['type'] == 'italic':
                                    text_requests.append({
                                        'updateTextStyle': {
                                            'objectId': body_id,
                                            'textRange': {
                                                'type': 'FIXED_RANGE',
                                                'startIndex': start_idx,
                                                'endIndex': end_idx
                                            },
                                            'style': {
                                                'italic': True
                                            },
                                            'fields': 'italic'
                                        }
                                    })
                                    self.logger.info(
                                        f"Applied italic formatting to mixed list item from {start_idx} to {end_idx}")
                            else:
                                self.logger.warning(
                                    f"Skipping invalid {format_info['type']} formatting for mixed list item: {start_idx}-{end_idx}")

                        # Apply links in mixed list items
                        for link_idx, link_info in enumerate(item_links):
                            # Get the adjusted positions
                            start_pos = link_info.get('start_pos', 0)
                            end_pos = link_info.get('end_pos', 0)

                            self.logger.info(
                                f"Applying link {link_idx} to mixed list item: url={link_info['url']}, start={start_pos}, end={end_pos}")

                            # Get mixed list item text length for link validation
                            item_text_length = insertion_index + len(item['text']) + 1  # +1 for newline
                            
                            # Apply safety checks to link positions
                            start_idx = min(insertion_index + start_pos, item_text_length - 1)
                            end_idx = min(insertion_index + end_pos, item_text_length)
                            
                            # Only apply link if range is valid
                            if start_idx < end_idx:
                                # Create hyperlink in the slide
                                text_requests.append({
                                    'updateTextStyle': {
                                        'objectId': body_id,
                                        'textRange': {
                                            'type': 'FIXED_RANGE',
                                            'startIndex': start_idx,
                                            'endIndex': end_idx
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
                                self.logger.info(f"Applied link to mixed list item from {start_idx} to {end_idx}")
                            else:
                                self.logger.warning(f"Skipping invalid link range for mixed list item: {start_idx}-{end_idx}")

                        # Update insertion index after adding the text
                        insertion_index += len(item['text']) + 1
                        self.logger.info(f"Advanced insertion index to {insertion_index}")

                        # Apply bullet formatting for all list types
                        # We'll use standard bullet formatting for all items
                        
                        # Check if the item is a heading (starts with #)
                        is_heading = False
                        item_content = item.get('text', '') if isinstance(item, dict) else ''
                        original_item_content = item_content
                        
                        if item_content.lstrip().startswith('#'):
                            is_heading = True
                            
                            # Count the heading level and remove the # symbols from displayed text
                            heading_chars = 0
                            for char in item_content.lstrip():
                                if char == '#':
                                    heading_chars += 1
                                else:
                                    break
                            
                            if heading_chars > 0:
                                # Remove the # symbols from the displayed text
                                clean_text = item_content.lstrip()[heading_chars:].strip()
                                
                                # Update the item text to remove the # symbols
                                if isinstance(item, dict):
                                    item['text'] = clean_text
                                
                                # Also update our local variable
                                item_content = clean_text
                            
                            self.logger.info(f"Detected heading in mixed list - will not apply bullet formatting: {original_item_content[:30]}...")
                        
                        if not is_heading:
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
                        else:
                            # For headings, don't create a bullet request
                            bullet_request = None

                        # Apply indentation using text styling instead of bullet nesting
                        if item_level > 0 and not is_heading:
                            # Calculate the indentation based on level (20pts per level)
                            indent_size = item_level * 20
                            
                            # Add indentation to the bullet through paragraph style
                            text_requests.append({
                                'updateParagraphStyle': {
                                    'objectId': body_id,
                                    'textRange': {
                                        'type': 'FIXED_RANGE',
                                        'startIndex': start_index,
                                        'endIndex': end_index
                                    },
                                    'style': {
                                        'indentStart': {
                                            'magnitude': indent_size,
                                            'unit': 'PT'
                                        }
                                    },
                                    'fields': 'indentStart'
                                }
                            })
                            
                            self.logger.info(f"Applied indentation of {indent_size}pt for level {item_level} mixed list bullet")

                        # Apply heading formatting if this is a heading
                        if is_heading:
                            # Make headings bold always
                            heading_level = 1  # Default to H1
                            
                            # Count # symbols to determine heading level
                            heading_chars = 0
                            for char in item_content.lstrip():
                                if char == '#':
                                    heading_chars += 1
                                else:
                                    break
                                    
                            if heading_chars > 0 and heading_chars <= 3:
                                heading_level = heading_chars
                            
                            # Apply heading style (bold + size based on level)
                            text_requests.append({
                                'updateTextStyle': {
                                    'objectId': body_id,
                                    'textRange': {
                                        'type': 'FIXED_RANGE',
                                        'startIndex': start_index,
                                        'endIndex': end_index
                                    },
                                    'style': {
                                        'bold': True,
                                        'fontSize': {
                                            'magnitude': 18 - (heading_level * 1),  # H1=18pt, H2=17pt, H3=16pt
                                            'unit': 'PT'
                                        }
                                    },
                                    'fields': 'bold,fontSize'
                                }
                            })
                            
                            self.logger.info(f"Applied heading style to mixed list item: {item_content[:30]}...")
                        
                        # Only add bullet request if it's not a heading and the request exists
                        if bullet_request is not None:
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

            # Apply gray color and size 14pt to the body text
            text_requests.append({
                'updateTextStyle': {
                    'objectId': body_id,
                    'textRange': {
                        'type': 'ALL'
                    },
                    'style': {
                        'fontSize': {
                            'magnitude': 14,
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