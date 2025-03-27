import os
import json
from tqdm import tqdm
from markdownify import markdownify as md
from typing import Callable, Dict, List, Union, Optional, Any
from pathlib import Path
import logging
import io
from googleapiclient.http import MediaIoBaseDownload
import docx
import PyPDF2
import pptx
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

# Import from your existing project structure
import requests
from google.oauth2 import service_account
from googleapiclient.discovery import build

from app.config.chat_config import ChatConfig
from app.services.enhancement_service import enhancement_service

class CustomJsonLoader(BaseLoader):
    """Custom loader for JSON data"""

    def __init__(
            self, data: Union[Dict, List, Path], dataset_mapping_function: Callable[[Dict], Document]
    ):
        if isinstance(data, (dict, list)):
            self.data = data
        elif isinstance(data, (Path, str)):
            path = Path(data).resolve()
            with open(path, "r", encoding='utf-8') as f:
                self.data = json.load(f)
        self.dataset_mapping_function = dataset_mapping_function

    def load(self) -> List[Document]:
        """Load documents."""
        return list(map(self.dataset_mapping_function, self.data))

class GoogleDriveIndexer:
    """Class to handle the process of fetching Google Drive data and indexing to Pinecone"""

    def __init__(self, config: Optional[ChatConfig] = None):
        """Initialize the indexer with configuration"""
        self.config = config or ChatConfig()
        self.last_chunks = []  # Store chunks for reporting

        # Validate configuration
        missing_settings = self.config.validate_settings()
        if missing_settings:
            raise ValueError(f"Missing required settings: {', '.join(missing_settings)}")

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Create output directory if needed
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)

        # Initialize Google Drive API
        self._initialize_drive_api()

    def _initialize_drive_api(self):
        """Initialize the Google Drive API client"""
        try:
            # Load service account credentials from the specified file
            credentials = service_account.Credentials.from_service_account_file(
                self.config.GOOGLE_DRIVE_CREDENTIALS_FILE,
                scopes=['https://www.googleapis.com/auth/drive.readonly']
            )

            # Build the Drive API client
            self.drive_service = build('drive', 'v3', credentials=credentials)
            self.logger.info("Successfully initialized Google Drive API client")
        except Exception as e:
            self.logger.error(f"Failed to initialize Google Drive API client: {str(e)}")
            raise

    def list_folder_contents(self, folder_id=None):
        """List files and folders accessible to the service account"""
        self.logger.info(f"Listing files and folders accessible to service account")

        # If no specific folder is requested, get all accessible files
        if folder_id is None:
            query = "trashed = false"
        else:
            query = f"'{folder_id}' in parents and trashed = false"

        try:
            results = []
            page_token = None

            while True:
                response = self.drive_service.files().list(
                    q=query,
                    spaces='drive',
                    fields='nextPageToken, files(id, name, mimeType, webViewLink)',
                    pageToken=page_token,
                    pageSize=100
                ).execute()

                items = response.get('files', [])
                results.extend(items)

                page_token = response.get('nextPageToken')
                if not page_token:
                    break

            self.logger.info(f"Found {len(results)} items")
            return results
        except Exception as e:
            self.logger.error(f"Error listing files: {str(e)}")
            return []

    def get_supported_files(self, folder_id=None, recursive=True):
        """Get all supported files, optionally from a specific folder"""
        all_files = []
        items = self.list_folder_contents(folder_id)

        # Define supported MIME types
        supported_mime_types = [
            'application/vnd.google-apps.document',  # Google Docs
            'application/vnd.google-apps.spreadsheet',  # Google Sheets
            'application/vnd.google-apps.presentation',  # Google Slides
            'application/pdf',  # PDF files
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',  # DOCX files
            'application/vnd.openxmlformats-officedocument.presentationml.presentation',  # PPTX files
            'text/plain',  # Text files
            'text/markdown',  # Markdown files
            'text/html'  # HTML files
        ]

        for item in items:
            if item['mimeType'] in supported_mime_types:
                # It's a supported file
                all_files.append(item)
            elif item['mimeType'] == 'application/vnd.google-apps.folder' and recursive and folder_id is not None:
                # It's a folder, and we want to process recursively
                # Only process subfolders if we're starting from a specific folder
                folder_files = self.get_supported_files(item['id'], recursive=True)
                all_files.extend(folder_files)

        return all_files

    def download_and_extract_content(self, file_item):
        """Download and extract content from a Google Drive file"""
        file_id = file_item['id']
        mime_type = file_item['mimeType']
        file_name = file_item['name']

        self.logger.info(f"Processing file: {file_name} (MIME type: {mime_type})")

        try:
            # Handle Google Workspace files (need to export)
            if mime_type.startswith('application/vnd.google-apps'):
                if mime_type == 'application/vnd.google-apps.document':
                    # Export Google Docs as plain text
                    request = self.drive_service.files().export_media(fileId=file_id, mimeType='text/plain')
                    content = self._download_file(request)
                    return content.decode('utf-8')

                elif mime_type == 'application/vnd.google-apps.spreadsheet':
                    # Export Google Sheets as CSV
                    request = self.drive_service.files().export_media(fileId=file_id, mimeType='text/csv')
                    content = self._download_file(request)
                    return content.decode('utf-8')

                elif mime_type == 'application/vnd.google-apps.presentation':
                    # Check if we should use the enhanced slide extraction
                    if hasattr(self.config, 'USE_ENHANCED_SLIDES') and self.config.USE_ENHANCED_SLIDES:
                        # Use the enhanced slide extraction with GPT-4 Vision
                        return self._extract_slides_with_vision(file_id, file_name)
                    else:
                        # Fallback to basic text extraction
                        request = self.drive_service.files().export_media(fileId=file_id, mimeType='text/plain')
                        content = self._download_file(request)
                        return content.decode('utf-8')

                else:
                    self.logger.warning(f"Unsupported Google Workspace format: {mime_type}")
                    return ""

            # Handle regular files (need to download)
            else:
                request = self.drive_service.files().get_media(fileId=file_id)
                content = self._download_file(request)

                # Process based on MIME type
                if mime_type == 'application/pdf':
                    return self._extract_text_from_pdf(content)
                elif mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                    return self._extract_text_from_docx(content)
                elif mime_type == 'application/vnd.openxmlformats-officedocument.presentationml.presentation':
                    return self._extract_text_from_pptx(content)
                elif mime_type in ['text/plain', 'text/markdown', 'text/html']:
                    text_content = content.decode('utf-8')
                    if mime_type == 'text/html':
                        return self.html_to_markdown(text_content)
                    return text_content
                else:
                    self.logger.warning(f"Unsupported file format: {mime_type}")
                    return ""

        except Exception as e:
            self.logger.error(f"Error processing file {file_name}: {str(e)}")
            return ""

    def _download_file(self, request):
        """Download a file from Google Drive using the provided request"""
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False

        while not done:
            status, done = downloader.next_chunk()

        fh.seek(0)
        return fh.read()

    def _extract_text_from_pdf(self, content):
        """Extract text from a PDF file"""
        pdf_file = io.BytesIO(content)
        text = ""

        try:
            reader = PyPDF2.PdfReader(pdf_file)
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text() + "\n\n"
            return text
        except Exception as e:
            self.logger.error(f"Error extracting text from PDF: {str(e)}")
            return ""

    def _extract_text_from_docx(self, content):
        """Extract text from a DOCX file"""
        docx_file = io.BytesIO(content)
        text = ""

        try:
            doc = docx.Document(docx_file)
            for para in doc.paragraphs:
                text += para.text + "\n"
            return text
        except Exception as e:
            self.logger.error(f"Error extracting text from DOCX: {str(e)}")
            return ""

    # Then add this new helper method to the GoogleDriveIndexer class
    def _extract_text_from_pptx(self, content):
        """Extract text from a PPTX file"""
        pptx_file = io.BytesIO(content)
        text = ""

        try:
            presentation = pptx.Presentation(pptx_file)

            # Extract title if available
            if presentation.core_properties.title:
                text += f"# {presentation.core_properties.title}\n\n"

            # Extract text from slides
            for i, slide in enumerate(presentation.slides):
                text += f"## Slide {i + 1}\n\n"

                # Extract text from shapes
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        text += f"{shape.text.strip()}\n\n"

                # Add slide separator
                text += "---\n\n"

            return text
        except Exception as e:
            self.logger.error(f"Error extracting text from PPTX: {str(e)}")
            return ""
            
    def _extract_slides_with_vision(self, presentation_id, presentation_name):
        """
        Enhanced method to extract content from Google Slides using the Slides API
        and GPT-4 Vision for richer content extraction including visuals.
        
        Args:
            presentation_id: The ID of the Google Slides presentation
            presentation_name: The name of the presentation
            
        Returns:
            A markdown string containing the enhanced slide content
        """
        self.logger.info(f"Extracting slides with vision for {presentation_name} ({presentation_id})")
        
        try:
            # Initialize slides service if not already done
            if not hasattr(self, 'slides_service'):
                credentials = service_account.Credentials.from_service_account_file(
                    self.config.GOOGLE_DRIVE_CREDENTIALS_FILE,
                    scopes=['https://www.googleapis.com/auth/drive.readonly']
                )
                self.slides_service = build('slides', 'v1', credentials=credentials)
                
            # Get presentation
            presentation = self.slides_service.presentations().get(
                presentationId=presentation_id
            ).execute()
            
            # Get all slides
            slides = presentation.get('slides', [])
            self.logger.info(f"Processing {len(slides)} slides from {presentation_name}")
            
            # Final markdown content
            full_content = f"# {presentation_name}\n\n"
            
            # Limit number of slides to process if specified in config
            max_slides = getattr(self.config, 'MAX_SLIDES_TO_PROCESS', len(slides))
            slides_to_process = slides[:max_slides]
            
            # Process each slide
            for slide_index, slide in enumerate(slides_to_process):
                slide_number = slide_index + 1
                self.logger.info(f"Processing slide {slide_number}/{len(slides)}")
                
                # Export the slide as PNG
                try:
                    export_request = self.slides_service.presentations().pages().getThumbnail(
                        presentationId=presentation_id,
                        pageObjectId=slide['objectId'],
                        thumbnailProperties_thumbnailSize='LARGE'
                    ).execute()
                    
                    # Get the thumbnail URL
                    thumbnail_url = export_request.get('contentUrl')
                    
                    # Download the image
                    image_response = requests.get(thumbnail_url, timeout=30)
                    if image_response.status_code != 200:
                        self.logger.warning(f"Failed to download slide {slide_number} image: {image_response.status_code}")
                        continue
                        
                    image_content = image_response.content
                    
                    # Analyze with GPT-4 Vision
                    slide_markdown = self._analyze_slide_with_llm(image_content, slide_number)
                    if slide_markdown:
                        full_content += f"{slide_markdown}\n\n---\n\n"
                    else:
                        # Fallback to basic extraction if vision analysis fails
                        slide_title = f"Slide {slide_number}"
                        slide_content = []
                        
                        # Extract text elements
                        for element in slide.get('pageElements', []):
                            if 'shape' in element and 'text' in element['shape']:
                                text_content = ""
                                for textElement in element['shape']['text'].get('textElements', []):
                                    if 'textRun' in textElement and 'content' in textElement['textRun']:
                                        text_content += textElement['textRun']['content']
                                
                                # Check if this might be a title
                                if 'title' in element.get('objectId', '').lower() or (
                                        'transform' in element and element['transform'].get('scaleY', 0) > 1):
                                    slide_title = text_content.strip()
                                else:
                                    # Add to content if not empty
                                    if text_content.strip():
                                        slide_content.append(text_content.strip())
                        
                        # Format as markdown
                        slide_md = f"## {slide_title}\n\n"
                        for item in slide_content:
                            slide_md += f"{item}\n\n"
                            
                        full_content += f"{slide_md}\n\n---\n\n"
                
                except Exception as e:
                    self.logger.error(f"Error processing slide {slide_number}: {str(e)}")
                    continue
            
            return full_content
            
        except Exception as e:
            self.logger.error(f"Error extracting slides with vision: {str(e)}")
            # Fall back to plain text extraction
            request = self.drive_service.files().export_media(fileId=presentation_id, mimeType='text/plain')
            content = self._download_file(request)
            return content.decode('utf-8')
    
    def _analyze_slide_with_llm(self, image_content, slide_number):
        """
        Analyze a slide image using GPT-4 Vision
        
        Args:
            image_content: Raw image data
            slide_number: The slide number for reference
            
        Returns:
            Markdown string with the analysis result
        """
        prompt = self.config.GOOGLE_SLIDE_IMAGE_TO_TEXT_PROMPT
        # Use the enhancement service to analyze the image
        return enhancement_service.analyze_image_with_llm(image_content, prompt, getattr(self.config, 'OPENAI_VISION_MODEL', 'gpt-4o'))

    def condense_content_using_llm(self, content):
        """Summarize content using OpenAI's API"""
        # Use the enhancement service to condense content
        return enhancement_service.condense_content_using_llm(content)

    def html_to_markdown(self, html_text):
        """Convert HTML to Markdown format"""
        markdown_text = md(html_text, newline_style="BACKSLASH", default_title=True, heading_style='ATX')
        markdown_text = markdown_text.replace('\n\n', '\n')

        if self.config.SUMMARIZE_CONTENT:
            self.logger.info("Summarizing content using LLM...")
            markdown_text = self.condense_content_using_llm(markdown_text)

        return markdown_text

    def prepare_drive_documents(self):
        """Process all supported files from Google Drive"""
        self.logger.info("Fetching and processing Google Drive documents...")

        # Get folder_id from config if available, otherwise use None (root)
        folder_id = getattr(self.config, 'GOOGLE_DRIVE_FOLDER_ID', None)
        recursive = getattr(self.config, 'GOOGLE_DRIVE_RECURSIVE', True)

        files = self.get_supported_files(folder_id, recursive)
        records = []

        for file in tqdm(files):
            try:
                self.logger.info(f"Processing {file['name']}...")

                # Download and extract content
                content = self.download_and_extract_content(file)

                # Skip if content extraction failed
                if not content.strip():
                    self.logger.warning(f"Empty content for file: {file['name']}")
                    continue

                # Add file title if not in content
                if file['name'] not in content:
                    content = f"# {file['name']}\n\n{content}"

                # Summarize if needed
                if self.config.SUMMARIZE_CONTENT:
                    content = self.condense_content_using_llm(content)

                # Create record
                record = {
                    'title': file['name'],
                    'url': file.get('webViewLink', ''),
                    'markdown': content
                }
                records.append(record)

            except Exception as ex:
                self.logger.error(f"Could not process {file['name']} due to: {ex}")

        self.logger.info(f"Prepared {len(records)} document records!")

        # Save intermediate file if needed
        if self.config.SAVE_INTERMEDIATE_FILES:
            processed_file = os.path.join(self.config.OUTPUT_DIR, "drive_processed.json")
            with open(processed_file, 'w') as f:
                json.dump(records, f, indent=2)

        return records

    def get_embedding_dimensions(self, model_name):
        """Get the dimensions for a specific OpenAI embedding model"""
        # Common OpenAI embedding model dimensions
        model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
            # Add more models as needed
        }
        return model_dimensions.get(model_name, 1536)  # Default to 1536 if unknown

    def index_to_pinecone(self, records):
        """Index processed records to Pinecone"""
        self.logger.info(f"Indexing {len(records)} records to Pinecone...")

        # Then in your initialize_pinecone method or before creating the index:
        embedding_dim = self.get_embedding_dimensions(self.config.OPENAI_EMBEDDING_MODEL)

        # Initialize Pinecone
        pc = Pinecone(api_key=self.config.PINECONE_API_KEY)

        # Check if index exists, create if not
        available_indexes = pc.list_indexes().names()
        if self.config.PINECONE_INDEX_NAME not in available_indexes:
            self.logger.info(f"Index '{self.config.PINECONE_INDEX_NAME}' not found. Creating a new index...")

            pc.create_index(
                name=self.config.PINECONE_INDEX_NAME,
                dimension=embedding_dim,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud=self.config.PINECONE_CLOUD,
                    region=self.config.PINECONE_REGION
                )
            )
            self.logger.info(f"Index '{self.config.PINECONE_INDEX_NAME}' created successfully.")

        # Create loader for the records
        loader = CustomJsonLoader(
            records,
            dataset_mapping_function=lambda item: Document(
                page_content=item["markdown"] or "",
                metadata={'url': item["url"], "title": item["title"]}
            )
        )

        # Initialize embeddings with model-appropriate parameters
        # Newer models don't support explicit dimensions parameter
        if self.config.OPENAI_EMBEDDING_MODEL in ["text-embedding-3-small", "text-embedding-3-large"]:
            embeddings = OpenAIEmbeddings(
                api_key=self.config.OPENAI_API_KEY,
                model=self.config.OPENAI_EMBEDDING_MODEL
            )
        else:
            # Older models like text-embedding-ada-002 support dimensions
            embeddings = OpenAIEmbeddings(
                api_key=self.config.OPENAI_API_KEY,
                model=self.config.OPENAI_EMBEDDING_MODEL,
                dimensions=embedding_dim
            )

        # Load and split documents
        documents = loader.load()
        self.logger.info(f"Loaded {len(documents)} documents.")

        # Split text into smaller chunks for better embedding
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n"],
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP
        )
        docs = text_splitter.split_documents(documents)
        self.last_chunks = docs  # Store for reporting
        self.logger.info(f"Split into {len(docs)} chunks.")

        # Store in Pinecone
        try:
            self.logger.info("Uploading documents to Pinecone...")
            vectorstore = PineconeVectorStore.from_documents(
                docs,
                index_name=self.config.PINECONE_INDEX_NAME,
                pinecone_api_key=self.config.PINECONE_API_KEY,
                embedding=embeddings
            )
            self.logger.info(
                f"Successfully indexed {len(docs)} document chunks to Pinecone index '{self.config.PINECONE_INDEX_NAME}'.")
            return True
        except Exception as e:
            self.logger.error(f"Error indexing to Pinecone: {str(e)}")
            return False

    def run_full_process(self):
        """Initialize the Google Drive API and validate settings"""
        try:
            # Validate credentials and API access
            files = self.list_folder_contents()
            
            self.logger.info(f"Successfully connected to Google Drive API, found {len(files)} accessible files")
            return {
                "status": "success", 
                "message": f"Successfully connected to Google Drive API, found {len(files)} accessible files",
                "files_count": len(files)
            }

        except Exception as e:
            self.logger.error(f"Error initializing Google Drive API: {str(e)}")
            return {"status": "error", "message": str(e)}
            
    def get_google_drive_files(self) -> Dict[str, Any]:
        """Get list of indexed Google Drive files"""
        try:
            # Try to load the Google Drive processed files
            drive_path = os.path.join(self.config.OUTPUT_DIR, "drive_processed.json")
            
            if os.path.exists(drive_path):
                with open(drive_path, "r") as f:
                    files = json.load(f)
                    
                # Extract basic file information
                file_list = [
                    {
                        "id": idx,
                        "title": file.get("title", "Unknown"),
                        "url": file.get("url", ""),
                        "size": len(file.get("markdown", "")) if "markdown" in file else 0
                    }
                    for idx, file in enumerate(files)
                ]
                
                return {
                    "status": "success",
                    "files": file_list,
                    "count": len(file_list)
                }
            else:
                # Check if we can query the vector store directly
                try:
                    pc = Pinecone(api_key=self.config.PINECONE_API_KEY)
                    
                    if self.config.PINECONE_INDEX_NAME in pc.list_indexes().names():
                        index = pc.Index(self.config.PINECONE_INDEX_NAME)
                        stats = index.describe_index_stats()
                        
                        return {
                            "status": "success",
                            "files": [],
                            "count": 0,
                            "vector_count": stats.total_vector_count,
                            "message": "Drive file list not available, but vectors are in the index"
                        }
                except Exception as e:
                    self.logger.error(f"Error querying Pinecone for Google Drive files: {str(e)}")
                
                # No processed files available
                return {
                    "status": "success",
                    "files": [],
                    "count": 0,
                    "message": "No Google Drive files indexed or file list not available"
                }
                
        except Exception as e:
            self.logger.error(f"Error retrieving Google Drive files: {str(e)}")
            return {"status": "error", "message": str(e)}