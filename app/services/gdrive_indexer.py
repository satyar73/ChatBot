import os
import json
from tqdm import tqdm
from markdownify import markdownify as md
from typing import Callable, Dict, List, Union, Optional
from pathlib import Path
import logging
import io
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
import docx
import PyPDF2
import pptx
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

# Import from your existing project structure
from app.config.chat_config import ChatConfig

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
                    # Export Google Slides as plain text
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

    def condense_content_using_llm(self, content):
        """Summarize content using OpenAI's API"""
        client = OpenAI(api_key=self.config.OPENAI_API_KEY)

        response = client.chat.completions.create(
            model=self.config.OPENAI_SUMMARY_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"<markdown>\n{content}\n</markdown>\nYou should shorten the above markdown text to MAXIMUM OF 800 characters while making sure ALL THE HEADINGS AND HYPERLINKS are retained so that the users can refer to those links later. In your response, don't include <markdown> tags."
                        }
                    ]
                }
            ],
            temperature=0,
            max_tokens=1000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].message.content

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
            processed_file = getattr(self.config, 'DRIVE_PROCESSED_FILE',
                                     os.path.join(self.config.OUTPUT_DIR, "drive_processed.json"))
            with open(self.config.DRIVE_PROCESSED_FILE, 'w') as f:
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

        # Initialize embeddings
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
        """Run the complete indexing process"""
        try:
            # Step 1: Process all files from Google Drive
            records = self.prepare_drive_documents()

            # Step 2: Index to Pinecone
            success = self.index_to_pinecone(records)

            if success:
                self.logger.info("✅ Complete process finished successfully!")
                return {"status": "success", "message": "Indexing completed successfully"}
            else:
                self.logger.error("❌ Process completed with errors in indexing step.")
                return {"status": "error", "message": "Indexing failed"}

        except Exception as e:
            self.logger.error(f"Error in main process: {str(e)}")
            return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    # Run the indexer directly if executed as a script
    indexer = GoogleDriveIndexer()
    result = indexer.run_full_process()
    print(result)