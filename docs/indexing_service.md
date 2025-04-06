# Indexing Service Documentation
Documentation for the ChatBotExample's Indexing Service, manage content ingestion and vector indexing
for Retrieval-Augmented Generation (RAG).

## 1. Overview
The Indexing Service is responsible for:
- Fetching content from various sources (Shopify, Google Drive --> supported now)
- Processing and transforming content to embeddings
- Indexing content to a vector database (Right now only Pinecone is supported)
- Managing vector indexes

The service consists of four main components:
- **IndexService**: The primary service interface that coordinates indexing
- **ContentProcessor**: Base class for processing and indexing document content
- **ShopifyIndexer**: Fetches and processes Shopify content
- **GoogleDriveIndexer**: Fetches and processes Google Drive content

## 2. IndexService

The `IndexService` class provides a high-level interface for creating and managing vector indexes.
### 2.1 Key Methods

#### `create_shopify_index(store, summarize) -> Dict`
Creates and populates a new vector index with Shopify content.

**Parameters**:
- `store`: Optional Shopify store domain
- `summarize`: Optional boolean to enable content summarization

**Returns**:
- Dictionary with status and result information

#### `create_index_from_drive(folder_id, recursive, summarize) -> Dict`
Creates and populates a vector index with Google Drive data.

**Parameters**:
- `folder_id`: Optional Google Drive folder ID
- `recursive`: Optional boolean to enable recursive folder processing
- `summarize`: Optional boolean to enable content summarization
- 'enhanced_slides': Optional boolean for enhanced slide processing

**Returns**:
- Dictionary with status and result information

#### `get_google_drive_files() -> Dict[str, Any]`
Get list of indexed Google Drive files

**Returns**:
- Dictionary with:
  - File Id
  - Title
  - URL
  - File size

#### `get_index_info() -> Dict`
Gets information about the current vector index.

**Returns**:
- Dictionary with:
  - Index existence status
  - Index name
  - Index statistics
  - Content metadata

#### `delete_index() -> Dict`
Deletes the current vector index.

**Returns**:
- Dictionary with status and result message

### 2.2 Configuration

The IndexService relies on configuration from:
- `app.config.chat_config.ChatConfig`: Contains API keys and index settings

## 3. ContentProcessor
The `ContentProcessor` class provides a base class for processing and indexing document content to vector stores. It's designed to be a common foundation for all indexers in the system.

### 3.1 Key Components
- **Base Processing Framework**: Provides common functionality used by different content indexers
- **Vector Store Integration**: Supports indexing to multiple vector stores through VectorStoreClient
- **EnhancementService Integration**: Leverages the enhancement service for content enrichment 
- **Adaptive Chunking**: Implements specialized chunking strategies for different content types
- **Attribution Metadata**: Enriches documents with attribution-related metadata
- **Embedding Optimization**: Creates enhanced embedding prompts for technical content
- **Keyword Extraction**: Adds relevant keywords based on content analysis

### 3.2 Key Methods

#### `get_index_info() -> Dict[str, Any]`
Gets information about the current vector index.

**Returns**:
- Dictionary containing:
  - total_documents: Number of documents indexed
  - total_chunks: Number of chunks/vectors
  - index_name: Name of the vector index
  - last_updated: Timestamp of last update
  - dimension: Embedding dimension
  
#### `prepare_documents_for_indexing(records) -> List[Document]`
Processes documents and prepares them for indexing by chunking content, adding metadata, and creating optimal embedding prompts.

**Parameters**:
- `records`: List of content records with title, url, and markdown

**Flow**:
1. Extracts keywords from QA content for all chunks
2. For each content record:
   - Analyzes content type and decides on chunking strategy:
      - Preserves Q&A pairs without splitting
      - Uses smaller chunks with more overlap for technical content
      - Uses standard chunking for general content
   - Adds attribution metadata with the enhancement service
   - Enhances chunks with relevant keywords
   - Creates optimized embedding prompts for better retrieval
   - Assembles Document objects with content and metadata

**Returns**:
- List of Document objects ready for indexing to vector stores

#### `index_to_vector_store(docs) -> bool`
Indexes documents to all configured vector stores.

**Parameters**:
- `docs`: List of Document objects to index

**Flow**:
1. Iterates through all configured vector stores
2. For each vector store:
   - Gets the appropriate vector store client
   - Indexes the documents to that vector store
   - Tracks success across all operations

**Returns**:
- True if indexing was successful for all vector stores, False otherwise

## 4. ShopifyIndexer
The `ShopifyIndexer` class fetches content from a Shopify store and works with the ContentProcessor for processing and indexing to vector stores.

### 4.1 Key Components
- **Shopify Admin API Integration**: 
  - Connects securely to the Shopify Admin API
  - Fetches blogs, articles, and products using authenticated requests
  - Supports pagination for large content collections
- **Content Processing**: 
  - Converts Shopify HTML content to well-structured markdown
  - Organizes content into records with consistent metadata
  - Preserves content relationships (blogs to articles, products to variants)
- **EnhancementService Integration**:
  - Leverages the enhancement service for content enrichment
  - Identifies attribution and marketing terminology
  - Extracts keywords for better content categorization
  - Processes Q&A content with specialized handling
- **ContentProcessor Collaboration**:
  - Prepares formatted content records for the ContentProcessor
  - Lets ContentProcessor handle chunking, embedding, and indexing
  - Maintains separation of concerns: fetching vs. processing/indexing

### 4.2 Key Methods

#### `get_blogs() -> List[Dict]`
Gets all blogs from Shopify store using the Admin API.

**Flow**:
1. Constructs the Shopify Admin API URL for blogs
2. Sets up authentication with the Shopify API key
3. Makes the API request with appropriate parameters
4. Processes the response and extracts blog data

**Returns**:
- List of blog objects containing id, handle, title, and updated_at

#### `get_articles(blog_id) -> List[Dict]`
Gets all articles for a specific blog.

**Parameters**:
- `blog_id`: The Shopify blog ID

**Flow**:
1. Constructs the Shopify Admin API URL for articles
2. Sets up authentication and pagination parameters
3. Makes the API request
4. Processes the response and extracts article data

**Returns**:
- List of article objects with content and metadata

#### `get_products() -> List[Dict]`
Gets all products from Shopify store.

**Flow**:
1. Constructs the Shopify Admin API URL for products
2. Sets up authentication and product field parameters
3. Makes the API request with appropriate pagination
4. Processes the response and extracts product data

**Returns**:
- List of product objects with descriptions and metadata

#### `html_to_markdown(html_content) -> str`
Converts HTML content to markdown for better processing.

**Parameters**:
- `html_content`: HTML content from Shopify

**Flow**:
1. Uses markdownify library to convert HTML to markdown
2. Cleans up the markdown formatting
3. Preserves important structural elements like headers and lists

**Returns**:
- Cleaned markdown string

#### `prepare_blog_articles() -> List[Dict[str, Any]]`
Prepares blog articles for processing by ContentProcessor.

**Flow**:
1. Fetches all blogs from Shopify store
2. For each blog, fetches all published articles
3. Converts article HTML content to markdown
4. Creates structured content records for each article
5. Adds consistent metadata including content type, URL, and title

**Returns**:
- List of article records ready for ContentProcessor

#### `prepare_products() -> List[Dict[str, Any]]`
Prepares products for processing by ContentProcessor.

**Flow**:
1. Fetches all products from Shopify store
2. Converts product HTML descriptions to markdown
3. Creates structured content records for each product
4. Adds metadata with product information and type

**Returns**:
- List of product records ready for ContentProcessor

#### `run_full_process(content_processor) -> Dict`
Coordinates the complete Shopify indexing process.

**Parameters**:
- `content_processor`: ContentProcessor instance for document processing

**Flow**:
1. Prepares blog and article records
2. Prepares product records
3. Combines all records into a single collection
4. Passes the records to ContentProcessor for:
   - Document preparation
   - Chunking
   - Embedding 
   - Vector store indexing
5. Returns status information with document counts

**Returns**:
- Dictionary with status, message, and document counts

### 4.3 Configuration
ShopifyIndexer requires the following configuration:
- `SHOPIFY_SHOP_DOMAIN` or `SHOPIFY_STORE`: Shopify store domain
- `SHOPIFY_API_KEY`: Shopify API key
- `SHOPIFY_API_VERSION`: Shopify API version
- `PINECONE_API_KEY`: Pinecone API key
- `PINECONE_INDEX_NAME`: Pinecone index name
- `PINECONE_DIMENSION`: Embedding dimensions (default 1536)
- `OPENAI_API_KEY`: OpenAI API key
- `OPENAI_EMBEDDING_MODEL`: OpenAI embedding model name (default "text-embedding-3-small")
- `OPENAI_VISION_MODEL`: OpenAI vision model for image analysis (default "gpt-4o")
- `INDEXER_FEATURE_FLAGS`: Feature flags controlling special indexing behaviors (see indexerfeatureflags.json)

## 5. GoogleDriveIndexer
The `GoogleDriveIndexer` class fetches content from Google Drive and works with ContentProcessor to process and index documents to vector stores, with specialized handling for different file types.

### 5.1 Key Components
- **Google Drive API Integration**: 
  - Connects securely to Google Drive using service account credentials
  - Supports file and folder operations with appropriate permissions
  - Provides recursive folder traversal capabilities
- **Multi-Format Document Processing**: 
  - Supports multiple file formats through specialized handlers:
    - Microsoft Office formats (DOCX, PPTX)
    - PDF documents
    - Plain text and Markdown
    - Google Workspace files (Docs, Sheets, Slides)
  - Extracts text with format preservation
- **Document Organization**:
  - Maintains folder structure information
  - Tracks document paths and relationships
  - Identifies client-specific content through folder analysis
- **ContentProcessor Collaboration**:
  - Prepares document records for the ContentProcessor
  - Delegates chunking, embedding, and indexing to ContentProcessor
  - Maintains separation of concerns: fetching vs. processing/indexing

### 5.2 Key Methods

#### `_initialize_drive_api()`
Initializes the Google Drive API client with service account credentials.

**Flow**:
1. Loads service account credentials from the specified JSON file
2. Configures API access scopes for read-only operations
3. Builds the Drive API service client

#### `get_supported_files(folder_id, recursive) -> List[DriveItem]`
Gets all supported files from Google Drive, optionally from a specific folder.

**Parameters**:
- `folder_id`: Optional folder ID to start from (uses root if not specified)
- `recursive`: Whether to process subfolders recursively

**Flow**:
1. Queries Google Drive API for files matching supported MIME types
2. Handles pagination for large folder contents
3. If recursive, traverses subfolders with proper path tracking
4. Builds complete file metadata including paths and client information

**Returns**:
- List of DriveItem objects with complete metadata

#### `download_and_extract_content(file_item) -> str`
Downloads and extracts text content from a Google Drive file based on its type.

**Parameters**:
- `file_item`: DriveItem object with file metadata

**Flow**:
1. Determines the appropriate extraction method based on MIME type
2. Downloads the file content using the Drive API
3. Processes the content based on file type:
   - Extracts text from DOCX using docx library
   - Extracts text from PDF using PyPDF2
   - Extracts text and structure from PPTX using pptx library
   - Handles plain text and other formats directly
4. Returns the extracted content with format preservation

**Returns**:
- Extracted text content in markdown format

#### `prepare_drive_documents(content_processor) -> List[Dict]`
Prepares documents from Google Drive for processing by ContentProcessor.

**Parameters**:
- `content_processor`: ContentProcessor instance

**Flow**:
1. Gets all supported files using get_supported_files
2. For each file:
   - Downloads and extracts content
   - Creates structured records with metadata
   - Adds client and path information
3. Prepares the records for processing

**Returns**:
- List of document records ready for ContentProcessor

#### `run_full_process(content_processor) -> Dict`
Coordinates the complete Google Drive indexing process.

**Parameters**:
- `content_processor`: ContentProcessor instance

**Flow**:
1. Prepares document records from Google Drive
2. Passes records to ContentProcessor for:
   - Document preparation (chunking, metadata enhancement)
   - Embedding generation
   - Vector store indexing
3. Returns status information with document counts

**Returns**:
- Dictionary with status, message, and document counts

### 5.3 Configuration

GoogleDriveIndexer requires the following configuration:
- `GOOGLE_DRIVE_CREDENTIALS_FILE`: Path to service account credentials
- `GOOGLE_DRIVE_FOLDER_ID`: Optional folder ID to process
- `GOOGLE_DRIVE_RECURSIVE`: Whether to process recursively (default true)
- `PINECONE_API_KEY`: Pinecone API key
- `PINECONE_INDEX_NAME`: Pinecone index name
- `PINECONE_DIMENSION`: Embedding dimensions (default 1536)
- `OPENAI_API_KEY`: OpenAI API key
- `OPENAI_EMBEDDING_MODEL`: OpenAI embedding model name (default "text-embedding-3-small")
- `OPENAI_SUMMARY_MODEL`: Model to use for summarization (default "gpt-4o")
- `OPENAI_VISION_MODEL`: OpenAI vision model for image analysis (default "gpt-4o")
- `VISION_MAX_TOKENS`: Maximum tokens for vision response (default 4000)
- `INDEXER_FEATURE_FLAGS`: Feature flags controlling special indexing behaviors:
  - `use_vision_api`: Whether to analyze images in documents and presentations
  - `extract_slides_images`: Whether to extract and process images from slides
  - `enhance_with_keywords`: Whether to enhance documents with extracted keywords
  - `use_enhanced_embeddings`: Whether to use specialized embedding prompts

## 6. Usage Examples

### 6.1 Creating a Shopify Index

```python
from app.services.indexing.index_service import IndexService

# Initialize the service
index_service = IndexService()

# Create an index with all Shopify content
result = await index_service.create_index(
    store="your-store.myshopify.com",
    summarize=True
)

# Check the result
if result["status"] == "success":
    print(f"Successfully indexed {result['message']}")
else:
    print(f"Indexing failed: {result['message']}")
```

### 6.2 Creating a Google Drive Index

```python
from app.services.indexing.index_service import IndexService

# Initialize the service
index_service = IndexService()

# Create an index from Google Drive
result = await index_service.create_index_from_drive(
    folder_id="your_folder_id",  # Optional - uses root folder if not specified
    recursive=True,  # Process subfolders recursively
    summarize=True  # Use AI to summarize long documents
)

# Check the result
if result["status"] == "success":
    print(f"Google Drive content indexed successfully: {result['count']} documents processed")
else:
    print(f"Indexing failed: {result['message']}")
```

### 6.3 Complete Example with Feature Flags

```python
import asyncio
import json
from app.services.indexing.index_service import IndexService


async def run_indexer():
    # Load feature flags from config file
    with open('indexerfeatureflags.json', 'r') as f:
        feature_flags = json.load(f)

    # Initialize the service with custom configuration
    index_service = IndexService(
        pinecone_api_key="your_pinecone_api_key",
        index_name="your_index_name",
        dimension=1536
    )

    # First, check if index exists and delete if needed
    info = await index_service.get_index_info()
    if info["exists"]:
        print(f"Deleting existing index '{info['name']}'")
        await index_service.delete_index()

    # Enable special features for image processing
    feature_flags["use_vision_api"] = True
    feature_flags["extract_slides_images"] = True

    # Set the feature flags
    index_service.set_feature_flags(feature_flags)

    # Index both Google Drive and Shopify content
    drive_result = await index_service.create_index_from_drive(
        folder_id="1ABC123XYZ",
        recursive=True
    )

    shopify_result = await index_service.create_index(
        store="your-store.myshopify.com"
    )

    # Show final results
    total_docs = drive_result.get('count', 0) + shopify_result.get('count', 0)
    print(f"Indexed {total_docs} total documents")

    # Get final index information
    info = await index_service.get_index_info()
    print(f"Index contains {info['stats']['total_vector_count']} vectors")


# Run the async function
if __name__ == "__main__":
    asyncio.run(run_indexer())
```

### 6.4 Getting Index Information

```python
from app.services.indexing.index_service import IndexService

# Initialize the service
index_service = IndexService()

# Get index info
info = await index_service.get_index_info()

# Check if index exists
if info["exists"]:
    print(f"Index '{info['name']}' exists with {info['stats']['total_vector_count']} vectors")

    # Print namespaces
    for ns_name, ns_info in info["stats"]["namespaces"].items():
        print(f"Namespace '{ns_name}' has {ns_info['vector_count']} vectors")
else:
    print(f"Index '{info['name']}' does not exist")
```

### 6.5 Deleting an Index

```python
from app.services.indexing.index_service import IndexService

# Initialize the service
index_service = IndexService()

# Delete the index
result = await index_service.delete_index()

# Check the result
if result["status"] == "success":
    print(result["message"])
else:
    print(f"Deletion failed: {result['message']}")
```

## 7. Dependencies
The Indexing Service components depend on:

- **Pinecone**: For vector database storage
- **OpenAI**: For generating embeddings and summarization
- **LangChain**: For document processing and vector store integration
- **Google API Client**: For Google Drive access
- **Requests**: For Shopify API access
- **Various document processing libraries**:
  - `markdownify`: For HTML to markdown conversion
  - `docx`: For Word document processing
  - `PyPDF2`: For PDF processing
  - `pptx`: For PowerPoint processing

## 8. Troubleshooting

### 8.1 Common Issues
- **API Authentication Errors**: Verify API keys and permissions
- **Rate Limiting**: Implement retries and backoff for API requests
- **Large File Handling**: Monitor memory usage when processing large files
- **Embedding Errors**: Check compatibility between embedding dimension and Pinecone index

### 8.2 Logging

The Indexing Service uses standard logging to track progress:
- `INFO`: Records general progress and statistics
- `WARNING`: Notes potential issues that don't stop indexing
- `ERROR`: Records failures in indexing process
- `DEBUG`: Provides detailed information for troubleshooting

## 9. Best Practices

### 9.1 Content Optimization
- **Incremental Indexing**: Implement delta updates rather than full reindexing for efficiency
- **Content Pre-processing**: Optimize content before embedding:
  - Clean HTML and remove boilerplate elements
  - Extract and process images with vision models
  - Preserve document structure and hierarchy
  - Transform content to well-formatted markdown
- **Enhanced Chunking Strategies**:
  - Use smaller chunks (256-512 tokens) with more overlap (25-50%) for technical content
  - Use standard chunks (1000 tokens) with minimal overlap (10%) for general content
  - Apply document-specific chunking for different file types
  - Preserve special terms and relationships during chunking

### 9.2 Embedding Quality
- **Custom Embedding Prompts**: Use the EnhancementService for better quality:
  - Create prompts that highlight technical marketing terms
  - Add domain-specific definitions for specialized terminology
  - Emphasize attribution-related concepts in the embedding context
  - Include relationship context for connected concepts
- **AI-Enhanced Content**: Leverage AI capabilities for content understanding:
  - Extract textual content from images using vision models
  - Generate detailed descriptions for slides and figures
  - Summarize long documents while preserving key information
  - Identify key concepts and technical terminology

### 9.3 Metadata and Retrieval
- **Rich Metadata Enrichment**: Add detailed metadata for improved retrieval:
  - Keyword categorization and hierarchical classification
  - Attribution terminology detection and relationship mapping
  - Technical term identification with definitions
  - Q&A relationship preservation and context awareness
  - Image content tagging and description metadata
- **Semantic Filtering**: Add semantic filters to improve response accuracy:
  - Filter by document type (presentation, article, product)
  - Filter by content domain (attribution, tracking, measurement)
  - Filter by technical complexity (beginner, intermediate, advanced)
  - Filter by recency and relevance

### 9.4 Performance and Scaling
- **Batch Processing**: Use efficient batching strategies:
  - Process vectors in batches of 100-200 for optimal performance
  - Implement parallel processing for large document sets
  - Use async operations for I/O-bound tasks
  - Add retry logic for API rate limits
- **Resource Management**:
  - Monitor memory usage when processing large files
  - Implement timeout handling for long-running operations
  - Add progress tracking and logging for observability
  - Use connection pooling for database operations