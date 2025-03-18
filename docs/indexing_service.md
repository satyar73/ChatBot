# Indexing Service Documentation
Documentation for the ChatBotExample's Indexing Service, manage content ingestion and vector indexing
for Retrieval-Augmented Generation (RAG).

## 1. Overview
The Indexing Service is responsible for:
- Fetching content from various sources (Shopify, Google Drive)
- Processing and transforming content to embeddings
- Indexing content to a vector database (Right now only Pinecone is supported)
- Managing vector indexes

The service consists of three main components:
- **IndexService**: The primary service interface that coordinates indexing
- **ShopifyIndexer**: Fetches and processes Shopify content
- **GoogleDriveIndexer**: Fetches and processes Google Drive content

## 2. IndexService

The `IndexService` class provides a high-level interface for creating and managing vector indexes.
### 2.1 Key Methods

#### `create_index(store, summarize) -> Dict`
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

**Returns**:
- Dictionary with status and result information

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

## 3. ShopifyIndexer
The `ShopifyIndexer` class fetches content from a Shopify store and indexes it to Pinecone.

### 3.1 Key Components

- **API Integration**: Connects to Shopify Admin API
- **Content Processors**: Converts HTML to markdown and processes content
- **Vector Indexing**: Creates embeddings and uploads to Pinecone
- **Attribution Enhancement**: Adds marketing attribution metadata

### 3.2 Key Methods

#### `run_full_process() -> Dict`
Runs the complete Shopify indexing process.

**Returns**:
- Dictionary with status and message

#### `index_all_content() -> bool`
Indexes all Shopify content (blogs, articles, products) to Pinecone.

**Returns**:
- True if indexing was successful, False otherwise

#### `get_blogs() -> List[Dict]`
Gets all blogs from Shopify store.

**Returns**:
- List of blog objects containing id, handle, title, and updated_at

#### `get_articles(blog_id) -> List[Dict]`
Gets all articles for a specific blog.

**Parameters**:
- `blog_id`: The Shopify blog ID

**Returns**:
- List of article objects
#### `get_products() -> List[Dict]`

Gets all products from Shopify store.

**Returns**:
- List of product objects

#### `html_to_markdown(html_content) -> str`
Converts HTML content to markdown for better chunking and indexing.

**Parameters**:
- `html_content`: HTML content to convert

**Returns**:
- Markdown string

#### `index_to_pinecone(records) -> bool`
Indexes content records to Pinecone vector database.

**Parameters**:
- `records`: List of content records with title, url, and markdown

**Returns**:
- True if indexing was successful, False otherwise

### 3.3 Configuration
ShopifyIndexer requires the following configuration:
- `SHOPIFY_SHOP_DOMAIN` or `SHOPIFY_STORE`: Shopify store domain
- `SHOPIFY_API_KEY`: Shopify API key
- `SHOPIFY_API_VERSION`: Shopify API version
- `PINECONE_API_KEY`: Pinecone API key
- `PINECONE_INDEX_NAME`: Pinecone index name
- `PINECONE_DIMENSION`: Embedding dimensions
- `OPENAI_API_KEY`: OpenAI API key
- `OPENAI_EMBEDDING_MODEL`: OpenAI embedding model name

## 4. GoogleDriveIndexer
The `GoogleDriveIndexer` class fetches content from Google Drive and indexes it to Pinecone.

### 4.1 Key Components
- **Google Drive API**: Connects to Google Drive
- **Document Processors**: Extracts text from various file formats
- **Content Conversion**: Converts documents to markdown
- **Vector Indexing**: Creates embeddings and uploads to Pinecone

### 4.2 Key Methods
#### `run_full_process() -> Dict`

Runs the complete Google Drive indexing process.

**Returns**:
- Dictionary with status and message

#### `prepare_drive_documents() -> List[Dict]`
Processes all supported files from Google Drive.

**Returns**:
- List of document records with title, url, and content

#### `get_supported_files(folder_id, recursive) -> List[Dict]`

Gets all supported files, optionally from a specific folder.

**Parameters**:
- `folder_id`: Optional folder ID to start from
- `recursive`: Whether to process subfolders recursively

**Returns**:
- List of file objects

#### `download_and_extract_content(file_item) -> str`

Downloads and extracts text content from a Google Drive file.

**Parameters**:
- `file_item`: Google Drive file object

**Returns**:
- Extracted text content

#### `condense_content_using_llm(content) -> str`

Summarize content using OpenAI's API.

**Parameters**:
- `content`: Original content text

**Returns**:
- Summarized content

#### `index_to_pinecone(records) -> bool`

Indexes content records to Pinecone vector database.

**Parameters**:
- `records`: List of content records with title, url, and markdown

**Returns**:
- True if indexing was successful, False otherwise

### 4.3 Configuration

GoogleDriveIndexer requires the following configuration:
- `GOOGLE_DRIVE_CREDENTIALS_FILE`: Path to service account credentials
- `GOOGLE_DRIVE_FOLDER_ID`: Optional folder ID to process
- `GOOGLE_DRIVE_RECURSIVE`: Whether to process recursively
- `PINECONE_API_KEY`: Pinecone API key
- `PINECONE_INDEX_NAME`: Pinecone index name
- `OPENAI_API_KEY`: OpenAI API key
- `OPENAI_EMBEDDING_MODEL`: OpenAI embedding model name
- `OPENAI_SUMMARY_MODEL`: Model to use for summarization

## 5. Usage Examples

### 5.1 Creating a Shopify Index

```python
from app.services.index_service import IndexService

# Initialize the service
index_service = IndexService()

# Create an index
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

### 5.2 Creating a Google Drive Index

```python
from app.services.index_service import IndexService

# Initialize the service
index_service = IndexService()

# Create an index from Google Drive
result = await index_service.create_index_from_drive(
    folder_id="your_folder_id",
    recursive=True,
    summarize=True
)

# Check the result
if result["status"] == "success":
    print("Google Drive content indexed successfully")
else:
    print(f"Indexing failed: {result['message']}")
```

### 5.3 Getting Index Information

```python
from app.services.index_service import IndexService

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

### 5.4 Deleting an Index

```python
from app.services.index_service import IndexService

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

## 6. Dependencies
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

## 7. Troubleshooting

### 7.1 Common Issues
- **API Authentication Errors**: Verify API keys and permissions
- **Rate Limiting**: Implement retries and backoff for API requests
- **Large File Handling**: Monitor memory usage when processing large files
- **Embedding Errors**: Check compatibility between embedding dimension and Pinecone index

### 7.2 Logging

The Indexing Service uses standard logging to track progress:
- `INFO`: Records general progress and statistics
- `WARNING`: Notes potential issues that don't stop indexing
- `ERROR`: Records failures in indexing process
- `DEBUG`: Provides detailed information for troubleshooting

## 8. Best Practices
- **Incremental Indexing**: Consider implementing delta updates rather than full reindexing
- **Content Pre-processing**: Optimize content before embedding (clean HTML, remove boilerplate)
- **Index Monitoring**: Regularly check vector counts and dimensions
- **Content Chunking**: Adjust chunk size based on content type for better retrieval
- **Summarization**: Use summarization for very long documents, but preserve key information
- **Metadata Enrichment**: Add rich metadata to improve filtering during retrieval