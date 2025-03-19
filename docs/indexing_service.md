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
The `ShopifyIndexer` class fetches content from a Shopify store and indexes it to Pinecone with 
metadata enrichment and more optimal embedding techniques.

### 3.1 Key Components
- **API Integration**: Connects to Shopify Admin API to fetch blogs, articles, and products
- **Content Processors**: Converts HTML to markdown and processes content with specialized handling
- **Metadata Enrichment**: Enhances records with:
  - Marketing attribution terminology detection
  - Technical term identification and definitions
  - Keyword extraction and categorization
  - Q&A pair processing
- **Optimized Embedding Generation**: Creates custom embedding prompts that highlight technical marketing terms
- **Vector Indexing**: Creates embeddings and uploads to Pinecone with appropriate metadata
- **Chunking Strategies**: Adaptive text chunking based on content type and technical terminology

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

#### `prepare_blog_articles() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]`
Prepares blog articles for indexing.

**Flow**:
1. Fetches all blogs from Shopify store
2. For each blog, creates a blog record with title and URL
3. Fetches all articles for each blog
4. Converts article HTML content to markdown
5. Creates article records with title, URL, and markdown content

**Returns**:
- Tuple of (blog_records, article_records)

#### `prepare_products() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]`
Prepares products for indexing.

**Flow**:
1. Fetches all products from Shopify store
2. Converts product HTML descriptions to markdown
3. Creates product records with title, URL, and markdown content
4. Processes product variants if needed

**Returns**:
- Tuple of (product_records, variant_records)

#### `prepare_qa_pairs(qa_content) -> List[Dict[str, Any]]`
Processes Q&A content to preserve question-answer relationships.

**Parameters**:
- `qa_content`: Raw Q&A content with questions and answers

**Flow**:
1. Parses Q&A content to extract question-answer pairs
2. Preserves the relationship between questions and answers
3. Adds special metadata for certain types of Q&A content
4. Formats Q&A pairs for optimal retrieval

**Returns**:
- List of processed Q&A records

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

#### `extract_keywords_from_qa(qa_content) -> Dict[str, List[str]]`
Extracts keywords from Q&A pairs to use for tagging articles.

**Parameters**:
- `qa_content`: Raw Q&A content with questions and answers

**Returns**:
- Dictionary mapping keyword categories to related terms

#### `enhance_records_with_keywords(records, keyword_map) -> List[Dict]`
Enhances content records with keywords based on content analysis.

**Parameters**:
- `records`: List of content records
- `keyword_map`: Dictionary of keywords and related terms

**Returns**:
- Enhanced records with keywords added

#### `create_embedding_prompt(text, metadata) -> str`
Creates an optimized prompt for embedding that highlights attribution terms and technical concepts.

**Parameters**:
- `text`: Original text to embed
- `metadata`: Metadata associated with the text

**Returns**:
- Enhanced prompt for embedding with additional context

#### `enrich_attribution_metadata(content) -> Dict`
Analyzes content for attribution terminology and creates enhanced metadata.

**Parameters**:
- `content`: Markdown or text content to analyze

**Returns**:
- Dictionary of attribution-related metadata

#### `index_to_pinecone(records) -> bool`
Indexes content records to Pinecone vector database with enhanced metadata and embedding techniques.

**Parameters**:
- `records`: List of content records with title, url, and markdown

**Flow**:
1. Processes records to extract metadata and keywords
2. Creates optimized embedding prompts for technical content
3. Uses adaptive chunking strategies based on content type
4. Applies different chunking sizes for technical vs. general content
5. Generates enhanced embeddings with custom prompts
6. Batches vectors for efficient Pinecone uploads

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
- **Adaptive Content Chunking**: Adjust chunk size based on content type and technical terminology:
  - Use smaller chunks with more overlap for technical content
  - Use standard chunk sizes for general content
  - Preserve special terms during chunking
- **Custom Embedding Prompts**: Enhance embedding quality with context-specific prompting:
  - Add technical definitions for specialized terminology
  - Highlight attribution-related content
  - Provide additional context for domain-specific terms
- **Metadata Enrichment**: Add rich metadata for improved retrieval:
  - Keyword categorization
  - Attribution terminology detection
  - Technical term identification
  - Q&A relationship preservation
- **Batch Processing**: Use efficient batching for vector uploads to improve performance