# Indexing Service Documentation
Documentation for the ChatBotExample's Indexing Service, manage content ingestion and vector indexing
for Retrieval-Augmented Generation (RAG).

## 1. Overview
The Indexing Service is responsible for:
- Fetching content from various sources (Shopify, Google Drive --> supported now)
- Processing and transforming content to embeddings
- Indexing content to a vector database (Right now only Pinecone is supported)
- Managing vector indexes with namespace organization
- Providing a RESTful API for content indexing operations

The service consists of several key components:
- **RESTful API**: Standardized endpoints for indexing operations using source parameters
- **IndexService**: The primary service interface that coordinates indexing
- **ContentProcessor**: Base class for processing and indexing document content
- **ShopifyIndexer**: Fetches and processes Shopify content
- **GoogleDriveIndexer**: Fetches and processes Google Drive content
- **API Models**: Pydantic models for request/response validation and documentation

### 1.1 Key Features
- **Unified RESTful API**: Consistent API design with standardized endpoints and parameters
- **Source Parameter Pattern**: Single set of endpoints handling multiple content sources
- **Namespace Support**: Content organization and isolation for multi-tenant scenarios
- **Enhanced Documentation**: Comprehensive Swagger documentation with examples
- **Adaptive Chunking**: Specialized content processing for different document types
- **Vision API Integration**: Extracts and processes content from images and slides
- **Enhanced Embedding**: Creates optimized embedding prompts for better retrieval

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
- `DEFAULT_NAMESPACE`: Default namespace for indexed content (default "default")
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
- `DEFAULT_NAMESPACE`: Default namespace for indexed content (default "default")
- `INDEXER_FEATURE_FLAGS`: Feature flags controlling special indexing behaviors:
  - `use_vision_api`: Whether to analyze images in documents and presentations
  - `extract_slides_images`: Whether to extract and process images from slides
  - `enhance_with_keywords`: Whether to enhance documents with extracted keywords
  - `use_enhanced_embeddings`: Whether to use specialized embedding prompts
  - `use_namespaces`: Whether to enable namespace support for content organization

## 6. REST API

The Indexing Service provides a RESTful API for managing content indexes. All endpoints follow a standardized approach using a `source` parameter to determine the content source.

### 6.1 API Endpoints

#### `POST /api/index/`
Creates a new index for the specified source.

**Request Body**:
```json
{
  "source": "shopify",            // Required: "shopify" or "google_drive"
  "namespace": "default",         // Optional: Namespace for organizing content
  "summarize": true,              // Optional: Whether to summarize content
  "parameters": {                 // Source-specific parameters
    "store": "your-store.myshopify.com", // For Shopify
    // OR
    "folder_id": "your_folder_id",       // For Google Drive
    "recursive": true                    // For Google Drive
  }
}
```

**Response**:
```json
{
  "status": "success",
  "message": "Successfully indexed content",
  "count": 42,
  "namespace": "default"
}
```

#### `GET /api/index/`
Gets information about the current index.

**Query Parameters**:
- `source`: (Optional) Filter by source ("shopify" or "google_drive")
- `namespace`: (Optional) Filter by namespace

**Response**:
```json
{
  "exists": true,
  "name": "your-index-name",
  "stats": {
    "total_vector_count": 1250,
    "dimension": 1536,
    "namespaces": {
      "default": {
        "vector_count": 750
      },
      "client1": {
        "vector_count": 500
      }
    }
  },
  "metadata": {
    "content_types": ["articles", "products"],
    "last_updated": "2025-04-06T12:34:56Z"
  }
}
```

#### `DELETE /api/index/`
Deletes the index for the specified source.

**Query Parameters**:
- `source`: (Optional) Filter by source ("shopify" or "google_drive")
- `namespace`: (Optional) Filter by namespace

**Response**:
```json
{
  "status": "success",
  "message": "Successfully deleted index"
}
```

#### `GET /api/index/files`
Gets a list of files indexed from the specified source.

**Query Parameters**:
- `source`: Required - Source type ("shopify" or "google_drive")
- `namespace`: (Optional) Filter by namespace

**Response**:
```json
{
  "files": [
    {
      "id": "file_id_1",
      "title": "Document Title",
      "url": "https://example.com/doc1",
      "size": 12345,
      "source": "google_drive",
      "namespace": "default"
    },
    // More files...
  ],
  "total": 42
}
```

### 6.2 Usage Examples

#### Creating a Shopify Index

```python
import requests
import json

# API endpoint
url = "http://localhost:8005/api/index/"

# Request payload
payload = {
    "source": "shopify",
    "namespace": "client_store",
    "summarize": True,
    "parameters": {
        "store": "your-store.myshopify.com"
    }
}

# Send request
response = requests.post(url, json=payload)
result = response.json()

# Check the result
if result["status"] == "success":
    print(f"Successfully indexed {result['count']} items in namespace {result['namespace']}")
else:
    print(f"Indexing failed: {result['message']}")
```

#### Creating a Google Drive Index

```python
import requests
import json

# API endpoint
url = "http://localhost:8005/api/index/"

# Request payload
payload = {
    "source": "google_drive",
    "namespace": "marketing_docs",
    "summarize": True,
    "parameters": {
        "folder_id": "your_folder_id",
        "recursive": True
    }
}

# Send request
response = requests.post(url, json=payload)
result = response.json()

# Check the result
if result["status"] == "success":
    print(f"Google Drive content indexed successfully: {result['count']} documents in namespace {result['namespace']}")
else:
    print(f"Indexing failed: {result['message']}")
```

### 6.3 Frontend JavaScript Examples

#### API Service Integration

```javascript
// api.js - Unified API service for indexing operations

/**
 * Create a new index
 * @param {string} source - The source type ("shopify" or "google_drive")
 * @param {object} parameters - Source-specific parameters
 * @param {string} namespace - Optional namespace
 * @param {boolean} summarize - Whether to summarize content
 * @returns {Promise<object>} - Result of the operation
 */
export const createIndex = async (source, parameters = {}, namespace = "default", summarize = false) => {
  try {
    const response = await fetch('/api/index/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        source,
        namespace,
        summarize,
        parameters
      }),
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to create index');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error creating index:', error);
    throw error;
  }
};

/**
 * Get index information
 * @param {string} source - Optional source filter
 * @param {string} namespace - Optional namespace filter
 * @returns {Promise<object>} - Index information
 */
export const getIndexInfo = async (source = null, namespace = null) => {
  try {
    let url = '/api/index/';
    const params = new URLSearchParams();
    
    if (source) params.append('source', source);
    if (namespace) params.append('namespace', namespace);
    
    if (params.toString()) {
      url += `?${params.toString()}`;
    }
    
    const response = await fetch(url);
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to get index info');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error getting index info:', error);
    throw error;
  }
};

/**
 * Delete an index
 * @param {string} source - Optional source filter
 * @param {string} namespace - Optional namespace filter
 * @returns {Promise<object>} - Result of the operation
 */
export const deleteIndex = async (source = null, namespace = null) => {
  try {
    let url = '/api/index/';
    const params = new URLSearchParams();
    
    if (source) params.append('source', source);
    if (namespace) params.append('namespace', namespace);
    
    if (params.toString()) {
      url += `?${params.toString()}`;
    }
    
    const response = await fetch(url, {
      method: 'DELETE',
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to delete index');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error deleting index:', error);
    throw error;
  }
};

/**
 * Get files from an index
 * @param {string} source - The source type ("shopify" or "google_drive")
 * @param {string} namespace - Optional namespace filter
 * @returns {Promise<object>} - List of files
 */
export const getSourceFiles = async (source, namespace = null) => {
  try {
    let url = `/api/index/files?source=${source}`;
    
    if (namespace) {
      url += `&namespace=${namespace}`;
    }
    
    const response = await fetch(url);
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to get files');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error getting files:', error);
    throw error;
  }
};
```

#### React Hook Example

```javascript
// useGoogleDriveActions.js - Hook for managing Google Drive indexing
import { useState, useCallback } from 'react';
import { createIndex, deleteIndex, getIndexInfo, getSourceFiles } from '../../services/api';

export function useGoogleDriveActions() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState(null);
  const [files, setFiles] = useState([]);
  const [indexStats, setIndexStats] = useState(null);
  
  // Create Google Drive index
  const createGoogleDriveIndex = useCallback(async (
    folder_id, 
    recursive = true, 
    summarize = false,
    namespace = "default"
  ) => {
    setLoading(true);
    setError(null);
    
    try {
      const result = await createIndex(
        "google_drive", 
        { folder_id, recursive }, 
        namespace,
        summarize
      );
      
      setResults(result);
      
      // Refresh index stats and files after successful indexing
      await refreshIndexInfo(namespace);
      await loadIndexedFiles(namespace);
      
      return result;
    } catch (err) {
      setError(err.message || 'Failed to create Google Drive index');
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);
  
  // Delete Google Drive index
  const deleteGoogleDriveIndex = useCallback(async (namespace = null) => {
    setLoading(true);
    setError(null);
    
    try {
      const result = await deleteIndex("google_drive", namespace);
      setResults(result);
      
      // Clear files and refresh stats
      setFiles([]);
      await refreshIndexInfo();
      
      return result;
    } catch (err) {
      setError(err.message || 'Failed to delete Google Drive index');
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);
  
  // Load indexed files
  const loadIndexedFiles = useCallback(async (namespace = null) => {
    setLoading(true);
    setError(null);
    
    try {
      const data = await getSourceFiles("google_drive", namespace);
      setFiles(data.files || []);
      return data;
    } catch (err) {
      setError(err.message || 'Failed to load indexed files');
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);
  
  // Refresh index information
  const refreshIndexInfo = useCallback(async (namespace = null) => {
    try {
      const info = await getIndexInfo("google_drive", namespace);
      setIndexStats(info);
      return info;
    } catch (err) {
      setError(err.message || 'Failed to refresh index info');
      throw err;
    }
  }, []);
  
  return {
    loading,
    error,
    results,
    files,
    indexStats,
    createGoogleDriveIndex,
    deleteGoogleDriveIndex,
    loadIndexedFiles,
    refreshIndexInfo
  };
}
```

### 6.4 Complete Example with Feature Flags

```javascript
// Example of using the React hooks in a component

import React, { useState, useEffect } from 'react';
import { useGoogleDriveActions } from '../hooks/useGoogleDriveActions';
import { useShopifyActions } from '../hooks/useShopifyActions';

function IndexingPage() {
  const [namespace, setNamespace] = useState('default');
  const [folderId, setFolderId] = useState('');
  const [shopifyStore, setShopifyStore] = useState('');
  const [summarize, setSummarize] = useState(false);
  const [recursive, setRecursive] = useState(true);
  
  const {
    loading: gdLoading,
    error: gdError,
    results: gdResults,
    files: gdFiles,
    indexStats: gdStats,
    createGoogleDriveIndex,
    deleteGoogleDriveIndex,
    loadIndexedFiles: loadGdFiles,
    refreshIndexInfo: refreshGdInfo
  } = useGoogleDriveActions();
  
  const {
    loading: shopifyLoading,
    error: shopifyError,
    results: shopifyResults,
    createShopifyIndex,
    deleteShopifyIndex
  } = useShopifyActions();
  
  useEffect(() => {
    // Load initial data when component mounts
    refreshGdInfo();
    loadGdFiles();
  }, [refreshGdInfo, loadGdFiles]);
  
  const handleCreateGoogleDriveIndex = async () => {
    try {
      await createGoogleDriveIndex(folderId, recursive, summarize, namespace);
      alert('Google Drive indexing completed successfully!');
    } catch (error) {
      // Error already handled in the hook
    }
  };
  
  const handleCreateShopifyIndex = async () => {
    try {
      await createShopifyIndex(shopifyStore, summarize, namespace);
      alert('Shopify indexing completed successfully!');
    } catch (error) {
      // Error already handled in the hook
    }
  };
  
  const handleDeleteAllIndexes = async () => {
    if (window.confirm('Are you sure you want to delete all indexes?')) {
      try {
        // Delete both indexes from the namespace
        await deleteGoogleDriveIndex(namespace);
        await deleteShopifyIndex(namespace);
        alert('All indexes deleted successfully!');
      } catch (error) {
        alert(`Error deleting indexes: ${error.message}`);
      }
    }
  };
  
  return (
    <div>
      <h1>Content Indexing</h1>
      
      <div className="form-group">
        <label>Namespace:</label>
        <input 
          type="text" 
          value={namespace} 
          onChange={(e) => setNamespace(e.target.value)} 
          placeholder="default"
        />
      </div>
      
      <h2>Google Drive Indexing</h2>
      {gdLoading && <p>Loading...</p>}
      {gdError && <p className="error">Error: {gdError}</p>}
      
      <div className="form-group">
        <label>Folder ID:</label>
        <input 
          type="text" 
          value={folderId} 
          onChange={(e) => setFolderId(e.target.value)} 
          placeholder="Google Drive Folder ID"
        />
      </div>
      
      <div className="form-group">
        <label>
          <input 
            type="checkbox" 
            checked={recursive} 
            onChange={(e) => setRecursive(e.target.checked)} 
          />
          Process folders recursively
        </label>
      </div>
      
      <div className="form-group">
        <label>
          <input 
            type="checkbox" 
            checked={summarize} 
            onChange={(e) => setSummarize(e.target.checked)} 
          />
          Summarize content
        </label>
      </div>
      
      <button 
        onClick={handleCreateGoogleDriveIndex} 
        disabled={gdLoading || !folderId}
      >
        Index Google Drive Content
      </button>
      
      <h2>Shopify Indexing</h2>
      {shopifyLoading && <p>Loading...</p>}
      {shopifyError && <p className="error">Error: {shopifyError}</p>}
      
      <div className="form-group">
        <label>Store Domain:</label>
        <input 
          type="text" 
          value={shopifyStore} 
          onChange={(e) => setShopifyStore(e.target.value)} 
          placeholder="your-store.myshopify.com"
        />
      </div>
      
      <button 
        onClick={handleCreateShopifyIndex} 
        disabled={shopifyLoading || !shopifyStore}
      >
        Index Shopify Content
      </button>
      
      <hr />
      
      <button 
        onClick={handleDeleteAllIndexes} 
        className="danger"
      >
        Delete All Indexes
      </button>
      
      {gdStats && (
        <div className="stats">
          <h3>Index Statistics</h3>
          <p>Total Vectors: {gdStats.stats?.total_vector_count || 0}</p>
          <p>
            Vectors in namespace "{namespace}": 
            {gdStats.stats?.namespaces?.[namespace]?.vector_count || 0}
          </p>
        </div>
      )}
      
      {gdFiles.length > 0 && (
        <div className="files">
          <h3>Indexed Files ({gdFiles.length})</h3>
          <ul>
            {gdFiles.map(file => (
              <li key={file.id}>
                <a href={file.url} target="_blank" rel="noopener noreferrer">
                  {file.title}
                </a>
                <span className="file-size">({Math.round(file.size / 1024)} KB)</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default IndexingPage;
```

### 6.5 Axios API Client Example

```javascript
// indexApi.js - Axios-based API client for index operations
import axios from 'axios';

const apiClient = axios.create({
  baseURL: '/api',
  timeout: 50000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const indexApi = {
  /**
   * Create a new index
   */
  createIndex: async (source, parameters = {}, namespace = "default", summarize = false) => {
    try {
      const response = await apiClient.post('/index/', {
        source,
        namespace,
        summarize,
        parameters
      });
      return response.data;
    } catch (error) {
      console.error('Error creating index:', error);
      throw error.response?.data || error;
    }
  },
  
  /**
   * Get index information
   */
  getIndexInfo: async (source = null, namespace = null) => {
    try {
      const params = {};
      if (source) params.source = source;
      if (namespace) params.namespace = namespace;
      
      const response = await apiClient.get('/index/', { params });
      return response.data;
    } catch (error) {
      console.error('Error getting index info:', error);
      throw error.response?.data || error;
    }
  },
  
  /**
   * Delete an index
   */
  deleteIndex: async (source = null, namespace = null) => {
    try {
      const params = {};
      if (source) params.source = source;
      if (namespace) params.namespace = namespace;
      
      const response = await apiClient.delete('/index/', { params });
      return response.data;
    } catch (error) {
      console.error('Error deleting index:', error);
      throw error.response?.data || error;
    }
  },
  
  /**
   * Get files from a source
   */
  getSourceFiles: async (source, namespace = null) => {
    try {
      const params = { source };
      if (namespace) params.namespace = namespace;
      
      const response = await apiClient.get('/index/files', { params });
      return response.data;
    } catch (error) {
      console.error('Error getting source files:', error);
      throw error.response?.data || error;
    }
  }
};
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
  - Namespace-based content organization for multi-client or multi-project scenarios
- **Semantic Filtering**: Add semantic filters to improve response accuracy:
  - Filter by document type (presentation, article, product)
  - Filter by content domain (attribution, tracking, measurement)
  - Filter by technical complexity (beginner, intermediate, advanced)
  - Filter by recency and relevance
  - Filter by namespace for proper content isolation

### 9.4 API Design Best Practices
- **RESTful Interface Design**:
  - Use standardized endpoints with consistent resource naming
  - Follow HTTP method semantics (GET for reading, POST for creating, DELETE for removing)
  - Implement query parameters for filtering and sorting
  - Use request bodies for complex operations
  - Return appropriate HTTP status codes and descriptive error messages
- **Source Parameter Pattern**:
  - Use a unified `source` parameter approach instead of different endpoint paths
  - Standardize request and response formats across different sources
  - Ensure backward compatibility during API transitions
  - Document source-specific parameters clearly
- **Namespace Support**:
  - Implement namespaces for content organization and isolation
  - Use namespaces for multi-tenant or multi-client scenarios
  - Allow filtering operations by namespace
  - Provide default namespace for backward compatibility
- **Documentation**:
  - Use Pydantic models with field descriptions and examples
  - Provide comprehensive docstrings for all API endpoints
  - Include example requests and responses
  - Document all parameters and their effects

### 9.5 Performance and Scaling
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