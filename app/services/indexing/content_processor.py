"""
Content processor service for document processing and vector store indexing.
"""
import os
import copy
from typing import List, Dict, Any, Optional

from app.utils.logging_utils import get_logger
from app.config.chat_config import ChatConfig, chat_config
from app.services.common.enhancement_service import enhancement_service
from app.utils.text_splitters import TokenTextSplitter
from app.utils.vectorstore_client import VectorStoreClient
from langchain.schema import Document


class ContentProcessor:
    """
    Base class for processing and indexing document content to vector stores.
    Provides common indexing functionality for different content sources.
    """

    def __init__(self, config: Optional[ChatConfig] = None):
        """
        Initialize the content processor with configuration.
        
        Args:
            config: Configuration object with vector store parameters
        """
        self.config = config or ChatConfig()
        self.logger = get_logger(__name__)
        self.logger.info("ContentProcessor initialized")
        self.enhancement_service = enhancement_service
        
        # Create output directory if needed
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)

    def get_index_info(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about the current vector index.
        
        Args:
            namespace: Optional namespace to filter results
            
        Returns:
            Dictionary containing index information
        """
        try:
            # Initialize vector store clients from all configurations
            all_info = {}
            total_documents = 0
            total_chunks = 0
            namespaces = set()
            
            # Process each vector store config
            for name, chat_model_config in self.config.chat_model_configs.items():
                vector_store_config = chat_model_config.vector_store_config
                
                # Skip if namespace doesn't match (when filter is specified)
                if namespace and hasattr(vector_store_config, 'namespace'):
                    if vector_store_config.namespace != namespace:
                        continue
                
                # Get client and info
                vector_store = VectorStoreClient.get_vector_store_client(vector_store_config)
                
                # If client supports namespace parameter, use it
                if hasattr(vector_store, 'get_index_info_by_namespace'):
                    # Get info filtered by namespace
                    namespace_to_check = namespace or vector_store_config.namespace
                    index_info = vector_store.get_index_info_by_namespace(namespace_to_check)
                else:
                    # Just get general info
                    index_info = vector_store.get_index_info()
                
                # Extract general stats
                total_documents += index_info.get("total_documents", 0)
                total_chunks += index_info.get("total_chunks", 0)
                
                # Get namespaces (if available)
                if "namespaces" in index_info:
                    for ns in index_info["namespaces"]:
                        namespaces.add(ns)
                        
                # Store individual config info
                all_info[name] = index_info
                
                # Store namespace info
                if namespace and "namespace" not in index_info:
                    index_info["namespace"] = namespace
            
            # Build consolidated result
            result = {
                "total_documents": total_documents,
                "total_chunks": total_chunks,
                "index_name": next(iter(all_info.values()), {}).get("index_name", ""),
                "last_updated": next(iter(all_info.values()), {}).get("last_updated", ""),
                "dimension": next(iter(all_info.values()), {}).get("dimension", 0),
                "namespaces": list(namespaces),
                "configs": all_info
            }
            
            # Add namespace to result if specified
            if namespace:
                result["namespace"] = namespace
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting index info: {str(e)}")
            return {
                "total_documents": 0,
                "total_chunks": 0,
                "index_name": "",
                "last_updated": "",
                "dimension": 0,
                "namespaces": [],
                "error": str(e)
            }

    def save_chunks_to_file(self, all_chunks: List[Dict[str, Any]], filename: str = "document_chunks.json") -> str:
        """
        Save document chunks to a JSON file for analysis.
        
        Args:
            all_chunks: List of document chunks with metadata
            filename: Name of the output file
            
        Returns:
            Path to the saved file
        """
        import json
        import datetime
        
        # Add timestamp to filename to avoid overwriting
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_with_timestamp = f"{os.path.splitext(filename)[0]}_{timestamp}{os.path.splitext(filename)[1]}"
        
        output_path = os.path.join(self.config.OUTPUT_DIR, filename_with_timestamp)
        
        with open(output_path, 'w') as f:
            json.dump(all_chunks, f, indent=2)
            
        self.logger.info(f"Saved {len(all_chunks)} chunks to {output_path}")
        return output_path

    def prepare_documents_for_indexing(self, records: List[Dict[str, Any]]) \
                                                -> List[Document]:
        """
        Prepare documents for indexing by splitting content into chunks and adding metadata.
        
        Args:
            records: List of content records with title, url, and markdown
            
        Returns:
            List of Document objects ready for indexing
        """
        docs = []

        # Create a list to store all chunks for debugging/logging
        all_chunks = []
        
        for i, record in enumerate(records):
            # Check if record has markdown content
            if 'markdown' not in record:
                self.logger.warning(f"Record {i} missing 'markdown' field: {record}")
                continue  # Skip records without markdown

            # Split content into chunks
            is_qa_pair = record.get('type') == 'qa_pair'
            is_slide = record.get('document_type') == 'presentation_slide'
            
            # Don't split QA pairs or presentation slides
            if is_qa_pair or is_slide:
                # For Q&A content and slide content, don't split the content
                chunks = [record['markdown']]
                
                if is_qa_pair:
                    self.logger.debug(f"QA Pair not split: {record['title']}")
                    special_handling = "QA pair preserved as single chunk"
                    doc_type = "qa_pair"
                else:
                    self.logger.debug(f"Presentation slide not split: {record['title']}")
                    special_handling = "Slide preserved as single chunk"
                    doc_type = record.get('type')  # Preserve original type (e.g., "client")
                
                # Create a temporary text splitter just for token counting
                temp_splitter = TokenTextSplitter(
                    chunk_size=self.config.CHUNK_SIZE,
                    chunk_overlap=self.config.CHUNK_OVERLAP,
                    model_name=self.config.OPENAI_EMBEDDING_MODEL
                )
                
                # Add record to chunks list with metadata
                chunk_data = {
                    "doc_title": record['title'],
                    "doc_url": record.get('url', 'No URL'),
                    "doc_source": record.get('source', 'Unknown'),
                    "doc_type": doc_type,
                    "chunk_index": 0,
                    "chunk_content": record['markdown'],
                    "chunk_token_count": temp_splitter.count_tokens(record['markdown']),
                    "special_handling": special_handling
                }
                
                # Add slide-specific metadata if available
                if record.get('document_type') == 'presentation_slide':
                    chunk_data["document_type"] = "presentation_slide"
                    chunk_data["parent_presentation"] = record.get('parent_presentation', '')
                    chunk_data["slide_number"] = record.get('slide_number', 0)
                
                all_chunks.append(chunk_data)
            else:
                # Define special technical terms to preserve
                special_terms = [
                    "advanced attribution multiplier",
                    "attribution multiplier",
                    "marketing mix modeling",
                    # Add other multi-word technical terms
                ]

                # For technical content, use smaller chunks with more overlap
                is_technical = any(term in record['markdown'].lower() for term in special_terms)
                if is_technical:
                    self.logger.debug(f"Using smaller chunks for technical content: {record['title']}")
                    text_splitter = TokenTextSplitter(
                        chunk_size=self.config.CHUNK_SIZE // 2,  # Smaller chunks for technical content
                        chunk_overlap=self.config.CHUNK_OVERLAP * 2,  # More overlap
                        model_name=self.config.OPENAI_EMBEDDING_MODEL
                    )
                else:
                    text_splitter = TokenTextSplitter(
                        chunk_size=self.config.CHUNK_SIZE,
                        chunk_overlap=self.config.CHUNK_OVERLAP,
                        model_name=self.config.OPENAI_EMBEDDING_MODEL
                    )
                chunks = text_splitter.split_text(record['markdown'])
                
                # Log chunking information
                self.logger.debug(f"Split document '{record['title']}' into {len(chunks)} chunks")
                
                # Save chunks with document info for debugging
                for j, chunk in enumerate(chunks):
                    chunk_data = {
                        "doc_title": record['title'],
                        "doc_url": record.get('url', 'No URL'),
                        "doc_source": record.get('source', 'Unknown'),
                        "doc_type": record.get('type', 'Unknown'),
                        "chunk_index": j,
                        "chunk_content": chunk,
                        "chunk_token_count": text_splitter.count_tokens(chunk),
                        "total_chunks": len(chunks)
                    }
                    
                    # Add technical content flag if applicable
                    if is_technical:
                        chunk_data["special_handling"] = "Technical content with smaller chunks"
                        
                    all_chunks.append(chunk_data)

            # Create documents with metadata
            for j, chunk in enumerate(chunks):
                # Get attribution metadata
                attribution_metadata = self.enhancement_service.enrich_attribution_metadata(chunk)

                # Enhance chunk with keywords
                chunk_keywords = self.enhancement_service.enhance_chunk_with_keywords(chunk)

                # Merge with standard metadata
                metadata = {
                    "title": record['title'],
                    "url": record['url'],
                    "chunk": j,
                    "source": f"{record.get('type', 'content')}"
                }
                metadata.update(attribution_metadata)
                # Add chunk-specific keywords
                if chunk_keywords:
                    metadata["keywords"] = chunk_keywords

                if record.get('source') == 'Google Drive':
                    metadata["type"] = record.get('type')
                    metadata["client"] = record.get('client')
                    
                    # Add slide-specific metadata if this is a presentation slide
                    if record.get('document_type') == 'presentation_slide':
                        metadata["document_type"] = "presentation_slide"
                        metadata["parent_presentation"] = record.get('parent_presentation', '')
                        metadata["slide_number"] = record.get('slide_number', 0)
                        
                    # Add namespace information if present in the record
                    if 'namespace' in record:
                        metadata["namespace"] = record.get('namespace')

                # Create embedding prompt. This is one that is used for semantic search
                optimized_text = self.enhancement_service.create_embedding_prompt(chunk, metadata)
                
                # Add metadata to the corresponding chunk data for logging
                for chunk_data in all_chunks:
                    if (chunk_data.get("doc_title") == record['title'] and 
                        chunk_data.get("chunk_index") == j):
                        # Add the full metadata to the chunk data
                        chunk_data["metadata"] = metadata.copy()
                        # Add the optimized text used for embedding
                        chunk_data["embedding_text"] = optimized_text
                        break

                doc = Document(
                    page_content=optimized_text,
                    metadata=metadata
                )
                docs.append(doc)
        
        # Save all chunks to file for analysis
        if all_chunks:
            self.save_chunks_to_file(all_chunks)

        self.logger.info(f"Returned {len(all_chunks)} chunks for indexing.")
        return docs

    def index_to_vector_store(self, docs: List[Document]) -> bool:
        """
        Go through each configured vector store (e.g. Pinecone, Neon, etc.) and index the documents.
        If documents have namespace information, they will be indexed in their specific namespaces.
        """
        # First, check if we have namespace information in the documents
        has_namespace_info = any("namespace" in doc.metadata for doc in docs)
        
        if has_namespace_info:
            self.logger.debug("Documents contain namespace information - using namespace-aware indexing")
            
            # Group documents by namespace
            namespace_groups = {}
            for doc in docs:
                namespace = doc.metadata.get("namespace", "default")
                if namespace not in namespace_groups:
                    namespace_groups[namespace] = []
                namespace_groups[namespace].append(doc)
                
            # Log namespace distribution
            for namespace, docs_list in namespace_groups.items():
                self.logger.debug(f"Namespace '{namespace}' has {len(docs_list)} documents to index")
                
            # Index each namespace group separately
            success = True
            for namespace, docs_list in namespace_groups.items():
                self.logger.debug(f"Indexing {len(docs_list)} documents in namespace '{namespace}'...")
                
                for chat_model_config in chat_config.chat_model_configs.values():
                    # Create a copy of the config to modify the namespace
                    config_copy = copy.deepcopy(chat_model_config)
                    
                    # Update namespace in the config copy
                    if hasattr(config_copy.vector_store_config, 'namespace'):
                        original_ns = config_copy.vector_store_config.namespace
                        config_copy.vector_store_config._namespace = namespace
                        self.logger.debug(f"Changed namespace from '{original_ns}' "
                                          f"to '{namespace}' for indexing")
                    
                    # Get vector store client and index documents
                    vector_store_client = VectorStoreClient.get_vector_store_client(config_copy.vector_store_config)
                    success &= vector_store_client.index_to_vector_store(config_copy, docs_list)

                    self.logger.info(f"Finished Indexing {len(docs_list)} "
                                     f"docs for namespace '{namespace}'.")
            return success
        else:
            # Legacy mode - no namespace information in documents
            self.logger.debug("No namespace information in documents - using standard indexing")
            success = True
            for chat_model_config in chat_config.chat_model_configs.values():
                vector_store_config = chat_model_config.vector_store_config
                vector_store_client = VectorStoreClient.get_vector_store_client(vector_store_config)
                success &= vector_store_client.index_to_vector_store(chat_model_config, docs)

            self.logger.info(f"No namespace. Finished Indexing {len(docs)} docs.")
            return success
