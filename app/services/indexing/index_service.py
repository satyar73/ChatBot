import asyncio
from typing import Dict, Optional, Any

from app.config.chat_config import chat_config
from app.services.indexing.providers.shopify_indexer import ShopifyIndexer
from app.services.indexing.providers.gdrive_indexer import GoogleDriveIndexer
from app.services.indexing.content_processor import ContentProcessor
from app.utils.logging_utils import get_logger
from app.utils.vectorstore_client import VectorStoreClient

class IndexService:
    """Service for managing content indexing from various sources."""

    def __init__(self):
        """Initialize the index service."""
        self.logger = get_logger(__name__, "DEBUG")
        self.logger.debug("IndexService initialized")
        
        # Initialize indexers and processor
        self.shopify_indexer = ShopifyIndexer()
        self.gdrive_indexer = GoogleDriveIndexer()
        self.content_processor = ContentProcessor()

    async def create_shopify_index(self, store: Optional[str] = None, summarize: Optional[bool] = None, 
                            namespace: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a vector index from Shopify content.
        
        Args:
            store: Optional Shopify store domain
            summarize: Optional boolean to enable content summarization
            namespace: Optional namespace to use for this index
            
        Returns:
            Dict containing status and message
        """
        try:
            # Update store domain if provided
            if store:
                self.shopify_indexer.config.SHOPIFY_SHOP_DOMAIN = store
                self.shopify_indexer.setup_shopify_indexer()
            
            # Update namespace if provided
            if namespace:
                for config in self.content_processor.config.chat_model_configs.values():
                    if hasattr(config.vector_store_config, 'namespace'):
                        old_namespace = config.vector_store_config._namespace
                        config.vector_store_config._namespace = namespace
                        self.logger.info(f"Changed namespace from '{old_namespace}' to '{namespace}'")
            
            # Fetch content from Shopify
            self.logger.debug("Fetching content from Shopify")
            all_records = self.shopify_indexer.get_all_content()
            self.logger.info(f"Fetched {len(all_records)} records from Shopify")
            
            # Prepare documents for indexing
            self.logger.debug("Preparing documents for indexing")
            docs = self.content_processor.prepare_documents_for_indexing(all_records)
            self.logger.info(f"Prepared {len(docs)} documents for indexing")
            
            # Index the documents in vector store
            self.logger.debug("Indexing documents in vector store")
            success = self.content_processor.index_to_vector_store(docs)
            
            if success:
                namespace_info = ""
                for config in self.content_processor.config.chat_model_configs.values():
                    if hasattr(config.vector_store_config, 'namespace'):
                        namespace_info = f" in namespace '{config.vector_store_config.namespace}'"
                        break
                
                return {
                    "status": "success",
                    "message": f"Successfully indexed {len(docs)} documents from Shopify{namespace_info}"
                }
            else:
                return {
                    "status": "error",
                    "message": "Failed to index documents from Shopify"
                }

        except Exception as e:
            self.logger.error(f"Error creating index from Shopify: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def create_gdrive_index(self, folder_id: Optional[str] = None, recursive: Optional[bool] = None, 
                                summarize: Optional[bool] = None, enhanced_slides: Optional[bool] = None,
                                namespace: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a vector index from Google Drive content.
        
        Args:
            folder_id: Optional Google Drive folder ID
            recursive: Optional boolean to enable recursive folder processing
            summarize: Optional boolean to enable content summarization
            enhanced_slides: Optional boolean to use enhanced slide processing
            namespace: Optional namespace to use for this index
            
        Returns:
            Dict containing status and message
        """
        try:
            # Update Google Drive settings if provided
            if folder_id:
                self.gdrive_indexer.config.GOOGLE_DRIVE_FOLDER_ID = folder_id
            if recursive is not None:
                self.gdrive_indexer.config.GOOGLE_DRIVE_RECURSIVE = recursive
            if enhanced_slides is not None:
                self.gdrive_indexer.config.USE_ENHANCED_SLIDES = enhanced_slides
                
            # Update namespace if provided
            if namespace:
                for config in self.content_processor.config.chat_model_configs.values():
                    if hasattr(config.vector_store_config, 'namespace'):
                        old_namespace = config.vector_store_config._namespace
                        config.vector_store_config._namespace = namespace
                        self.logger.info(f"Changed namespace from '{old_namespace}' to '{namespace}'")
            
            # Use asyncio to run the document preparation in a separate thread
            records = await asyncio.to_thread(self.gdrive_indexer.prepare_drive_documents)
            
            if not records:
                return {
                    "status": "error",
                    "message": "No records found in Google Drive"
                }
            
            # Prepare documents for indexing
            self.logger.debug("Preparing documents for indexing")
            docs = self.content_processor.prepare_documents_for_indexing(records)
            self.logger.info(f"Prepared {len(docs)} documents for indexing")
            
            # Index the documents in vector store
            self.logger.debug("Indexing documents in vector store")
            success = self.content_processor.index_to_vector_store(docs)
            
            if success:
                namespace_info = ""
                for config in self.content_processor.config.chat_model_configs.values():
                    if hasattr(config.vector_store_config, 'namespace'):
                        namespace_info = f" in namespace '{config.vector_store_config.namespace}'"
                        break
                
                return {
                    "status": "success",
                    "message": f"Successfully indexed {len(docs)} documents from Google Drive{namespace_info}"
                }
            else:
                return {
                    "status": "error",
                    "message": "Failed to index documents from Google Drive"
                }

        except Exception as e:
            self.logger.error(f"Error creating index from Google Drive: {str(e)}", exc_info=True)
            return {"status": "error", "message": f"Failed to create index from Google Drive: {str(e)}"}

    async def get_index_info(self, source: Optional[str] = None, namespace: Optional[str] = None) -> Dict:
        """
        Get information about the current vector index.
        
        Args:
            source: Optional source type to filter results ('shopify' or 'google_drive')
            namespace: Optional namespace to filter results
            
        Returns:
            Dict containing status and index information
        """
        try:
            # Get index info from vector store
            index_info = self.content_processor.get_index_info(namespace=namespace)
            
            # Filter by source if specified
            if source and "info" in index_info:
                filtered_info = {}
                for key, value in index_info["info"].items():
                    if isinstance(value, dict) and "content" in value:
                        # Filter content by source
                        source_content = [item for item in value.get("content", []) 
                                         if item.get("source") == source]
                        
                        # Also filter by namespace if specified
                        if namespace:
                            source_content = [item for item in source_content
                                            if item.get("namespace") == namespace]
                                            
                        if source_content:
                            filtered_info[key] = value.copy()
                            filtered_info[key]["content"] = source_content
                
                index_info["info"] = filtered_info
            # Filter by namespace only if source not specified
            elif namespace and "info" in index_info:
                filtered_info = {}
                for key, value in index_info["info"].items():
                    if isinstance(value, dict) and "content" in value:
                        # Filter content by namespace
                        namespace_content = [item for item in value.get("content", []) 
                                           if item.get("namespace") == namespace]
                                            
                        if namespace_content:
                            filtered_info[key] = value.copy()
                            filtered_info[key]["content"] = namespace_content
                
                index_info["info"] = filtered_info
            
            # Add namespace information to the response
            if namespace:
                index_info["namespace"] = namespace
            
            return {
                "status": "success",
                "info": index_info
            }
        except Exception as e:
            self.logger.error(f"Error getting index info: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def get_google_drive_files(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """
        Get list of indexed Google Drive files.
        
        Args:
            namespace: Optional namespace to filter files
            
        Returns:
            Dictionary with file information
        """
        try:
            # Create Google Drive indexer
            indexer = GoogleDriveIndexer(self.gdrive_indexer.config)
            
            # Use the indexer's method to get file information with namespace
            return indexer.get_google_drive_files(namespace=namespace)
            
        except Exception as e:
            self.logger.error(f"Error retrieving Google Drive files: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def delete_index(self, source: Optional[str] = None, namespace: Optional[str] = None) -> Dict[str, Any]:
        """
        Delete the vector index.
        
        Args:
            source: Optional source type to filter what to delete ('shopify' or 'google_drive')
            namespace: Optional namespace to filter what to delete
            
        Returns:
            Dictionary containing status and results of the deletion
        """
        try:
            # Note: This is a simplified implementation. In a real system, you would need to 
            # implement source-specific deletion logic and namespace filtering.
            # For now, we're just deleting all indices regardless of source/namespace.
            
            results = []

            for chat_model_config in chat_config.chat_model_configs.values():
                vector_store_config = chat_model_config.vector_store_config
                
                # Apply namespace filter if provided
                if namespace and hasattr(vector_store_config, 'namespace') and vector_store_config.namespace != namespace:
                    # Skip configs that don't match the requested namespace
                    continue
                    
                vector_store_client: VectorStoreClient = VectorStoreClient.get_vector_store_client(vector_store_config)
                results.append(vector_store_client.delete_index())

            # Check if all deletions were successful
            all_successful = all(result.get("status") == "success" for result in results)
            
            # Construct a clear message based on filters applied
            if source and namespace:
                message_base = f"indices for source '{source}' in namespace '{namespace}'"
            elif source:
                message_base = f"indices for source '{source}'"
            elif namespace:
                message_base = f"indices in namespace '{namespace}'"
            else:
                message_base = "all indices"
                
            message = f"Successfully deleted {message_base}" if all_successful else f"Failed to delete some {message_base}"
            
            return {
                "status": "success" if all_successful else "error",
                "results": results,
                "message": message
            }
        except Exception as e:
            self.logger.error(f"Error in delete_index: {str(e)}")
            return {"status": "error", "message": str(e)}
        
