import asyncio
from typing import Dict, Optional, Any

from app.config.chat_config import chat_config
from app.services.shopify_indexer import ShopifyIndexer
from app.services.gdrive_indexer import GoogleDriveIndexer
from app.services.content_processor import ContentProcessor
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

    async def create_shopify_index(self, store: Optional[str] = None, summarize: Optional[bool] = None) -> Dict[str, Any]:
        """
        Create a vector index from Shopify content.
        
        Args:
            store: Optional Shopify store domain
            summarize: Optional boolean to enable content summarization
            
        Returns:
            Dict containing status and message
        """
        try:
            # Update store domain if provided
            if store:
                self.shopify_indexer.config.SHOPIFY_SHOP_DOMAIN = store
                self.shopify_indexer.setup_shopify_indexer()
            
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
                return {
                    "status": "success",
                    "message": f"Successfully indexed {len(docs)} documents from Shopify"
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
                                summarize: Optional[bool] = None, enhanced_slides: Optional[bool] = None) -> Dict[str, Any]:
        """
        Create a vector index from Google Drive content.
        
        Args:
            folder_id: Optional Google Drive folder ID
            recursive: Optional boolean to enable recursive folder processing
            summarize: Optional boolean to enable content summarization
            enhanced_slides: Optional boolean to use enhanced slide processing
            
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
                return {
                    "status": "success",
                    "message": f"Successfully indexed {len(docs)} documents from Google Drive"
                }
            else:
                return {
                    "status": "error",
                    "message": "Failed to index documents from Google Drive"
                }

        except Exception as e:
            self.logger.error(f"Error creating index from Google Drive: {str(e)}", exc_info=True)
            return {"status": "error", "message": f"Failed to create index from Google Drive: {str(e)}"}

    async def get_index_info(self) -> Dict:
        """Get information about the current vector index"""
        try:
            # Get index info from vector store
            index_info = self.content_processor.get_index_info()
            return {
                "status": "success",
                "info": index_info
            }
        except Exception as e:
            self.logger.error(f"Error getting index info: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def get_google_drive_files(self) -> Dict[str, Any]:
        """Get list of indexed Google Drive files"""
        try:
            # Create Google Drive indexer
            indexer = GoogleDriveIndexer(self.gdrive_indexer.config)
            
            # Use the indexer's method to get file information
            return indexer.get_google_drive_files()
            
        except Exception as e:
            self.logger.error(f"Error retrieving Google Drive files: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def delete_index(self) -> Dict[str, Any]:
        """
        Delete the vector index.
        
        Returns:
            Dictionary containing status and results of the deletion
        """
        try:
            results = []

            for chat_model_config in chat_config.chat_model_configs.values():
                vector_store_config = chat_model_config.vector_store_config
                vector_store_client: VectorStoreClient = VectorStoreClient.get_vector_store_client(vector_store_config)
                results.append(vector_store_client.delete_index())

            # Check if all deletions were successful
            all_successful = all(result.get("status") == "success" for result in results)
            
            return {
                "status": "success" if all_successful else "error",
                "results": results,
                "message": "All indices deleted successfully" if all_successful else "Some indices failed to delete"
            }
        except Exception as e:
            self.logger.error(f"Error in delete_index: {str(e)}")
            return {"status": "error", "message": str(e)}
        
