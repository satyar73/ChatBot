import asyncio
import os
from typing import Dict, Optional, Any

from app.config.chat_config import chat_config
from app.services.shopify_indexer import ShopifyIndexer
from app.services.gdrive_indexer import GoogleDriveIndexer
from app.services.content_processor import ContentProcessor
from app.utils.logging_utils import get_logger
from app.utils.vectorstore_client import VectorStoreClient

class IndexService:
    def __init__(self):
        self.config = chat_config
        self.logger = get_logger(__name__, "DEBUG")
        self.content_processor = ContentProcessor(self.config)

    async def create_index_from_shopify_store(self,
                                              store: Optional[str] = None,
                                              summarize: Optional[bool] = None) -> Dict:
        """Create and populate a new vector index from Shopify content"""
        try:
            # Initialize the Shopify indexer for content reading only
            indexer = ShopifyIndexer()

            # Override configuration if parameters are provided
            if summarize is not None:
                indexer.config.SUMMARIZE_CONTENT = summarize

            if store:
                indexer.config.SHOPIFY_STORE = store

            # Initialize Shopify client and parameters
            setup = indexer.setup_shopify_indexer()
            if setup.get("status") != "success":
                return setup
                
            # Fetch all content from Shopify
            self.logger.debug("Fetching content from Shopify")
            all_records = indexer.get_all_content()
            self.logger.info(f"Fetched {len(all_records)} records from Shopify")
            
            # Process and enhance records
            self.logger.debug("Processing and enhancing records")
            enhanced_records = self.content_processor.process_records(all_records)
            
            # File saving is handled in the Shopify indexer
            
            # Index the enhanced records in vector store
            self.logger.debug("Indexing records in vector store")
            success = self.content_processor.index_to_vector_store(enhanced_records)
            
            if success:
                return {
                    "status": "success",
                    "message": f"Successfully indexed {len(enhanced_records)} Shopify records",
                    "record_count": len(enhanced_records)
                }
            else:
                return {
                    "status": "error",
                    "message": "Failed to index content to Pinecone"
                }

        except Exception as e:
            self.logger.error(f"Error creating index from Shopify: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def create_index_from_google_drive(
            self,
            folder_id: Optional[str] = None,
            recursive: Optional[bool] = True,
            summarize: Optional[bool] = None,
            enhanced_slides: Optional[bool] = None
    ) -> Dict[str, Any]:
        """Create and populate a vector index with Google Drive data"""
        try:
            # Update config if parameters provided
            if folder_id:
                self.config.update_setting("GOOGLE_DRIVE_FOLDER_ID", folder_id)

            if recursive is not None:
                self.config.update_setting("GOOGLE_DRIVE_RECURSIVE", recursive)

            if summarize is not None:
                self.config.update_setting("SUMMARIZE_CONTENT", summarize)
                
            if enhanced_slides is not None:
                self.config.update_setting("USE_ENHANCED_SLIDES", enhanced_slides)

            # Set environment variable to use Google Drive
            os.environ["USE_GOOGLE_DRIVE"] = "true"

            # Create indexer for content reading
            indexer = GoogleDriveIndexer(self.config)
            
            # Use asyncio to run the document preparation in a separate thread
            records = await asyncio.to_thread(indexer.prepare_drive_documents)
            self.logger.info(f"Fetched {len(records)} records from Google Drive")
            
            # File saving is handled in the Google Drive indexer
                    
            # Process and enhance records
            self.logger.debug("Processing and enhancing records")
            enhanced_records = self.content_processor.process_records(records)
            
            # Index the enhanced records in vector store
            self.logger.debug("Indexing records to vector store")
            success = self.content_processor.index_to_vector_store(enhanced_records)
            
            if success:
                return {
                    "status": "success",
                    "message": f"Successfully indexed {len(enhanced_records)} Google Drive records",
                    "files_processed": len(enhanced_records)
                }
            else:
                return {
                    "status": "error",
                    "message": "Failed to index content to Pinecone"
                }

        except Exception as e:
            self.logger.error(f"Error creating index from Google Drive: {str(e)}", e, exc_info=True)
            return {"status": "error", "message": f"Failed to create index from Google Drive: {str(e)}"}

    async def get_index_info(self) -> Dict:
        """Get information about the current vector index"""
        try:
            ret_val = []
            for chat_model_config in chat_config.chat_model_configs.values():
                vector_store_config = chat_model_config.vector_store_config
                vector_store_client: VectorStoreClient = VectorStoreClient.get_vector_store_client(vector_store_config)
                ret_val.append(vector_store_client.get_index_info())
        except Exception as e:
            print(f"Error in get_index_info: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def get_google_drive_files(self) -> Dict[str, Any]:
        """Get list of indexed Google Drive files"""
        try:
            # Create Google Drive indexer
            indexer = GoogleDriveIndexer(self.config)
            
            # Use the indexer's method to get file information
            return indexer.get_google_drive_files()
            
        except Exception as e:
            self.logger.error(f"Error retrieving Google Drive files: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def delete_index(self) -> Dict:
        """Delete the current vector index"""
        try:
            ret_val = []

            for chat_model_config in chat_config.chat_model_configs.values():
                vector_store_config = chat_model_config.vector_store_config
                vector_store_client: VectorStoreClient = VectorStoreClient.get_vector_store_client(vector_store_config)
                ret_val.append(vector_store_client.delete_index())

            return ret_val
        except Exception as e:
            print(f"Error in delete_index: {str(e)}")
            return {"status": "error", "message": str(e)}
        
