import os
import asyncio
from typing import Dict, Optional, Any
import logging
from pinecone import Pinecone

from app.config.chat_config import ChatConfig
from app.services.shopify_indexer import ShopifyIndexer
from app.services.gdrive_indexer import GoogleDriveIndexer

class IndexService:
    def __init__(self):
        self.config = ChatConfig()
        self.logger = logging.getLogger(__name__)

    async def create_index(self, store: Optional[str] = None, summarize: Optional[bool] = None) -> Dict:
        """Create and populate a new vector index"""
        try:
            # Initialize the indexer
            indexer = ShopifyIndexer()

            # Override configuration if parameters are provided
            if summarize is not None:
                indexer.config.SUMMARIZE_CONTENT = summarize

            if store:
                indexer.config.SHOPIFY_STORE = store

            # Run the indexing process
            result = indexer.run_full_process()

            # Ensure the result is serializable
            if isinstance(result, dict):
                # Return a new dict with only serializable items
                return {
                    "status": result.get("status", "success"),
                    "message": result.get("message", "Indexing completed successfully"),
                    # Add any other important keys from result that are serializable
                }
            else:
                # If result isn't a dict, create a serializable response
                return {
                    "status": "success",
                    "message": "Indexing completed successfully"
                }

            return result

        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def create_index_from_drive(
            self,
            folder_id: Optional[str] = None,
            recursive: Optional[bool] = True,
            summarize: Optional[bool] = None
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

            # Set environment variable to use Google Drive
            os.environ["USE_GOOGLE_DRIVE"] = "true"

            # Create indexer and run process in a separate thread to avoid blocking
            indexer = GoogleDriveIndexer(self.config)
            result = await asyncio.to_thread(indexer.run_full_process)
            return result

        except Exception as e:
            self.logger.error(f"Error creating index from Google Drive: {str(e)}")
            return {"status": "error", "message": f"Failed to create index from Google Drive: {str(e)}"}

    async def get_index_info(self) -> Dict:
        """Get information about the current vector index"""
        try:
            pc = Pinecone(api_key=self.config.PINECONE_API_KEY)

            # Check if index exists
            available_indexes = pc.list_indexes().names()

            if self.config.PINECONE_INDEX_NAME in available_indexes:
                # Get index stats
                index = pc.Index(self.config.PINECONE_INDEX_NAME)
                stats_raw = index.describe_index_stats()

                # Convert any non-serializable objects to strings or simple types
                stats = {
                    "dimension": stats_raw.dimension,
                    "index_fullness": stats_raw.index_fullness,
                    "namespaces": {k: {"vector_count": v.vector_count} for k, v in stats_raw.namespaces.items()},
                    "total_vector_count": stats_raw.total_vector_count
                }

                return {
                    "status": "success",
                    "exists": True,
                    "name": self.config.PINECONE_INDEX_NAME,
                    "stats": stats
                }
            else:
                return {
                    "status": "success",
                    "exists": False,
                    "name": self.config.PINECONE_INDEX_NAME
                }

        except Exception as e:
            print(f"Error in get_index_info: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def delete_index(self) -> Dict:
        """Delete the current vector index"""
        try:
            pc = Pinecone(api_key=self.config.PINECONE_API_KEY)

            # Check if index exists
            available_indexes = pc.list_indexes().names()

            if self.config.PINECONE_INDEX_NAME in available_indexes:
                # Delete the index
                pc.delete_index(self.config.PINECONE_INDEX_NAME)
                return {"status": "success", "message": f"Index '{self.config.PINECONE_INDEX_NAME}' deleted successfully"}
            else:
                return {"status": "success", "message": f"Index '{self.config.PINECONE_INDEX_NAME}' does not exist"}

        except Exception as e:
            return {"status": "error", "message": str(e)}
