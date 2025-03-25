import os
import json
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

            # Create a serializable response by default
            return {
                    "status": "success",
                    "message": "Indexing completed successfully"
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def create_index_from_drive(
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
                
                # Try to load the Shopify content from saved files
                content = []
                try:
                    # Check if product and article files exist
                    product_path = os.path.join(self.config.OUTPUT_DIR, "products.json")
                    article_path = os.path.join(self.config.OUTPUT_DIR, "articles.json")
                    
                    if os.path.exists(product_path):
                        with open(product_path, "r") as f:
                            products = json.load(f)
                            content.extend(products)
                    
                    if os.path.exists(article_path):
                        with open(article_path, "r") as f:
                            articles = json.load(f)
                            content.extend(articles)
                except Exception as e:
                    self.logger.warning(f"Could not load content files: {str(e)}")
                
                return {
                    "status": "success",
                    "exists": True,
                    "name": self.config.PINECONE_INDEX_NAME,
                    "stats": stats,
                    "content": content
                }
            else:
                return {
                    "status": "success",
                    "exists": False,
                    "name": self.config.PINECONE_INDEX_NAME,
                    "content": []
                }

        except Exception as e:
            print(f"Error in get_index_info: {str(e)}")
            return {"status": "error", "message": str(e)}
            
    async def get_google_drive_files(self) -> Dict:
        """Get list of indexed Google Drive files"""
        try:
            # Try to load the Google Drive processed files
            drive_path = os.path.join(self.config.OUTPUT_DIR, "drive_processed.json")
            
            if os.path.exists(drive_path):
                with open(drive_path, "r") as f:
                    files = json.load(f)
                    
                # Extract basic file information
                file_list = [
                    {
                        "id": idx,
                        "title": file.get("title", "Unknown"),
                        "url": file.get("url", ""),
                        "size": len(file.get("markdown", "")) if "markdown" in file else 0
                    }
                    for idx, file in enumerate(files)
                ]
                
                return {
                    "status": "success",
                    "files": file_list,
                    "count": len(file_list)
                }
            else:
                # Check if we can query the vector store directly
                try:
                    pc = Pinecone(api_key=self.config.PINECONE_API_KEY)
                    
                    if self.config.PINECONE_INDEX_NAME in pc.list_indexes().names():
                        index = pc.Index(self.config.PINECONE_INDEX_NAME)
                        stats = index.describe_index_stats()
                        
                        return {
                            "status": "success",
                            "files": [],
                            "count": 0,
                            "vector_count": stats.total_vector_count,
                            "message": "Drive file list not available, but vectors are in the index"
                        }
                except Exception as e:
                    self.logger.error(f"Error querying Pinecone for Google Drive files: {str(e)}")
                
                return {
                    "status": "success",
                    "files": [],
                    "count": 0,
                    "message": "No Google Drive files indexed or file list not available"
                }
                
        except Exception as e:
            self.logger.error(f"Error retrieving Google Drive files: {str(e)}")
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
