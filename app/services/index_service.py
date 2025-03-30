import asyncio
import os
import json
from typing import Dict, Optional, Any
from pinecone import Pinecone

from app.config.chat_config import ChatConfig
from app.services.shopify_indexer import ShopifyIndexer
from app.services.gdrive_indexer import GoogleDriveIndexer
from app.services.content_processor import ContentProcessor
from app.utils.logging_utils import get_logger

class IndexService:
    def __init__(self):
        self.config = ChatConfig()
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
            
            # Prepare documents for indexing
            self.logger.debug("Preparing documents for indexing")
            docs = self.content_processor.prepare_documents_for_indexing(all_records)
            self.logger.info(f"Prepared {len(docs)} documents for indexing")
            
            # Index the documents to Pinecone
            self.logger.debug("Indexing documents to Pinecone")
            success = self.content_processor.index_to_pinecone(docs)
            
            if success:
                return {
                    "status": "success",
                    "message": f"Successfully indexed {len(docs)} document chunks",
                    "chunk_count": len(docs)
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
            
            if not records:
                return {
                    "status": "error",
                    "message": "No records found in Google Drive"
                }
            
            # Prepare documents for indexing
            self.logger.debug("Preparing documents for indexing")
            docs = self.content_processor.prepare_documents_for_indexing(records)
            self.logger.info(f"Prepared {len(docs)} documents for indexing")
            
            # Index the documents to Pinecone
            self.logger.debug("Indexing documents to Pinecone")
            success = self.content_processor.index_to_pinecone(docs)
            
            if success:
                return {
                    "status": "success",
                    "message": f"Successfully indexed {len(docs)} document chunks",
                    "chunk_count": len(docs)
                }
            else:
                return {
                    "status": "error",
                    "message": "Failed to index content to Pinecone"
                }

        except Exception as e:
            self.logger.error(f"Error creating index from Google Drive: {str(e)}")
            return {"status": "error", "message": str(e)}

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
