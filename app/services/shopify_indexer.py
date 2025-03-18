#!/usr/bin/env python
"""
Shopify Indexer Module

This module provides functionality to fetch content from Shopify (products and blogs) 
and index it to a vector database (Pinecone) for RAG applications.
"""

import os
import json
import logging
import requests
import time
from typing import List, Dict, Any, Tuple, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.docstore.document import Document
from pinecone import Pinecone, ServerlessSpec
import markdownify as md

from app.config.chat_config import ChatConfig
from app.utils.logging_utils import get_logger


class ShopifyIndexer:
    """
    Shopify content indexer for RAG applications.
    
    This class fetches content from a Shopify store (blogs and products) and
    indexes it to a Pinecone vector database for retrieval-augmented generation.
    """
    
    def __init__(self, config: ChatConfig = None):
        """
        Initialize the ShopifyIndexer with configuration.
        
        Args:
            config: Configuration object with Shopify and Pinecone parameters
        """
        self.config = config or ChatConfig()
        self.logger = get_logger(__name__)
        
        # API base URLs
        self.shopify_admin_api_base = f"https://{self.config.SHOPIFY_SHOP_DOMAIN}/admin/api/{self.config.SHOPIFY_API_VERSION}"
        
        # logging
        self.logger.info(f"ShopifyIndexer initialized with shop domain: {self.config.SHOPIFY_SHOP_DOMAIN}")
        
    def get_blogs(self) -> List[Dict[str, Any]]:
        """
        Get all blogs from Shopify store.
        
        Returns:
            List of blog objects containing id, handle, title, and updated_at
        """
        try:
            # Log the URL and API key (masked) being used
            masked_key = "***" + self.config.SHOPIFY_API_KEY[-4:] if self.config.SHOPIFY_API_KEY else "None"
            self.logger.info(f"Fetching blogs from {self.shopify_admin_api_base}/blogs.json with API key: {masked_key}")
            
            url = f"{self.shopify_admin_api_base}/blogs.json"
            params = {
                'fields': 'id,updated_at,handle,title',
                'limit': self.config.BLOG_FETCH_LIMIT
            }
            headers = {
                'X-Shopify-Access-Token': self.config.SHOPIFY_API_KEY
            }
            
            # Log the full request details (except API key)
            self.logger.info(f"Request: GET {url} with params={params}")
            
            response = requests.get(url=url, params=params, headers=headers)
            
            if response.status_code == 200:
                data = json.loads(response.content)
                blogs = data.get('blogs', [])
                self.logger.info(f"Retrieved {len(blogs)} blogs from Shopify")
                # Log first blog for debugging if any exist
                if blogs:
                    self.logger.info(f"First blog: {blogs[0]}")
                return blogs
            else:
                self.logger.error(f"Failed to get blogs: Status code {response.status_code}")
                self.logger.error(f"Response content: {response.content}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error retrieving blogs: {str(e)}")
            return []
    
    def get_articles(self, blog_id: int) -> List[Dict[str, Any]]:
        """
        Get all articles for a specific blog.
        
        Args:
            blog_id: The Shopify blog ID
            
        Returns:
            List of article objects
        """
        try:
            url = f"{self.shopify_admin_api_base}/blogs/{blog_id}/articles.json"
            params = {
                'status': 'active',
                'published_status': 'published',
                'fields': 'id,blog_id,updated_at,title,body_html,handle,author',
                'limit': self.config.ARTICLE_FETCH_LIMIT
            }
            headers = {
                'X-Shopify-Access-Token': self.config.SHOPIFY_API_KEY
            }
            
            response = requests.get(url=url, params=params, headers=headers)
            
            if response.status_code == 200:
                data = json.loads(response.content)
                articles = data.get('articles', [])
                self.logger.info(f"Retrieved {len(articles)} articles for blog ID {blog_id}")
                return articles
            else:
                self.logger.error(f"Failed to get articles: Status code {response.status_code}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error retrieving articles: {str(e)}")
            return []
    
    def get_products(self) -> List[Dict[str, Any]]:
        """
        Get all products from Shopify store.
        
        Returns:
            List of product objects
        """
        try:
            # Log the URL and API key (masked) being used
            masked_key = "***" + self.config.SHOPIFY_API_KEY[-4:] if self.config.SHOPIFY_API_KEY else "None"
            self.logger.info(f"Fetching products from {self.shopify_admin_api_base}/products.json with API key: {masked_key}")
            
            url = f"{self.shopify_admin_api_base}/products.json"
            params = {
                'status': 'active',
                'published_status': 'published',
                'fields': 'id,updated_at,title,body_html,handle',
                'presentment_currencies': 'USD',
                'limit': self.config.PRODUCT_FETCH_LIMIT
            }
            headers = {
                'X-Shopify-Access-Token': self.config.SHOPIFY_API_KEY
            }
            
            # Log the full request details (except API key)
            self.logger.info(f"Request: GET {url} with params={params}")
            
            response = requests.get(url=url, params=params, headers=headers)
            
            if response.status_code == 200:
                data = json.loads(response.content)
                products = data.get('products', [])
                self.logger.info(f"Retrieved {len(products)} products from Shopify")
                # Log first product for debugging if any exist
                if products:
                    self.logger.info(f"First product: {products[0]['title']} (ID: {products[0]['id']})")
                return products
            else:
                self.logger.error(f"Failed to get products: Status code {response.status_code}")
                self.logger.error(f"Response content: {response.content}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error retrieving products: {str(e)}")
            return []
    
    def html_to_markdown(self, html_content: str) -> str:
        """
        Convert HTML content to markdown for better chunking and indexing.
        
        Args:
            html_content: HTML content to convert
            
        Returns:
            Markdown string
        """
        try:
            # Convert HTML to markdown
            markdown_content = md.markdownify(html_content)

            # If configured, summarize long content
            if self.config.SUMMARIZE_CONTENT and len(markdown_content) > self.config.SUMMARIZE_THRESHOLD:
                # For this implementation, we're just returning as-is
                # You could add a summarization step here using OpenAI or another tool
                pass
                
            return markdown_content
            
        except Exception as e:
            self.logger.error(f"Error converting HTML to markdown: {str(e)}")
            # Return original content if conversion fails
            return html_content
    
    def prepare_blog_articles(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Prepare blog articles for indexing.
        
        Returns:
            Tuple of (blog_records, article_records)
        """
        blogs = self.get_blogs()
        all_blog_records = []
        all_article_records = []
        
        for blog in blogs:
            blog_id = blog.get('id')
            blog_handle = blog.get('handle')
            blog_title = blog.get('title')
            
            # Create blog record
            blog_url = f"{self.config.SHOPIFY_SITE_BASE_URL}/blogs/{blog_handle}"
            blog_record = {
                'title': blog_title,
                'url': blog_url,
                'type': 'blog',
                'markdown': f"Blog: {blog_title}"  # Add minimal markdown content for indexing
            }
            all_blog_records.append(blog_record)
            
            # Get articles for this blog
            articles = self.get_articles(blog_id)
            
            for article in articles:
                article_handle = article.get('handle')
                article_title = article.get('title')
                article_body_html = article.get('body_html', '')
                
                # Convert HTML to markdown
                article_markdown = self.html_to_markdown(article_body_html)
                
                # Create article record
                article_url = f"{self.config.SHOPIFY_SITE_BASE_URL}/blogs/{blog_handle}/{article_handle}"
                article_record = {
                    'title': article_title,
                    'url': article_url,
                    'markdown': article_markdown,
                    'type': 'article'
                }
                all_article_records.append(article_record)
        
        self.logger.info(f"Prepared {len(all_blog_records)} blogs and {len(all_article_records)} articles")
        return all_blog_records, all_article_records
    
    def prepare_products(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Prepare products for indexing.
        
        Returns:
            Tuple of (product_records, variant_records)
        """
        products = self.get_products()
        all_product_records = []
        all_variant_records = []
        
        for product in products:
            product_handle = product.get('handle')
            product_title = product.get('title')
            product_body_html = product.get('body_html', '')
            
            # Convert HTML to markdown
            product_markdown = self.html_to_markdown(product_body_html)
            
            # Create product record
            product_url = f"{self.config.SHOPIFY_SITE_BASE_URL}/products/{product_handle}"
            product_record = {
                'title': product_title,
                'url': product_url,
                'markdown': product_markdown,
                'type': 'product'
            }
            all_product_records.append(product_record)
            
            # Future: add variant records if needed
        
        self.logger.info(f"Prepared {len(all_product_records)} products")
        return all_product_records, all_variant_records

    def create_embedding_prompt(self, text: str, metadata: Dict[str, Any] = None) -> str:
        """
        Create an optimized prompt for embedding that highlights attribution terms.

        Args:
            text: Original text to embed
            metadata: Metadata associated with the text

        Returns:
            Enhanced prompt for embedding
        """
        metadata = metadata or {}

        # For tracking-specific Q&A
        if 'special_type' in metadata and metadata['special_type'] == 'tracking_types_examples':
            return f"""
            Context: Web and app tracking methods categorized as first-party and third-party tracking. 
            First-party tracking uses first-party cookies and internal systems.
            Third-party tracking uses third-party cookies and external platforms.

            {text}
            """

        # For attribution-specific texts, add context
        is_attribution_related = any(term in text.lower() for term in [
            "attribution", "incrementality", "MMM", "MTA", "CAC", "last click",
            "self-attribution", "self-attributed", "base attribution",
            "advanced attribution", "advanced attribution multiplier"
        ])

        if is_attribution_related:
            return f"Marketing attribution context: {text}"

        return text

    def index_to_pinecone(self, records: List[Dict[str, Any]]) -> bool:
        """
        Index content records to Pinecone vector database.
        
        Args:
            records: List of content records with title, url, and markdown
            
        Returns:
            True if indexing was successful, False otherwise
        """
        try:
            # If no records, return success
            if not records:
                self.logger.warning("No records to index")
                return True
                
            self.logger.info(f"Indexing {len(records)} records to Pinecone index '{self.config.PINECONE_INDEX_NAME}'")
            
            # Initialize Pinecone
            pc = Pinecone(api_key=self.config.PINECONE_API_KEY)
            
            # Check if index exists
            existing_indexes = pc.list_indexes().names()
            
            # Create index if it doesn't exist
            if self.config.PINECONE_INDEX_NAME not in existing_indexes:
                self.logger.info(f"Creating new Pinecone index: {self.config.PINECONE_INDEX_NAME}")
                
                pc.create_index(
                    name=self.config.PINECONE_INDEX_NAME,
                    dimension=self.config.PINECONE_DIMENSION,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud=self.config.PINECONE_CLOUD,
                        region=self.config.PINECONE_REGION
                    )
                )
                
                # Wait for index to initialize
                self.logger.info("Waiting for index to initialize...")
                time.sleep(10)
            else:
                self.logger.info(f"Using existing Pinecone index: {self.config.PINECONE_INDEX_NAME}")
            
            # Prepare documents
            docs = []
            for i, record in enumerate(records):
                # Check if record has markdown content
                if 'markdown' not in record:
                    self.logger.warning(f"Record {i} missing 'markdown' field: {record}")
                    continue  # Skip records without markdown
                
                # Split content into chunks
                if record.get('type') == 'qa_pair':
                    # For Q&A content, don't split questions from answers
                    chunks = [record['markdown']]
                else:
                    # Regular content uses recursive splitting
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=self.config.CHUNK_SIZE,
                        chunk_overlap=self.config.CHUNK_OVERLAP,
                        separators=["\n\n", "\n", " ", ""]
                    )
                    chunks = text_splitter.split_text(record['markdown'])

                # Create documents with metadata
                for j, chunk in enumerate(chunks):
                    # Get attribution metadata
                    attribution_metadata = self.enrich_attribution_metadata(chunk)

                    # Merge with standard metadata
                    metadata = {
                        "title": record['title'],
                        "url": record['url'],
                        "chunk": j,
                        "source": f"{record.get('type', 'content')}"
                    }
                    metadata.update(attribution_metadata)

                    doc = Document(
                        page_content=chunk,
                        metadata=metadata
                    )
                    docs.append(doc)

            # Initialize embeddings
            # text-embedding-ada-002 has fixed dimensions of 1536, don't specify dimensions
            embeddings = OpenAIEmbeddings(
                api_key=self.config.OPENAI_API_KEY,
                model=self.config.OPENAI_EMBEDDING_MODEL,
                embedding_ctx_length=self.config.EMBEDDING_CONTEXT_LENGTH,
                show_progress_bar = True
            )
            
            # Index documents
            self.logger.info(f"Indexing {len(docs)} document chunks to Pinecone...")

            # Create custom embeddings with enhanced prompts
            texts = []
            metadatas = []
            for doc in docs:
                texts.append(self.create_embedding_prompt(doc.page_content, doc.metadata))
                metadatas.append(doc.metadata)

            # Generate embeddings
            embeddings_array = embeddings.embed_documents(texts)

            # Get the Pinecone index
            index = pc.Index(self.config.PINECONE_INDEX_NAME)

            # Batch size for uploads
            batch_size = 100
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = embeddings_array[i:i + batch_size]
                batch_metadatas = metadatas[i:i + batch_size]

                # Create vector records
                vectors = []
                for j, (text, embedding, metadata) in enumerate(zip(batch_texts, batch_embeddings, batch_metadatas)):
                    vectors.append({
                        "id": f"doc_{i + j}",
                        "values": embedding,
                        "metadata": {**metadata, "text": text}
                    })

                # Upsert vectors to Pinecone
                index.upsert(vectors=vectors)

            self.logger.info(f"Successfully indexed {len(texts)} documents with custom embeddings")

            return True
            
        except Exception as e:
            self.logger.error(f"Error indexing to Pinecone: {str(e)}")
            return False

    def prepare_qa_pairs(self, qa_content: str) -> List[Dict[str, Any]]:
        """
        Process Q&A content to preserve question-answer relationships

        Args:
            qa_content: Raw Q&A content with questions and answers

        Returns:
            List of processed Q&A records
        """
        import re

        qa_records = []
        # Split into question-answer pairs
        qa_pairs = re.findall(r"(.*?\?)\s*(.*?)(?=\n\n|$)", qa_content, re.DOTALL)

        for question, answer in qa_pairs:
            question = question.strip()
            answer = answer.strip()

            if "tracking" in question.lower() and "web and app" in question.lower():
                # Add special metadata for tracking questions
                record = {
                    'title': f"Q&A: {question[:50]}...",
                    'url': '#tracking-types',
                    'markdown': f"Q: {question}\n\nA: {answer}",
                    'type': 'qa_pair',
                    'special_type': 'tracking_types_examples'
                }
                qa_records.append(record)
            else:
                record = {
                    'title': f"Q&A: {question[:50]}...",
                    'url': '#qa',
                    'markdown': f"Q: {question}\n\nA: {answer}",
                    'type': 'qa_pair'
                }
                qa_records.append(record)

        return qa_records

    def enrich_attribution_metadata(self, content: str) -> Dict[str, Any]:
        """
        Analyze content for attribution terminology and create enhanced metadata.

        Args:
            content: Markdown or text content to analyze

        Returns:
            Dictionary of attribution-related metadata
        """
        # Key attribution terms to identify
        attribution_terms = [
            "attribution", "incrementality", "MMM", "marketing mix modeling",
            "MTA", "multi-touch attribution", "CAC", "iCAC", "multiplier",
            "last click", "geo testing", "holdout test", "scale test",
            "self-attribution", "self-attributed", "base attribution",
            "advanced attribution", "advanced attribution multiplier"
        ]

        metadata = {}
        # Check for attribution terms
        for term in attribution_terms:
            if term.lower() in content.lower():
                metadata[f"has_{term.replace(' ', '_').lower()}"] = True

        return metadata
    
    def index_all_content(self) -> bool:
        """
        Index all Shopify content (blogs, articles, products) to Pinecone.
        
        Returns:
            True if indexing was successful, False otherwise
        """
        try:
            self.logger.info("Starting full content indexing...")
            
            # Prepare blog articles
            self.logger.info("Fetching blog articles...")
            blog_records, article_records = self.prepare_blog_articles()
            
            # Prepare products
            self.logger.info("Fetching products...")
            product_records, variant_records = self.prepare_products()
            
            # Process Q&A content from paste.txt if present
            qa_records = []
            if hasattr(self.config, 'QA_SOURCE_FILE') and self.config.QA_SOURCE_FILE:
                try:
                    with open(self.config.QA_SOURCE_FILE, 'r') as f:
                        qa_content = f.read()
                    qa_records = self.prepare_qa_pairs(qa_content)
                    self.logger.info(f"Processed {len(qa_records)} Q&A pairs")
                except Exception as e:
                    self.logger.error(f"Error processing Q&A content: {str(e)}")

            # Combine all records
            all_records = article_records + product_records + qa_records + blog_records

            # Save intermediate files if configured
            if self.config.SAVE_INTERMEDIATE_FILES:
                os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
                
                with open(os.path.join(self.config.OUTPUT_DIR, "blogs.json"), "w") as f:
                    json.dump(blog_records, f, indent=2)
                    
                with open(os.path.join(self.config.OUTPUT_DIR, "articles.json"), "w") as f:
                    json.dump(article_records, f, indent=2)
                    
                with open(os.path.join(self.config.OUTPUT_DIR, "products.json"), "w") as f:
                    json.dump(product_records, f, indent=2)
            
            # Index to Pinecone
            self.logger.info(f"Indexing {len(all_records)} total records...")
            result = self.index_to_pinecone(all_records)
            
            if result:
                self.logger.info("Full content indexing completed successfully")
            else:
                self.logger.error("Full content indexing failed")
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error during full content indexing: {str(e)}")
            return False
    
    def run_full_process(self) -> dict:
        """
        Run the complete Shopify indexing process.
        This is the main entry point called by IndexService.
        
        Returns:
            A dictionary with status and message
        """
        try:
            # Check for store in both possible config attributes
            shop_domain = getattr(self.config, 'SHOPIFY_SHOP_DOMAIN', None)
            shop_store = getattr(self.config, 'SHOPIFY_STORE', None)
            
            # Use SHOPIFY_SHOP_DOMAIN if provided, otherwise use SHOPIFY_STORE
            shop_domain = shop_domain if shop_domain else shop_store
            
            # Update SHOPIFY_SHOP_DOMAIN with the value from either attribute
            self.config.SHOPIFY_SHOP_DOMAIN = shop_domain
            
            self.logger.info(f"Starting Shopify indexing process for store: {shop_domain}")
            
            # Check if we have a valid Shopify domain
            if not shop_domain:
                self.logger.error("No Shopify domain provided")
                return {
                    "status": "error",
                    "message": "No Shopify domain provided. Please set SHOPIFY_SHOP_DOMAIN in config or provide a store parameter."
                }
                
            # Check if we have a valid Shopify API key
            if not self.config.SHOPIFY_API_KEY:
                self.logger.error("No Shopify API key provided")
                return {
                    "status": "error",
                    "message": "No Shopify API key provided. Please set SHOPIFY_API_KEY in config."
                }
                
            # Update the API base URL with the proper domain
            self.shopify_admin_api_base = f"https://{shop_domain}/admin/api/{self.config.SHOPIFY_API_VERSION}"
            self.logger.info(f"Updated API base URL: {self.shopify_admin_api_base}")
            
            # Set site base URL if not already set
            if not self.config.SHOPIFY_SITE_BASE_URL:
                self.config.SHOPIFY_SITE_BASE_URL = f"https://{shop_domain}"
                self.logger.info(f"Updated site base URL: {self.config.SHOPIFY_SITE_BASE_URL}")
            
            # Run the indexing process
            success = self.index_all_content()
            
            # Get blog and product counts for better messaging
            blog_records, article_records = self.prepare_blog_articles()
            product_records, _ = self.prepare_products()
            
            if success:
                return {
                    "status": "success",
                    "message": f"Successfully indexed {len(product_records)} products and {len(article_records)} articles from {shop_domain}"
                }
            else:
                return {
                    "status": "error",
                    "message": f"Failed to index content from {shop_domain}"
                }
                
        except Exception as e:
            self.logger.error(f"Error in run_full_process: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }