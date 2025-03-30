#!/usr/bin/env python
"""
Shopify Indexer Module

This module provides functionality to fetch content from Shopify (products and blogs) 
and index it to a vector database (Pinecone) for RAG applications.
"""

import os
import json
import requests
import time
from typing import List, Dict, Any, Tuple

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from pinecone import Pinecone, ServerlessSpec
import markdownify as md

from app.config.chat_config import ChatConfig
from app.services.enhancement_service import enhancement_service
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
        self.shopify_admin_api_base = (f"https://{self.config.SHOPIFY_SHOP_DOMAIN}"
                                       f"/admin/api/{self.config.SHOPIFY_API_VERSION}")

        # Use the enhancement service
        self.enhancement_service = enhancement_service

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
            self.logger.info(
                f"Fetching products from {self.shopify_admin_api_base}/products.json with API key: {masked_key}")

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

    def get_all_content(self) -> List[Dict[str, Any]]:
        """
        Get all Shopify content (blogs, articles, products).

        Returns:
            List of all content records
        """
        try:
            self.logger.info("Fetching all Shopify content...")

            # Prepare blog articles
            self.logger.info("Fetching blog articles...")
            blog_records, article_records = self.prepare_blog_articles()

            # Prepare products
            self.logger.info("Fetching products...")
            product_records, variant_records = self.prepare_products()

            # Combine all records
            all_records = blog_records + article_records + product_records + variant_records
            self.logger.info(f"Fetched {len(all_records)} total records")
            
            # Save intermediate files if configured
            if self.config.SAVE_INTERMEDIATE_FILES:
                os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
                
                with open(os.path.join(self.config.OUTPUT_DIR, "blogs.json"), "w") as f:
                    blog_data = [r for r in all_records if r.get('type') == 'blog']
                    json.dump(blog_data, f, indent=2)
                
                with open(os.path.join(self.config.OUTPUT_DIR, "articles.json"), "w") as f:
                    article_data = [r for r in all_records if r.get('type') == 'article']
                    json.dump(article_data, f, indent=2)
                
                with open(os.path.join(self.config.OUTPUT_DIR, "products.json"), "w") as f:
                    product_data = [r for r in all_records if r.get('type') == 'product']
                    json.dump(product_data, f, indent=2)
            
            return all_records

        except Exception as e:
            self.logger.error(f"Error fetching Shopify content: {str(e)}")
            return []

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
        # Use the enhancement service to create an optimized embedding prompt
        return self.enhancement_service.create_embedding_prompt(text, metadata)

    def enrich_attribution_metadata(self, content: str) -> Dict[str, Any]:
        """
        Analyze content for attribution terminology and create enhanced metadata.

        Args:
            content: Markdown or text content to analyze

        Returns:
            Dictionary of attribution-related metadata
        """
        # Use the enhancement service to enrich attribution metadata
        return self.enhancement_service.enrich_attribution_metadata(content)

    def setup_shopify_indexer(self) -> dict:
        """
        Initialize the Shopify client and validate settings.
        This is a helper method for setup.
        
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

            self.logger.info(f"Initializing Shopify client for store: {shop_domain}")

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

            # Get content count for verification
            content = self.get_all_content()

            return {
                "status": "success",
                "message": f"Successfully initialized Shopify client with {len(content)} content items from {shop_domain}",
                "content_count": len(content)
            }

        except Exception as e:
            self.logger.error(f"Error in run_full_process: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
