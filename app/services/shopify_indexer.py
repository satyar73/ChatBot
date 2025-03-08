import os
import requests
import json
from tqdm import tqdm
from markdownify import markdownify as md
from typing import Callable, Dict, List, Union, Optional
from pathlib import Path
import logging
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

from app.config.chat_config import ChatConfig

class CustomJsonLoader(BaseLoader):
    """Custom loader for JSON data"""
    def __init__(
        self, data: Union[Dict, List, Path], dataset_mapping_function: Callable[[Dict], Document]
    ):
        if isinstance(data, (dict, list)):
            self.data = data
        elif isinstance(data, (Path, str)):
            path = Path(data).resolve()
            with open(path, "r", encoding='utf-8') as f:
                self.data = json.load(f)
        self.dataset_mapping_function = dataset_mapping_function

    def load(self) -> List[Document]:
        """Load documents."""
        return list(map(self.dataset_mapping_function, self.data))

class ShopifyIndexer:
    """Class to handle the process of fetching Shopify data and indexing to Pinecone"""

    def __init__(self, config: Optional[ChatConfig] = None):
        """Initialize the indexer with configuration"""
        self.config = config or ChatConfig()

        # Validate configuration
        missing_settings = self.config.validate_settings()
        if missing_settings:
            raise ValueError(f"Missing required settings: {', '.join(missing_settings)}")

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Create output directory if needed
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)

    def get_blogs(self):
        """Fetch blogs from Shopify"""
        self.logger.info("Fetching blogs from Shopify API...")
        endpoint = f"https://{self.config.SHOPIFY_STORE}.myshopify.com/admin/api/2024-04/blogs.json"
        params = {
            'fields': 'id,updated_at,handle,title',
            'limit': self.config.BLOG_FETCH_LIMIT
        }
        headers = {
            'X-Shopify-Access-Token': self.config.SHOPIFY_API_KEY
        }

        response = requests.get(url=endpoint, params=params, headers=headers)
        if response.status_code == 200:
            body = json.loads(response.content)
            self.logger.info(f"Successfully retrieved {len(body['blogs'])} blogs")
            return body['blogs']
        else:
            self.logger.error(f"Failed retrieving blogs. Status Code: {response.status_code}")
            return []

    def get_articles(self, blog_id):
        """Fetch articles for a specific blog"""
        endpoint = f"https://{self.config.SHOPIFY_STORE}.myshopify.com/admin/api/2024-04/blogs/{blog_id}/articles.json"
        params = {
            'status': 'active',
            'published_status': 'published',
            'fields': 'id,blog_id,updated_at,title,body_html,handle,author',
            'limit': self.config.ARTICLE_FETCH_LIMIT
        }
        headers = {
            'X-Shopify-Access-Token': self.config.SHOPIFY_API_KEY
        }

        response = requests.get(url=endpoint, params=params, headers=headers)
        if response.status_code == 200:
            body = json.loads(response.content)
            return body['articles']
        else:
            self.logger.error(f"Failed retrieving articles. Status Code: {response.status_code}")
            return []

    def get_products(self):
        """Fetch products from Shopify"""
        self.logger.info("Fetching products from Shopify API...")
        endpoint = f"https://{self.config.SHOPIFY_STORE}.myshopify.com/admin/api/2024-04/products.json"
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

        response = requests.get(url=endpoint, params=params, headers=headers)
        if response.status_code == 200:
            body = json.loads(response.content)
            self.logger.info(f"Successfully retrieved {len(body['products'])} products")
            return body['products']
        else:
            self.logger.error(f"Failed retrieving products. Status Code: {response.status_code}")
            return []

    def condense_content_using_llm(self, content):
        """Summarize content using OpenAI's API"""
        client = OpenAI(api_key=self.config.OPENAI_API_KEY)

        response = client.chat.completions.create(
            model=self.config.OPENAI_SUMMARY_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"<markdown>\n{content}\n</markdown>\nYou should shorten the above markdown text to MAXIMUM OF 800 characters while making sure ALL THE HEADINGS AND HYPERLINKS are retained so that the users can refer to those links later. In your response, don't include <markdown> tags."
                        }
                    ]
                }
            ],
            temperature=0,
            max_tokens=1000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].message.content

    def html_to_markdown(self, html_text):
        """Convert HTML to Markdown format"""
        markdown_text = md(html_text, newline_style="BACKSLASH", default_title=True, heading_style='ATX')
        markdown_text = markdown_text.replace('\n\n','\n')

        if self.config.SUMMARIZE_CONTENT:
            self.logger.info("Summarizing content using LLM...")
            markdown_text = self.condense_content_using_llm(markdown_text)

        return markdown_text

    def prepare_blog_articles(self):
        """Process all blogs and their articles"""
        self.logger.info("Fetching and processing blog articles...")
        blogs = self.get_blogs()
        records = []

        for blog in tqdm(blogs):
            try:
                blog['url'] = f"{self.config.SHOPIFY_SITE_BASE_URL}/blogs/{blog['handle']}"
                articles = self.get_articles(blog_id=blog['id'])

                for article in tqdm(articles, desc=f"Processing articles for {blog['title']}"):
                    article['url'] = f"{blog['url']}/{article['handle']}"
                    self.logger.info(f"Converting {article['title']} to markdown!")

                    markdown_text = self.html_to_markdown(article['body_html'])
                    if article['title'] not in markdown_text:
                        markdown_text = f"# {article['title']}\n\n{markdown_text}"

                    article['body_markdown'] = markdown_text
                    if len(markdown_text.strip()):
                        record = {
                            'title': article['title'],
                            'url': article['url'],
                            'markdown': article['body_markdown']
                        }
                        records.append(record)
            except Exception as ex:
                self.logger.error(f"Could not parse {blog['title']} due to: {ex}")

            blog['articles'] = articles

        self.logger.info(f"Prepared {len(records)} blog article records!")
        return blogs, records

    def prepare_products(self):
        """Process all products"""
        self.logger.info("Fetching and processing products...")
        products = self.get_products()
        records = []

        for product in tqdm(products):
            try:
                product['url'] = f"{self.config.SHOPIFY_SITE_BASE_URL}/products/{product['handle']}"
                self.logger.info(f"Converting {product['title']} to markdown!")

                markdown_text = self.html_to_markdown(product['body_html'])
                if product['title'] not in markdown_text:
                    markdown_text = f"# {product['title']}\n\n{markdown_text}"

                product['body_markdown'] = markdown_text
                if len(markdown_text.strip()):
                    record = {
                        'title': product['title'],
                        'url': product['url'],
                        'markdown': product['body_markdown']
                    }
                    records.append(record)
            except Exception as ex:
                self.logger.error(f"Could not parse {product['title']} due to: {ex}")

        self.logger.info(f"Prepared {len(records)} product records!")
        return products, records

    def index_to_pinecone(self, combined_records):
        """Index processed records to Pinecone"""
        self.logger.info(f"Indexing {len(combined_records)} records to Pinecone...")

        # Initialize Pinecone
        pc = Pinecone(api_key=self.config.PINECONE_API_KEY)

        # Check if index exists, create if not
        available_indexes = pc.list_indexes().names()
        if self.config.PINECONE_INDEX_NAME not in available_indexes:
            self.logger.info(f"Index '{self.config.PINECONE_INDEX_NAME}' not found. Creating a new index...")

            pc.create_index(
                name=self.config.PINECONE_INDEX_NAME,
                dimension=self.config.PINECONE_DIMENSION,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud=self.config.PINECONE_CLOUD,
                    region=self.config.PINECONE_REGION
                )
            )
            self.logger.info(f"Index '{self.config.PINECONE_INDEX_NAME}' created successfully.")

        # Create loader for the records
        loader = CustomJsonLoader(
            combined_records,
            dataset_mapping_function=lambda item: Document(
                page_content=item["markdown"] or "",
                metadata={'url': item["url"], "title": item["title"]}
            )
        )

        # Initialize embeddings
        embeddings = OpenAIEmbeddings(
            api_key=self.config.OPENAI_API_KEY,
            model=self.config.OPENAI_EMBEDDING_MODEL,
            dimensions=self.config.PINECONE_DIMENSION
        )

        # Load and split documents
        documents = loader.load()
        self.logger.info(f"Loaded {len(documents)} documents.")

        # Split text into smaller chunks for better embedding
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n"],
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP
        )
        docs = text_splitter.split_documents(documents)
        self.logger.info(f"Split into {len(docs)} chunks.")

        # Store in Pinecone
        try:
            self.logger.info("Uploading documents to Pinecone...")
            vectorstore = PineconeVectorStore.from_documents(
                docs,
                index_name=self.config.PINECONE_INDEX_NAME,
                pinecone_api_key=self.config.PINECONE_API_KEY,
                embedding=embeddings
            )
            self.logger.info(f"Successfully indexed {len(docs)} document chunks to Pinecone index '{self.config.PINECONE_INDEX_NAME}'.")
            return True
        except Exception as e:
            self.logger.error(f"Error indexing to Pinecone: {str(e)}")
            return False

    def run_full_process(self):
        """Run the complete indexing process"""
        try:
            # Step 1: Process blogs and articles
            blogs, blog_records = self.prepare_blog_articles()
            if self.config.SAVE_INTERMEDIATE_FILES:
                with open(self.config.BLOGS_FILE, 'w') as f:
                    json.dump(blogs, f, indent=2)
                with open(self.config.BLOGS_PROCESSED_FILE, 'w') as f:
                    json.dump(blog_records, f, indent=2)

            # Step 2: Process products
            products, product_records = self.prepare_products()
            if self.config.SAVE_INTERMEDIATE_FILES:
                with open(self.config.PRODUCTS_FILE, 'w') as f:
                    json.dump(products, f, indent=2)
                with open(self.config.PRODUCTS_PROCESSED_FILE, 'w') as f:
                    json.dump(product_records, f, indent=2)

            # Step 3: Combine records
            combined_records = blog_records + product_records
            self.logger.info(f"Combined {len(blog_records)} blog records and {len(product_records)} product records.")

            if self.config.SAVE_INTERMEDIATE_FILES:
                with open(self.config.COMBINED_FILE, 'w') as f:
                    json.dump(combined_records, f, indent=2)

            # Step 4: Index to Pinecone
            success = self.index_to_pinecone(combined_records)

            if success:
                self.logger.info("✅ Complete process finished successfully!")
                return {"status": "success", "message": "Indexing completed successfully"}
            else:
                self.logger.error("❌ Process completed with errors in indexing step.")
                return {"status": "error", "message": "Indexing failed"}

        except Exception as e:
            self.logger.error(f"Error in main process: {str(e)}")
            return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    # Run the indexer directly if executed as a script
    indexer = ShopifyIndexer()
    result = indexer.run_full_process()
    print(result)
