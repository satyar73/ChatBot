import unittest
from unittest.mock import patch, MagicMock
import json
import requests
import os
from app.config.chat_config import ChatConfig
from app.services.shopify_indexer import ShopifyIndexer  # Updated import path


class TestShopifyIndexer(unittest.TestCase):
    def setUp(self):
        # Create a config with test values
        self.config = ChatConfig()
        self.config.BLOG_FETCH_LIMIT = 10
        self.config.ARTICLE_FETCH_LIMIT = 20
        self.config.PRODUCT_FETCH_LIMIT = 30
        self.config.OUTPUT_DIR = "test_output"
        self.config.SAVE_INTERMEDIATE_FILES = False

        # Create test output directory
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)

        # Initialize the indexer with the test config
        self.indexer = ShopifyIndexer(config=self.config)

    def tearDown(self):
        # Clean up test directory if needed
        pass

    @patch('requests.get')
    def test_get_blogs_success(self, mock_get):
        # Set up the mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = json.dumps({
            "blogs": [
                {
                    "id": 123456,
                    "handle": "test-blog",
                    "title": "Test Blog",
                    "updated_at": "2023-01-01T00:00:00Z"
                }
            ]
        })
        mock_get.return_value = mock_response

        # Call the method
        blogs = self.indexer.get_blogs()

        # Verify the request was made correctly
        mock_get.assert_called_once_with(
            url="https://919904.myshopify.com/admin/api/2024-04/blogs.json",
            params={
                'fields': 'id,updated_at,handle,title',
                'limit': 10
            },
            headers={
                'X-Shopify-Access-Token': self.config.SHOPIFY_API_KEY
            }
        )

        # Verify the result
        self.assertEqual(len(blogs), 1)
        self.assertEqual(blogs[0]['id'], 123456)
        self.assertEqual(blogs[0]['title'], "Test Blog")

    @patch('requests.get')
    def test_get_blogs_failure(self, mock_get):
        # Set up the mock response for a failure
        mock_response = MagicMock()
        mock_response.status_code = 401  # Unauthorized
        mock_get.return_value = mock_response

        # Call the method
        blogs = self.indexer.get_blogs()

        # Verify an empty list is returned on failure
        self.assertEqual(blogs, [])

    @patch('requests.get')
    def test_get_articles_success(self, mock_get):
        # Set up the mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = json.dumps({
            "articles": [
                {
                    "id": 789012,
                    "blog_id": 123456,
                    "title": "Test Article",
                    "handle": "test-article",
                    "author": "Test Author",
                    "body_html": "<p>This is a test article</p>",
                    "updated_at": "2023-01-02T00:00:00Z"
                }
            ]
        })
        mock_get.return_value = mock_response

        # Call the method
        articles = self.indexer.get_articles(blog_id=123456)

        # Verify the request was made correctly
        mock_get.assert_called_once_with(
            url="https://919904.myshopify.com/admin/api/2024-04/blogs/123456/articles.json",
            params={
                'status': 'active',
                'published_status': 'published',
                'fields': 'id,blog_id,updated_at,title,body_html,handle,author',
                'limit': 20
            },
            headers={
                'X-Shopify-Access-Token': self.config.SHOPIFY_API_KEY
            }
        )

        # Verify the result
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0]['id'], 789012)
        self.assertEqual(articles[0]['title'], "Test Article")

    @patch('requests.get')
    def test_get_articles_failure(self, mock_get):
        # Set up the mock response for a failure
        mock_response = MagicMock()
        mock_response.status_code = 404  # Not Found
        mock_get.return_value = mock_response

        # Call the method
        articles = self.indexer.get_articles(blog_id=999999)  # Non-existent blog ID

        # Verify an empty list is returned on failure
        self.assertEqual(articles, [])

    @patch('requests.get')
    def test_get_products_success(self, mock_get):
        # Set up the mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = json.dumps({
            "products": [
                {
                    "id": 345678,
                    "title": "Test Product",
                    "handle": "test-product",
                    "body_html": "<p>This is a test product</p>",
                    "updated_at": "2023-01-03T00:00:00Z"
                }
            ]
        })
        mock_get.return_value = mock_response

        # Call the method
        products = self.indexer.get_products()

        # Verify the request was made correctly
        mock_get.assert_called_once_with(
            url="https://919904.myshopify.com/admin/api/2024-04/products.json",
            params={
                'status': 'active',
                'published_status': 'published',
                'fields': 'id,updated_at,title,body_html,handle',
                'presentment_currencies': 'USD',
                'limit': 30
            },
            headers={
                'X-Shopify-Access-Token': self.config.SHOPIFY_API_KEY
            }
        )

        # Verify the result
        self.assertEqual(len(products), 1)
        self.assertEqual(products[0]['id'], 345678)
        self.assertEqual(products[0]['title'], "Test Product")

    @patch('requests.get')
    def test_get_products_failure(self, mock_get):
        # Set up the mock response for a failure
        mock_response = MagicMock()
        mock_response.status_code = 500  # Server Error
        mock_get.return_value = mock_response

        # Call the method
        products = self.indexer.get_products()

        # Verify an empty list is returned on failure
        self.assertEqual(products, [])

    # Update the patch path to use the full module path
    @patch('app.services.shopify_indexer.md')
    def test_html_to_markdown(self, mock_md):
        # Set up the mock response
        mock_md.return_value = "# Test Heading\nThis is a test paragraph"

        # Test with summarize_content = False
        self.config.SUMMARIZE_CONTENT = False
        result = self.indexer.html_to_markdown("<h1>Test Heading</h1><p>This is a test paragraph</p>")

        # Verify the result
        self.assertEqual(result, "# Test Heading\nThis is a test paragraph")
        mock_md.assert_called_once()

    # Update the patch paths to use the full module paths
    @patch('app.services.shopify_indexer.ShopifyIndexer.get_blogs')
    @patch('app.services.shopify_indexer.ShopifyIndexer.get_articles')
    @patch('app.services.shopify_indexer.ShopifyIndexer.html_to_markdown')
    def test_prepare_blog_articles(self, mock_html_to_md, mock_get_articles, mock_get_blogs):
        # Set up mock responses
        mock_get_blogs.return_value = [
            {
                'id': 123,
                'handle': 'test-blog',
                'title': 'Test Blog'
            }
        ]

        mock_get_articles.return_value = [
            {
                'id': 456,
                'handle': 'test-article',
                'title': 'Test Article',
                'body_html': '<p>Article content</p>'
            }
        ]

        mock_html_to_md.return_value = "# Test Article\n\nArticle content"

        # Set shopify site base URL
        self.config.SHOPIFY_SITE_BASE_URL = "https://test-store.myshopify.com"

        # Call the method
        blogs, records = self.indexer.prepare_blog_articles()

        # Verify the results
        self.assertEqual(len(blogs), 1)
        self.assertEqual(blogs[0]['title'], 'Test Blog')
        self.assertEqual(blogs[0]['url'], 'https://test-store.myshopify.com/blogs/test-blog')

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['title'], 'Test Article')
        self.assertEqual(records[0]['url'], 'https://test-store.myshopify.com/blogs/test-blog/test-article')
        self.assertEqual(records[0]['markdown'], '# Test Article\n\nArticle content')

    # Update the patch paths to use the full module paths
    @patch('app.services.shopify_indexer.ShopifyIndexer.get_products')
    @patch('app.services.shopify_indexer.ShopifyIndexer.html_to_markdown')
    def test_prepare_products(self, mock_html_to_md, mock_get_products):
        # Set up mock responses
        mock_get_products.return_value = [
            {
                'id': 789,
                'handle': 'test-product',
                'title': 'Test Product',
                'body_html': '<p>Product description</p>'
            }
        ]

        mock_html_to_md.return_value = "# Test Product\n\nProduct description"

        # Set shopify site base URL
        self.config.SHOPIFY_SITE_BASE_URL = "https://test-store.myshopify.com"

        # Call the method
        products, records = self.indexer.prepare_products()

        # Verify the results
        self.assertEqual(len(products), 1)
        self.assertEqual(products[0]['title'], 'Test Product')
        self.assertEqual(products[0]['url'], 'https://test-store.myshopify.com/products/test-product')

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['title'], 'Test Product')
        self.assertEqual(records[0]['url'], 'https://test-store.myshopify.com/products/test-product')
        self.assertEqual(records[0]['markdown'], '# Test Product\n\nProduct description')

    # Here are the three test methods that were defined outside the class before
    @patch('app.services.shopify_indexer.Pinecone')
    @patch('app.services.shopify_indexer.PineconeVectorStore')
    @patch('app.services.shopify_indexer.OpenAIEmbeddings')
    def test_index_to_pinecone_new_index(self, mock_embeddings, mock_pinecone_vectorstore, mock_pinecone):
        """Test creating a new Pinecone index and indexing documents"""
        # Set up necessary configs
        self.config.PINECONE_INDEX_NAME = "test-index"
        self.config.PINECONE_DIMENSION = 1536
        self.config.PINECONE_CLOUD = "aws"
        self.config.PINECONE_REGION = "us-west-2"
        self.config.PINECONE_API_KEY = "test-pinecone-api-key"
        self.config.OPENAI_API_KEY = "test-openai-api-key"
        self.config.OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"
        self.config.CHUNK_SIZE = 1000
        self.config.CHUNK_OVERLAP = 200

        # Set up mock Pinecone instance and index
        mock_pinecone_instance = mock_pinecone.return_value
        mock_list_indexes = MagicMock()
        mock_list_indexes.names.return_value = []  # Empty list means index doesn't exist
        mock_pinecone_instance.list_indexes.return_value = mock_list_indexes

        # Set up mock for vector store
        mock_vectorstore = mock_pinecone_vectorstore.from_documents.return_value

        # Test data for indexing
        test_records = [
            {
                "title": "Test Article",
                "url": "https://test-store.myshopify.com/blogs/test-blog/test-article",
                "markdown": "# Test Article\n\nThis is a test article content"
            },
            {
                "title": "Test Product",
                "url": "https://test-store.myshopify.com/products/test-product",
                "markdown": "# Test Product\n\nThis is a test product description"
            }
        ]

        # Add mock for ServerlessSpec
        mock_serverless_spec = MagicMock()
        mock_serverless_spec.return_value = MagicMock()
        with patch('pinecone.ServerlessSpec', mock_serverless_spec):
            # Call the method
            result = self.indexer.index_to_pinecone(test_records)

        # Verify that Pinecone was initialized with the correct API key
        mock_pinecone.assert_called_once_with(api_key="test-pinecone-api-key")

        # Verify that we checked for existing indexes
        mock_pinecone_instance.list_indexes.assert_called_once()

        # Verify that a new index was created with correct parameters
        mock_pinecone_instance.create_index_from_shopify_store.assert_called_once()
        create_index_call = mock_pinecone_instance.create_index_from_shopify_store.call_args
        create_index_call = mock_pinecone_instance.create_index_from_shopify_store.call_args
        self.assertEqual(create_index_call[1]["name"], "test-index")
        self.assertEqual(create_index_call[1]["dimension"], 1536)
        self.assertEqual(create_index_call[1]["metric"], "cosine")
        # Check ServerlessSpec was passed correctly
        self.assertEqual(create_index_call[1]["spec"].cloud, "aws")
        self.assertEqual(create_index_call[1]["spec"].region, "us-west-2")

        # Verify that OpenAI embeddings were initialized with correct parameters
        mock_embeddings.assert_called_once_with(
            api_key="test-openai-api-key",
            model="text-embedding-ada-002",
            dimensions=1536
        )

        # Verify that documents were indexed
        mock_pinecone_vectorstore.from_documents.assert_called_once()
        from_docs_call = mock_pinecone_vectorstore.from_documents.call_args
        # Check the index name and API key in the from_documents call
        self.assertEqual(from_docs_call[1]["index_name"], "test-index")
        self.assertEqual(from_docs_call[1]["pinecone_api_key"], "test-pinecone-api-key")

        # Verify the result
        self.assertTrue(result)

    @patch('app.services.shopify_indexer.Pinecone')
    @patch('app.services.shopify_indexer.PineconeVectorStore')
    def test_index_to_pinecone_empty_records(self, mock_pinecone_vectorstore, mock_pinecone):
        """Test handling of empty records list"""
        # Empty test data
        test_records = []

        # Call the method
        result = self.indexer.index_to_pinecone(test_records)

        # Verify the result - implementation dependent
        # Some implementations might return True for empty lists (nothing to do = success)
        # Others might return False (nothing to index = failure)
        # Adjust as needed based on your implementation
        self.assertTrue(result)  # Adjust based on expected behavior

    @patch('app.services.shopify_indexer.Pinecone')
    @patch('app.services.shopify_indexer.PineconeVectorStore')
    @patch('app.services.shopify_indexer.OpenAIEmbeddings')
    def test_index_to_pinecone_exception_handling(self, mock_embeddings, mock_pinecone_vectorstore, mock_pinecone):
        """Test exception handling during indexing"""
        # Set up necessary configs
        self.config.PINECONE_API_KEY = "test-pinecone-api-key"
        self.config.PINECONE_INDEX_NAME = "test-index"
        self.config.PINECONE_DIMENSION = 1536
        self.config.OPENAI_API_KEY = "test-openai-api-key"
        self.config.OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"
        self.config.CHUNK_SIZE = 1000
        self.config.CHUNK_OVERLAP = 200

        # Set up mock Pinecone instance and index
        mock_pinecone_instance = mock_pinecone.return_value
        mock_list_indexes = MagicMock()
        mock_list_indexes.names.return_value = ["test-index"]  # Index exists
        mock_pinecone_instance.list_indexes.return_value = mock_list_indexes

        # Set up mock to raise an exception
        mock_pinecone_vectorstore.from_documents.side_effect = Exception("Test exception")

        # Test data for indexing
        test_records = [
            {
                "title": "Test Article",
                "url": "https://test-store.myshopify.com/blogs/test-blog/test-article",
                "markdown": "# Test Article\n\nThis is a test article content"
            }
        ]

        # Call the method
        result = self.indexer.index_to_pinecone(test_records)

        # Verify the result - should return False on exception
        self.assertFalse(result)

    def test_get_blogs_real(self):
        """This test uses the real API - only run when needed"""
        blogs = self.indexer.get_blogs()
        print(f"Retrieved {len(blogs)} blogs:")
        for blog in blogs:
            print(f"- {blog['title']} (ID: {blog['id']})")

        # Basic validation
        self.assertIsInstance(blogs, list)
        if blogs:
            self.assertIn('id', blogs[0])
            self.assertIn('title', blogs[0])

    def test_get_products_real(self):
        """This test uses the real API - only run when needed"""
        products = self.indexer.get_products()
        print(f"Retrieved {len(products)} products:")
        for product in products[:5]:  # Show first 5 products
            print(f"- {product['title']} (ID: {product['id']})")

        # Basic validation
        self.assertIsInstance(products, list)
        if products:
            self.assertIn('id', products[0])
            self.assertIn('title', products[0])


if __name__ == '__main__':
    # You have two options to run the tests:

    # Option 1: Run all tests
    # unittest.main()

    # Option 2: Run specific tests
    suite = unittest.TestSuite()

    # Add the specific tests you want to run
    suite.addTest(TestShopifyIndexer('test_index_to_pinecone_new_index'))
    suite.addTest(TestShopifyIndexer('test_index_to_pinecone_empty_records'))
    suite.addTest(TestShopifyIndexer('test_index_to_pinecone_exception_handling'))

    # Uncomment these if you want to run real API tests (be careful with these)
    # suite.addTest(TestShopifyIndexer('test_get_blogs_real'))
    # suite.addTest(TestShopifyIndexer('test_get_products_real'))

    # Run the selected tests
    unittest.TextTestRunner().run(suite)