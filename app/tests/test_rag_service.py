import unittest
import asyncio
import os
import json
from unittest.mock import patch, MagicMock, AsyncMock
import pytest

from app.config.chat_config import ChatConfig
from app.services.shopify_indexer import ShopifyIndexer
from app.services.chat_service import ChatService
from app.models.chat_models import Message

class TestRAGService(unittest.TestCase):
    """Test the complete RAG pipeline from indexing to chat retrieval"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all tests"""
        # Initialize configuration
        cls.config = ChatConfig()
        cls.config.OUTPUT_DIR = "test_rag_output"
        cls.config.SAVE_INTERMEDIATE_FILES = True

        # Create a unique test index name to avoid conflicts
        cls.config.PINECONE_INDEX_NAME = f"test-rag-index-{os.getpid()}"

        # Create test output directory
        os.makedirs(cls.config.OUTPUT_DIR, exist_ok=True)

        # Initialize chat service
        cls.chat_service = ChatService()

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests have run"""
        # Clean up the test index if it exists
        try:
            indexer = ShopifyIndexer(config=cls.config)
            from pinecone import Pinecone
            pc = Pinecone(api_key=cls.config.PINECONE_API_KEY)

            if cls.config.PINECONE_INDEX_NAME in pc.list_indexes().names():
                pc.delete_index(cls.config.PINECONE_INDEX_NAME)
        except Exception as e:
            print(f"Error cleaning up test index: {e}")

        # Clean up test files
        try:
            for filename in os.listdir(cls.config.OUTPUT_DIR):
                file_path = os.path.join(cls.config.OUTPUT_DIR, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            os.rmdir(cls.config.OUTPUT_DIR)
        except Exception as e:
            print(f"Error cleaning up test files: {e}")

    def setUp(self):
        """Set up before each test method"""
        # Create test data for indexing
        self.test_records = [
            {
                "title": "Advanced Attribution Masterclass",
                "url": "https://msquared.club/products/advanced-attribution-masterclass",
                "markdown": """# Advanced Attribution Masterclass

The Advanced Attribution Masterclass is our flagship training program designed to empower marketing professionals with the knowledge and skills to implement sophisticated attribution models.

## Key Features

- In-depth analysis of multi-touch attribution models
- Hands-on workshops with real-world marketing data
- Practical implementation strategies for various marketing channels
- Expert guidance on selecting the right attribution model for your business

## Who Should Attend

This masterclass is perfect for marketing analysts, digital marketers, and marketing managers who want to move beyond last-click attribution and gain a more holistic view of their marketing effectiveness."""
            },
            {
                "title": "Understanding Marketing Attribution",
                "url": "https://msquared.club/blogs/attribution-today/understanding-marketing-attribution",
                "markdown": """# Understanding Marketing Attribution

Marketing attribution is the process of identifying which marketing actions contribute to conversions or sales. In today's complex digital landscape, customers often interact with multiple touchpoints before making a purchase decision.

## Why Attribution Matters

Attribution helps marketers understand the customer journey and allocate marketing budgets more effectively. Without proper attribution, companies might overinvest in channels that appear to drive conversions but actually only capture the final click.

## Common Attribution Models

1. **Last-click attribution** - Gives 100% credit to the final touchpoint
2. **First-click attribution** - Gives 100% credit to the initial touchpoint
3. **Linear attribution** - Distributes credit equally across all touchpoints
4. **Time-decay attribution** - Gives more credit to touchpoints closer to conversion
5. **Position-based attribution** - Gives 40% to first and last touchpoints, with remaining 20% distributed among middle touchpoints

Choosing the right attribution model depends on your business goals, sales cycle, and marketing channels."""
            }
        ]

    @patch('app.agents.chat_agents.agent_manager')
    @patch('app.services.shopify_indexer.PineconeVectorStore')
    async def test_rag_pipeline(self, mock_pinecone_vectorstore, mock_agent_manager):
        """Test the complete RAG pipeline from indexing to retrieval"""
        # STEP 1: Configure mocks for RAG agent
        mock_rag_response = {
            "output": "The Advanced Attribution Masterclass is MSquared's flagship training program for marketing professionals. It covers multi-touch attribution models, includes hands-on workshops with real data, and provides implementation strategies for various marketing channels. It's designed for marketing analysts, digital marketers, and marketing managers who want to go beyond last-click attribution.",
            "sources": [
                {
                    "title": "Advanced Attribution Masterclass",
                    "url": "https://msquared.club/products/advanced-attribution-masterclass",
                    "content": "The Advanced Attribution Masterclass is our flagship training program designed to empower marketing professionals with the knowledge and skills to implement sophisticated attribution models."
                }
            ],
            "intermediate_steps": []
        }

        mock_std_response = {
            "output": "I don't have specific information about MSquared's Advanced Attribution Masterclass, but in general, attribution masterclasses typically cover topics related to marketing attribution models.",
            "intermediate_steps": []
        }

        # Configure mock RAG agent
        mock_rag_agent = AsyncMock()
        mock_rag_agent.ainvoke.return_value = mock_rag_response

        # Configure mock standard agent
        mock_std_agent = AsyncMock()
        mock_std_agent.ainvoke.return_value = mock_std_response

        # Configure mock database agent
        mock_db_agent = AsyncMock()

        # Assign agents to manager
        mock_agent_manager.rag_agent = mock_rag_agent
        mock_agent_manager.standard_agent = mock_std_agent
        mock_agent_manager.database_agent = mock_db_agent

        # STEP 2: Index test data to Pinecone
        # Configure PineconeVectorStore mock
        mock_vector_store = MagicMock()
        mock_pinecone_vectorstore.from_documents.return_value = mock_vector_store

        # Create and run the indexer
        indexer = ShopifyIndexer(config=self.config)

        # Mock the Pinecone class
        with patch('pinecone.Pinecone') as mock_pinecone:
            # Configure mock Pinecone instance
            mock_pinecone_instance = mock_pinecone.return_value
            mock_list_indexes = MagicMock()
            mock_list_indexes.names.return_value = []  # Empty list means index doesn't exist
            mock_pinecone_instance.list_indexes.return_value = mock_list_indexes

            # Mock ServerlessSpec
            with patch('pinecone.ServerlessSpec') as mock_serverless_spec:
                # Index the test records
                result = indexer.index_to_pinecone(self.test_records)

                # Verify indexing was successful
                self.assertTrue(result, "Indexing to Pinecone failed")

                # Verify that create_index was called
                mock_pinecone_instance.create_index_from_shopify_store.assert_called_once()

                # Verify that from_documents was called
                mock_pinecone_vectorstore.from_documents.assert_called_once()

        # STEP 3: Test chat service with a relevant query
        # Create a test query related to the indexed content
        test_query = "What is the Advanced Attribution Masterclass?"
        test_session_id = "test-session-123"

        # Create a message object
        message = Message(message=test_query, session_id=test_session_id)

        # Call the chat service
        response = await self.chat_service.chat(message)

        # STEP 4: Verify the results
        # Check that RAG content was returned
        self.assertIsNotNone(response.response.output)
        self.assertTrue("Advanced Attribution Masterclass" in response.response.output)

        # Check that sources were returned
        self.assertTrue(len(response.sources) > 0)

        # Verify the source contains our test data
        source_found = False
        for source in response.sources:
            if source.title == "Advanced Attribution Masterclass":
                source_found = True
                break

        self.assertTrue(source_found, "Expected source not found in response")

        # Verify the chat history was updated
        chat_history = self.chat_service.chat_histories.get(test_session_id)
        self.assertIsNotNone(chat_history)
        self.assertEqual(len(chat_history.get_messages()), 2)  # User message + AI response

        print("RAG pipeline test completed successfully!")

    @patch('app.services.chat_service.chat_cache')
    @patch('app.agents.chat_agents.agent_manager')
    async def test_is_database_query_routing(self, mock_agent_manager, mock_chat_cache):
        """Test that queries are properly routed between RAG and database agents"""
        # Configure mock agents
        mock_rag_agent = AsyncMock()
        mock_rag_agent.ainvoke.return_value = {
            "output": "RAG agent response",
            "sources": [],
            "intermediate_steps": []
        }

        mock_db_agent = AsyncMock()
        mock_db_agent.ainvoke.return_value = {
            "output": "Database agent response",
            "sources": [],
            "intermediate_steps": []
        }

        mock_std_agent = AsyncMock()
        mock_std_agent.ainvoke.return_value = {
            "output": "Standard agent response",
            "sources": [],
            "intermediate_steps": []
        }

        # Assign agents to manager
        mock_agent_manager.rag_agent = mock_rag_agent
        mock_agent_manager.database_agent = mock_db_agent
        mock_agent_manager.standard_agent = mock_std_agent

        # Configure cache mock
        mock_chat_cache.generate_query_hash.return_value = "test_hash"
        mock_chat_cache.get_cached_response.return_value = (None, False)

        # Test database query routing
        db_queries = [
            "Show me the metrics for our top campaigns",
            "What were our sales figures last month?",
            "Calculate the average conversion rate",
            "How many customers converted from social media?",
            "List the top 10 performing ads"
        ]

        # Test RAG query routing
        rag_queries = [
            "What is marketing attribution?",
            "Explain multi-touch attribution models",
            "Tell me about the Advanced Attribution Masterclass",
            "How does incrementality testing work?",
            "What best practices does MSquared recommend for attribution?"
        ]

        # Test database queries
        for query in db_queries:
            message = Message(message=query, session_id="test-session")
            response = await self.chat_service.chat(message)
            # Verify database agent was called
            mock_db_agent.ainvoke.assert_called()
            mock_db_agent.reset_mock()

        # Test RAG queries
        for query in rag_queries:
            message = Message(message=query, session_id="test-session")
            response = await self.chat_service.chat(message)
            # Verify RAG agent was called
            mock_rag_agent.ainvoke.assert_called()
            mock_rag_agent.reset_mock()

        print("Query routing test completed successfully!")


@pytest.mark.asyncio
async def test_rag_system():
    """Run the async tests using pytest"""
    test_instance = TestRAGService()
    test_instance.setUp()
    await test_instance.test_rag_pipeline()
    await test_instance.test_is_database_query_routing()
    test_instance.tearDown()


if __name__ == "__main__":
    # Run the async tests
    asyncio.run(test_rag_system())