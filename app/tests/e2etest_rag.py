#!/usr/bin/env python
"""
End-to-End RAG Test Script

This script performs a complete test of the RAG (Retrieval Augmented Generation) system
by indexing test content, running a query, and verifying the results.

Usage:
    python -m app.tests.e2etest_rag

The script will:
1. Create test content about marketing attribution
2. Index the content to a test Pinecone index
3. Query the ChatService with a related question
4. Display the response with sources
5. Clean up the test index and files
"""

import os
import sys
import json
import asyncio
import time
import random
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.absolute()
print(f"Project root: {project_root}")
if str(project_root) not in sys.path:
    print(f"Adding {project_root} to sys.path")
    sys.path.insert(0, str(project_root))

# Sample test content
TEST_CONTENT = [
    {
        "title": "Multi-Touch Attribution Models Explained",
        "url": "https://msquared.club/blogs/attribution-today/multi-touch-attribution-models-explained",
        "markdown": """# Multi-Touch Attribution Models Explained

Multi-touch attribution (MTA) is a method of marketing measurement that evaluates the impact of each touchpoint in driving conversions. Unlike single-touch models, MTA provides a more comprehensive view of the customer journey.

## Common Multi-Touch Attribution Models

### Linear Attribution
The Linear attribution model gives equal credit to each touchpoint in the customer journey. For example, if a customer interacts with five marketing channels before converting, each channel receives 20% of the conversion credit.

### Time Decay Attribution
The Time Decay model assigns more credit to touchpoints closer to the conversion. This model assumes that interactions that happened more recently had a stronger influence on the purchasing decision.

### Position-Based Attribution
Also known as the U-shaped model, Position-Based attribution assigns 40% of the credit to the first touch, 40% to the last touch, and divides the remaining 20% among the middle touches.

### Algorithmic Attribution
Algorithmic or data-driven attribution uses machine learning to dynamically assign fractional credit to touchpoints based on their actual impact on conversions.

## Choosing the Right Attribution Model

The right attribution model depends on your business model, sales cycle, and marketing channels. For businesses with longer sales cycles, multi-touch models typically provide more accurate insights than single-touch models like first or last click."""
    },
    {
        "title": "Incrementality Testing: The Gold Standard for Attribution",
        "url": "https://msquared.club/blogs/attribution-today/incrementality-testing-gold-standard",
        "markdown": """# Incrementality Testing: The Gold Standard for Attribution

Incrementality testing is considered the gold standard for measuring marketing effectiveness. It answers a fundamental question: "What would have happened if we hadn't run this marketing campaign?"

## What is Incrementality Testing?

Incrementality testing is a controlled experiment that measures the true impact of a marketing activity by comparing a test group (exposed to the marketing) with a control group (not exposed). The difference in performance between these groups represents the incremental impact of the marketing activity.

## Benefits of Incrementality Testing

### Eliminates Attribution Bias
Unlike traditional attribution models that make assumptions about how to assign credit, incrementality testing directly measures the causal impact of marketing.

### Identifies True Effectiveness
It helps identify which channels and campaigns are truly driving incremental conversions versus those that are simply intercepting customers who would have converted anyway.

### Optimizes Marketing Budget
With accurate measurement of incremental impact, marketers can allocate budget to channels that genuinely drive new business.

## How to Implement Incrementality Testing

1. Define clear test objectives
2. Establish statistically significant test and control groups
3. Ensure proper isolation between groups
4. Run the test for a sufficient duration
5. Analyze results and calculate incrementality lift

## Challenges and Limitations

While incrementality testing is powerful, it requires sufficient data volume, can be resource-intensive, and may not be practical for all marketing activities. It's often best used for evaluating major channels or campaigns rather than every marketing tactic."""
    },
    {
        "title": "Attribution Dashboard Pro",
        "url": "https://msquared.club/products/attribution-dashboard-pro",
        "markdown": """# Attribution Dashboard Pro

Attribution Dashboard Pro is our comprehensive analytics solution designed to give marketers complete visibility into their cross-channel performance.

## Key Features

### Multi-Model Attribution
Compare results across different attribution models (first-click, last-click, linear, time decay, position-based, and custom) to gain a complete understanding of your marketing performance.

### Custom Attribution Windows
Define custom lookback windows for different conversion types, recognizing that some customer journeys are longer than others.

### Cross-Device Tracking
Track user journeys across multiple devices for a complete view of the path to conversion.

### Marketing Mix Modeling Integration
Combine the power of attribution data with marketing mix modeling for both tactical and strategic insights.

### Automated Anomaly Detection
Receive alerts when metrics deviate significantly from expected values, allowing for quick response to issues or opportunities.

## Pricing and Plans

Attribution Dashboard Pro is available in three tiers:

* **Starter**: $499/month - Ideal for small businesses with up to 500,000 monthly sessions
* **Growth**: $999/month - Perfect for mid-sized companies with up to 2 million monthly sessions
* **Enterprise**: Custom pricing - For large organizations with complex attribution needs

All plans include email support, regular updates, and access to our attribution knowledge base."""
    }
]


class RAGEndToEndTest:
    """Complete end-to-end test for the RAG system"""

    def __init__(self):
        """Initialize the test environment"""
        # Import here to avoid circular imports with app.main
        from app.config.chat_config import ChatConfig
        from app.services.indexing.providers.shopify_indexer import ShopifyIndexer
        from app.services.chat.chat_service import ChatService
        from app.models.chat_models import Message
        
        # These classes need to be available as instance variables
        self.ChatService = ChatService
        self.Message = Message
        
        # Load environment variables
        load_dotenv()

        # Create test timestamp for unique identifiers
        self.timestamp = int(time.time())
        self.test_id = f"rag-test-{self.timestamp}-{random.randint(1000, 9999)}"

        # Initialize configuration
        self.config = ChatConfig()
        
        # Set up test-specific configuration
        self.config.OUTPUT_DIR = f"test_rag_output_{self.test_id}"
        self.config.SAVE_INTERMEDIATE_FILES = True

        # Create a unique test index name to avoid conflicts
        self.config.PINECONE_INDEX_NAME = f"test-rag-index-{self.test_id}"
        
        # Update region to work with free Pinecone plan if needed
        self.config.PINECONE_CLOUD = "aws"
        self.config.PINECONE_REGION = "us-east-1"  # Update this based on your Pinecone account

        # Create test output directory
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)

        # Initialize services
        self.indexer = ShopifyIndexer(config=self.config)
        self.chat_service = ChatService()

        # Test content
        self.test_content = TEST_CONTENT

        print(f"\n{'=' * 80}")
        print(f"🧪 STARTING RAG END-TO-END TEST (ID: {self.test_id})")
        print(f"{'=' * 80}\n")

    async def check_and_clean_indexes(self):
        """Check and clean old test indexes if needed"""
        try:
            from pinecone import Pinecone
            pc = Pinecone(api_key=self.config.PINECONE_API_KEY)

            # Get list of indexes
            print("📋 Checking for existing Pinecone indexes...")
            indexes = pc.list_indexes().names()

            # Find old test indexes (older than 1 hour)
            test_indexes = [idx for idx in indexes if idx.startswith('test-rag-index-')]

            if test_indexes:
                print(f"🧹 Found {len(test_indexes)} old test indexes, cleaning up...")

                # Delete old test indexes
                for idx in test_indexes:
                    try:
                        print(f"   🗑️ Deleting old test index: {idx}")
                        pc.delete_index(idx)
                    except Exception as e:
                        print(f"   ⚠️ Error deleting index {idx}: {str(e)}")

                # Wait a moment for deletion to complete
                time.sleep(2)

                print("✅ Cleanup complete")
            else:
                print("✅ No old test indexes found")

            return True

        except Exception as e:
            print(f"⚠️ Error checking/cleaning indexes: {str(e)}")
            return False

    async def run_test(self):
        """Run the complete end-to-end test"""
        try:
            # Step 0: Check and clean old test indexes
            await self.check_and_clean_indexes()

            # Step 1: Index test content
            success = await self.index_test_content()
            if not success:
                print("❌ TEST FAILED: Content indexing failed")
                return False

            # Step 2: Run test queries
            success = await self.run_test_queries()
            if not success:
                print("❌ TEST FAILED: Query testing failed")
                return False

            print(f"\n{'=' * 80}")
            print(f"✅ RAG END-TO-END TEST COMPLETED SUCCESSFULLY")
            print(f"{'=' * 80}\n")
            return True

        except Exception as e:
            print(f"❌ TEST FAILED with exception: {str(e)}")
            raise
        finally:
            # Clean up resources
            await self.cleanup()

    async def index_test_content(self):
        """Index test content to Pinecone"""
        try:
            print(f"📑 Indexing {len(self.test_content)} test documents to Pinecone...")

            # Save test content to file
            test_content_path = os.path.join(self.config.OUTPUT_DIR, "test_content.json")
            with open(test_content_path, "w") as f:
                json.dump(self.test_content, f, indent=2)

            # Index the content
            start_time = time.time()
            result = self.indexer.index_to_pinecone(self.test_content)
            duration = time.time() - start_time

            if result:
                print(f"✅ Content indexed successfully in {duration:.2f} seconds")
                return True
            else:
                print("❌ Failed to index content to Pinecone")
                return False

        except Exception as e:
            print(f"❌ Error indexing content: {str(e)}")
            return False

    async def run_test_queries(self):
        """Run test queries against the RAG system"""
        test_queries = [
            "What are the different types of multi-touch attribution models?",
            "How does incrementality testing work?",
            "What is the Attribution Dashboard Pro and how much does it cost?",
            "What's the difference between linear and time decay attribution?"
        ]

        success = True

        print(f"\n📝 Running {len(test_queries)} test queries...")

        for i, query in enumerate(test_queries, 1):
            print(f"\n----- Query {i}: \"{query}\" -----")

            try:
                # Create message object with required 'mode' parameter
                message = self.Message(
                    message=query, 
                    session_id=self.test_id,
                    mode="rag"  # Add the required mode parameter
                )

                # Process the query
                start_time = time.time()
                response = await self.chat_service.chat(message)
                duration = time.time() - start_time

                # Display results
                print(f"⏱️ Response time: {duration:.2f} seconds")
                print(f"\n📄 RESPONSE:\n{response.response.output}")

                # Display sources
                if response.sources:
                    print(f"\n📚 SOURCES ({len(response.sources)}):")
                    for idx, source in enumerate(response.sources, 1):
                        print(f"{idx}. {source.title}")
                        print(f"   URL: {source.url}")
                        if source.content and len(source.content) > 100:
                            print(f"   Preview: {source.content[:100]}...")
                        else:
                            print(f"   Content: {source.content}")
                else:
                    print("\n⚠️ No sources returned")

                # Basic verification
                if any(keyword in response.response.output.lower() for keyword in query.lower().split()):
                    print("\n✅ Response appears relevant to the query")
                else:
                    print("\n⚠️ Response may not be relevant to the query")
                    success = False

                # Save response to file
                response_file = os.path.join(self.config.OUTPUT_DIR, f"query_{i}_response.json")
                with open(response_file, "w") as f:
                    json.dump({
                        "query": query,
                        "response": response.response.output,
                        "no_rag_response": response.response.no_rag_output,
                        "sources": [s.dict() for s in response.sources if hasattr(s, 'dict')],
                        "timestamp": datetime.now().isoformat()
                    }, f, indent=2)

            except Exception as e:
                print(f"❌ Error processing query: {str(e)}")
                success = False

        return success

    async def cleanup(self):
        """Clean up test resources"""
        try:
            print(f"\n🧹 Cleaning up test resources...")

            # Attempt to delete the test index
            try:
                from pinecone import Pinecone
                pc = Pinecone(api_key=self.config.PINECONE_API_KEY)

                # Check if index exists
                indexes = pc.list_indexes().names()
                if self.config.PINECONE_INDEX_NAME in indexes:
                    print(f"🗑️ Deleting test index '{self.config.PINECONE_INDEX_NAME}'...")
                    pc.delete_index(self.config.PINECONE_INDEX_NAME)
                    print(f"✅ Test index deleted successfully")
            except Exception as e:
                print(f"⚠️ Error deleting test index: {str(e)}")

            # Keep the output directory for inspection
            print(f"📂 Test output saved to '{self.config.OUTPUT_DIR}'")
            print(f"   You may want to delete this directory manually after inspection")

        except Exception as e:
            print(f"⚠️ Error during cleanup: {str(e)}")


async def main():
    """Main entry point"""
    # Print debugging information
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Python path: {sys.path}")
    
    test = RAGEndToEndTest()
    await test.run_test()


if __name__ == "__main__":
    asyncio.run(main())