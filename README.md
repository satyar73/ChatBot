# ChatBot Example - RAG-Enabled Marketing Assistant

## Overview

This is a RAG-enabled chatbot system designed for marketing professionals and analytics experts. The chatbot helps answer questions about marketing services, blogs, products, and attribution strategies using specialized knowledge bases.

## Features

- **RAG (Retrieval-Augmented Generation)**: Enhances responses with domain-specific knowledge
- **Dual Response System**: Generates both RAG and non-RAG responses for comparison
- **Multiple Data Sources**:
    - Google Drive integration (docs, PDFs, slides)
    - Shopify storefront (products, blogs)
- **Vector Search**: Uses Pinecone and OpenAI embeddings for semantic similarity
- **FastAPI Backend**: Robust API with configurable endpoints
- **Advanced Logging**: Comprehensive logging for debugging and monitoring
- **Response Caching**: Efficient local caching system for faster responses
- **Cache Analytics**: Built-in metrics to measure cache performance
- **Multi-Mode UI**: Toggle between RAG, standard, and side-by-side comparison views
- **Advanced LLM Interaction Logging**: Detailed logs of tool usage, execution tracing, and response capture
- **Robust Response Mode Handling**: Improved handling of RAG, no-RAG, and comparison modes with enhanced error resilience

## Tech Stack

- **Framework**: FastAPI, LangChain
- **AI Models**: OpenAI GPT models
- **Vector DB**: Pinecone
- **Document Processing**: Support for PDF, DOCX, PPTX, and Markdown
- **Testing**: Includes test suite for semantic accuracy

## Prerequisites

- Python 3.12+
- OpenAI API key
- Pinecone API key
- (Optional) Google Drive API credentials
- (Optional) Shopify API credentials

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/satyar73/ChatBotExample/
   cd ChatBotExample
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables:
   ```
   # Create a .env file with your API keys
   OPENAI_API_KEY=your_openai_key
   PINECONE_API_KEY=your_pinecone_key
   PINECONE_ENVIRONMENT=your_pinecone_env
   PINECONE_INDEX=your_index_name
   ```

5. Additional setup for data sources:
    - For Google Drive: See [Google Drive Setup Guide](./setup/README.GoogleDriveSetup.md)
    - For Shopify: Configure Shopify API keys in your environment variables
   ```
   SHOPIFY_API_KEY=your_shopify_key_name
   ```

## Usage

### Running with Docker (Recommended)

1. Ensure Docker and Docker Compose are installed.
2. Copy the `.env.example` file to `.env` and update it with your API keys:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```
3. Build and start the containers:
   ```bash
   docker-compose up -d
   ```
4. The application will be available at:
    - Frontend: http://localhost
    - Backend API: http://localhost:8005

To stop the containers:
```bash
docker-compose down
```

To view logs:
```bash
docker-compose logs -f
```

### Running Locally
```bash
python run.py
```

or
```bash
uvicorn app:app --port 8005 --reload
```

## API Endpoints

### Chat & Session Management
- `/chat`: Submit chat messages
- `/chat/session/{session_id}`: Get chat history for a specific session
- `/chat/session/{session_id}`: Delete a specific chat session (DELETE method)

### Cache Management
- `/chat/cache/stats`: Get cache performance statistics
- `/chat/cache`: Clear cache entries (DELETE method, optional parameter: older_than_days)

### Testing
- `/chat/test`: Run a single test case
- `/chat/batch-test`: Run a batch of tests from a CSV file

### Indexing
- `/index/`: Get index information or create a new index (Shopify)
- `/index/google-drive`: Create and populate an index with Google Drive data
- `/health`: Check server status

## Development

- **Environment**: Set `ENVIRONMENT=development` for development mode with auto-reload
- **Logging**: Adjust log levels in `app/config/logging_config.py`
- **Testing**: Run test suite with included chat test data
- **Caching**: Configure cache behavior in `app/config/cache_config.py`
- **Cache Analytics**: Access cache statistics via the `/chat/cache/stats` endpoint

## Cache System

The application includes a SQLite-based response caching system:

- **Configuration**: Customize settings in `app/config/cache_config.py`
- **Cache Duration**: Default TTL is 24 hours (configurable)
- **Cache Size**: Limits maximum entries to prevent unbounded growth
- **Query Hashing**: Considers user input and recent conversation history
- **Analytics**: Track hit rates, response times, and time saved
- **Management**: Clear cache via API or adjust TTL as needed

## Managing the Cache

To clear the entire cache through the API:
```bash
curl -X DELETE http://localhost:8005/chat/cache
```

To clear only entries older than a specific number of days:
```bash
curl -X DELETE "http://localhost:8005/chat/cache?older_than_days=7"
```

To view cache performance statistics:
```bash
curl http://localhost:8005/chat/cache/stats
```

## Advanced Logging and Error Handling

The agent system includes comprehensive logging of all LLM interactions:

- **Detailed Tool Usage Logging**: All tool invocations are captured with input/output parameters
- **Chain Execution Tracing**: Logs full LLM reasoning chain execution
- **Structured JSON Logs**: Stored in `app/prompt_logs` for easy analysis
- **Message Content Preservation**: Full message contexts are preserved for debugging

The chat service now handles all response modes more reliably:

- **No-RAG Mode Fix**: Resolved errors in "no_rag" mode
- **Null-Safety**: Added comprehensive null checks
- **Primary Response Selection**: Improved selection logic
- **Error Resilience**: Graceful handling of incomplete responses

These improvements ensure a consistent and error-free experience across all response modes.

## Testing

The system now features enhanced testing capabilities for evaluating RAG performance with advanced metrics.

### Multi-dimensional Semantic Evaluation

The testing framework evaluates responses beyond simple similarity checks:

- **Quality Metrics**: Analyzes concept coverage, semantic similarity, factual accuracy, and specificity
- **Weighted Scoring**: Configurable weights to prioritize the most important quality dimensions
- **Enhanced Comparison**: Detailed RAG vs. non-RAG analysis with value ratings (High/Medium/Low/None/Negative)
- **LLM-based Analysis**: Sophisticated LLM evaluation for complex cases with strength/weakness identification
- **Statistical Reporting**: Comprehensive reports on RAG effectiveness across test cases

### Running Tests

```bash
# Run full test suite with enhanced evaluation
python -m app.services.chat_evaluator http://localhost:8005 app/services/chattests.csv 0.7

# Run a single test case
python -m app.services.chat_evaluator http://localhost:8005 app/services/chattests_single.csv 0.7

# Run batch mode with statistical analysis
python -m app.services.chat_evaluator http://localhost:8005 app/services/chattests.csv 0.7 batch
```

The tests generate detailed CSV reports with response quality metrics and RAG value assessments, making it easier to identify where retrieval augmentation provides the most benefit.

### End-to-End Tests
```bash
python -m app.tests.e2etest_rag
```

## License
This project is licensed under the [MIT License](LICENSE).

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
