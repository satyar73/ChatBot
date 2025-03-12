# ChatBot Example - RAG-Enabled Marketing Assistant

## Overview

This is a RAG-enabled chatbot system designed for marketing professionals and analytics experts. The chatbot helps
answer questions about marketing services, blogs, products, and attribution strategies using specialized knowledge
bases.

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

There are multiple ways to run the application:

### Option 1: Using Docker (Recommended)

The easiest way to run the application is using Docker Compose:

1. Make sure you have Docker and Docker Compose installed on your system
2. Copy the `.env.example` file to `.env` and fill in your API keys:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```
3. Build and start the containers:
   ```bash
   docker-compose up -d
   ```
4. The application will be available at:
    - Frontend: http://localhost (this will automatically communicate with the backend)
    - Backend API: http://localhost:8005 (also accessible directly for testing)

### How the Dockerized App Works

In the containerized setup:

1. The React frontend runs in a container with Nginx, accessible at http://localhost
2. The Python backend runs in a separate container, accessible at http://localhost:8005
3. When the frontend makes API calls like `/chat` or `/index`, the Nginx server:
    - Serves the static React files for frontend routes
    - Proxies API requests to the backend container using Docker's internal DNS
4. Docker Compose sets up an internal network where "backend" hostname resolves to the backend container

To stop the containers:

```bash
docker-compose down
```

To view logs:

```bash
docker-compose logs -f
```

### Option 2: From the project root directory

```bash
# Run the convenience script
python run.py

# Or use Python module
python -m app

# Or use uvicorn directly
uvicorn app:app --port 8005 --reload
```

### Option 3: From inside the app directory

```bash
cd app

# Use the local runner script
python run_local.py

# Or use uvicorn directly
uvicorn main:app --port 8005 --reload
```

The API will be available at `http://localhost:8005`

### API Endpoints:

- `/chat`: Submit chat messages
- `/health`: Check server status
- Various indexing endpoints for document management

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

### Managing the Cache

The cache can be managed through the API or using command-line tools:

#### Clearing the Cache (API)

To clear the entire cache through the API:

```bash
curl -X DELETE http://localhost:8005/chat/cache
```

To clear only entries older than a specific number of days:

```bash
curl -X DELETE "http://localhost:8005/chat/cache?older_than_days=7"
```

#### Viewing Cache Statistics

To view cache performance statistics:

```bash
curl http://localhost:8005/chat/cache/stats
```

#### Using Python to Clear the Cache

```python
import requests

# Clear all cache entries
requests.delete("http://localhost:8005/chat/cache")

# Clear entries older than 7 days
requests.delete("http://localhost:8005/chat/cache", params={"older_than_days": 7})
```

#### Programmatically from Frontend

The frontend includes API methods to clear the cache:

```javascript
import {indexApi} from './services/api';

// Clear all cache
await indexApi.clearCache();

// Clear cache entries older than 7 days
await indexApi.clearCache(7);

// Get cache statistics
const stats = await indexApi.getCacheStats();
console.log(stats);
```

#### Command-line Cache Management Tool

The project includes a command-line tool for managing the cache:

```bash
# Navigate to the tools directory
cd tools

# View cache statistics
./manage_cache.py stats

# Clear all cache entries
./manage_cache.py clear

# Clear entries older than 7 days
./manage_cache.py clear 7
```

This tool is useful for regular maintenance or for automated cleanup scripts.

## API Endpoints

- **Chat**: `/chat` - Main chat endpoint with RAG capabilities
- **Testing**: `/chat/test` - Run semantic similarity tests
- **Cache Stats**: `/chat/cache/stats` - View cache performance metrics
- **Clear Cache**: `/chat/cache` - Clear cache entries

## License

This project is licensed under the [MIT License](LICENSE).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.