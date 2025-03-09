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
   git clone <repository-url>
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

## Usage
1. Start the server:
   ```bash
   python -m app.main
   ```

2. The API will be available at `http://localhost:8005`

3. API Endpoints:
   - `/chat`: Submit chat messages
   - `/health`: Check server status
   - Various indexing endpoints for document management

## Development
- **Environment**: Set `ENVIRONMENT=development` for development mode with auto-reload
- **Logging**: Adjust log levels in `app/config/logging_config.py`
- **Testing**: Run test suite with included chat test data

## License
This project is licensed under the [MIT License](LICENSE).

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.