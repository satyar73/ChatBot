# Chat Service Documentation

Documentation for the ChatBotExample's Chat Service components,
including the core chat functionality, caching, and evaluation capabilities.

## 1. Overview
The Chat Service is the central component of the ChatBotExample system that manages conversations
between users and the underlying language models, with both RAG (Retrieval-Augmented Generation)
and non-RAG capabilities. It includes:
- **Core ChatService**: Manages chat sessions, processes user queries, and generates responses
- **QueryRewriter**: Enhances user queries to improve RAG retrieval
- **ChatCacheService**: Caches responses to improve performance and reduce API costs
- **AgentService**: Provides direct interaction with agents without session management

## 2. Core ChatService
The `ChatService` class manages chat interactions between users and the underlying LLM agents.

### 2.1 Key Components
- **Session Management**: Maintains chat histories for different user sessions
- **Query Routing**: Routes queries to appropriate agents (RAG, non-RAG, or database) using pattern matching
- **Query Enhancement**: Uses the `EnhancementService` with the following strategies to enhance RAG retrieval:
  - Abbreviation expansion for marketing terms
  - Synonym addition for improved matching
  - Technical term handling with definitions
  - Query broadening techniques
  - Contextual analysis from conversation history
- **Response Quality Assessment**: Evaluates RAG responses for adequacy and tries alternative query formulations when needed
- **Detailed Logging**: Provides logs of routing decisions and query transformations
- **Response Processing**: Handles both RAG and non-RAG responses and combines them with sources
- **Caching Integration**: Uses the cache service to store and retrieve responses with performance tracking

### 2.2 Key Methods

#### `chat(data: Message) -> ResponseMessage`
Processes a user chat message and returns a response with enhanced query rewriting and routing intelligence. This is the main entry point for client requests.
**Parameters**:
- `data`: A Message object containing:
  - `message`: The user's query text
  - `session_id`: Identifier for the user's session
  - `mode`: Response mode ("rag", "no_rag", or both)
  - `system_prompt`: Optional custom system prompt

**Flow**:
1. Generates a unique hash for the query based on content, session, and system prompt
2. Checks for special testing flags in the query (e.g., "test_routing:")
3. Checks cache for an existing response (skipped for database queries or test mode)
4. If cache miss:
   - Analyzes query with enhanced pattern matching to determine routing
   - Routes to database agent for data/analytics queries with detailed logging
   - For RAG queries, uses improved query rewriting with retry logic:
     - Tries multiple query formulations
     - Special handling for technical terms
     - Evaluates response quality to determine if alternative queries are needed
   - Optionally generates non-RAG response for comparison
5. Adds the response to the session chat history
6. Extracts and formats sources from RAG response
7. Caches the response for future use including query reformulations
8. Tracks and logs cache access statistics and response times
9. Returns formatted response with RAG/non-RAG outputs, sources, and query rewriting metadata

**Returns**:
- `ResponseMessage` object with:
  - RAG-based response
  - Non-RAG response (if requested)
  - Sources from retrieval
  - Metadata about the response process including alternative queries tried

#### `_execute_rag_with_retry(query, history, max_attempts, custom_system_prompt) -> Tuple[Dict, List[str]]`
Executes RAG agent with enhanced retry logic, query reformulation, and special handling for technical terms.
**Parameters**:
- `query`: The original user query
- `history`: Chat history
- `max_attempts`: Maximum number of query attempts (default: 3)
- `custom_system_prompt`: Optional custom system prompt

**Flow**:
1. Uses the EnhancementService's try_alternative_queries method to manage query reformulation
2. Generates alternative query formulations with context awareness
3. Special handling for technical marketing terms by trying exact term queries first
4. Evaluates response quality using the EnhancementService's _is_empty_or_inadequate_response
5. If the response appears to lack information or contains "not found" phrases, tries alternative formulations
6. Returns the best response and list of all queries tried

**Technical Term Handling**:
- Identifies technical marketing terms (e.g., "advanced attribution multiplier")
- Tries exact term queries first for precise matching
- Adds technical definitions to improve retrieval accuracy
- Extracts context topics from recent conversation history

**Returns**:
- Tuple containing (best_response, queries_tried)

#### `_is_database_query(query) -> bool`
Determines if a query should be routed to the database agent using an enhanced rule-based approach with exclusion and inclusion patterns.
**Parameters**:
- `query`: The user's input text

**Flow**:
1. First checks against exclusion patterns (high-priority rules that prevent database routing)
   - Excludes queries about incrementality tests, prospecting, marketing strategy, etc.
2. If no exclusions match, checks against database keyword patterns
   - Matches database/analytics terms, specific metrics mentions, etc.
3. Additionally checks against query patterns that suggest data requests
   - Patterns like "how many customers", "show me the data", etc.
4. Provides detailed logging for pattern matching decisions

**Returns**:
- `True` if the query appears to be a database/analytics query
- `False` otherwise

**Note**: This is a rule-based approach that will eventually be replaced with an LLM-based classifier for more accurate routing.

### 2.3 Configuration
The ChatService relies on configuration from:
- `app.config.chat_config.ChatConfig`: Contains system prompts, API settings
- `app.config.cache_config`: Contains cache configuration

## 3. Enhancement Service
The `EnhancementService` class enhances user queries, prompts, and content processing to improve RAG retrieval quality and response generation.

### 3.1 Features
- **Query Enhancement**:
  - Abbreviation expansion for marketing terms (e.g., "mmm" → "marketing mix modeling")
  - Addition of relevant synonyms to improve matching
  - Special handling for technical marketing terms with definitions
  - Query broadening by removing specific constraints
  - Extraction of context topics from conversation history
  - Semantic similarity filtering for diverse query alternatives
- **Content Enhancement**:
  - Attribution metadata enrichment for indexed content
  - Keyword extraction and categorization
  - Technical term identification and definitions
  - Enhanced embedding prompts for technical content
  - Q&A pair processing that preserves question-answer relationships
- **Prompt Enhancement**:
  - Extraction of key concepts from expected answers
  - Enhanced system prompts with concept guidance
  - Image content analysis with vision models

### 3.2 Key Methods for Query Enhancement

#### `enhance_query(original_query, conversation_context) -> Dict[str, Any]`
Generates enhanced query and alternative formulations to improve retrieval, with context awareness.

**Parameters**:
- `original_query`: The original user query
- `conversation_context`: Optional conversation history

**Returns**:
- Dictionary with:
  - `original_query`: The original query
  - `enhanced_query`: The enhanced query with technical definitions and expansions
  - `alt_queries`: List of alternative query formulations
  - `intent`: Detected query intent
  - `qa_match`: Information about matching Q&A pairs
  - `context_topics`: Topics extracted from conversation history
  - `related_questions`: Related questions from the knowledge base

#### `expand_abbreviations(query) -> str`
Expands common marketing abbreviations in the query using a comprehensive dictionary of terms.

**Parameters**:
- `query`: User query text

**Returns**:
- Query with expanded abbreviations

#### `add_synonyms(query) -> str`
Adds relevant synonyms to the query to improve matching based on a curated synonym mapping.

**Parameters**:
- `query`: User query text

**Returns**:
- Query with added synonyms

#### `create_broader_query(query) -> str`
Creates a more general version of the query by removing specific constraints and extracting core marketing concepts.

**Parameters**:
- `query`: User query text

**Returns**:
- Broader version of the query

#### `try_alternative_queries(original_query, process_function, is_adequate_function, history, max_attempts) -> tuple`
Tries multiple query formulations, evaluating responses for adequacy.

**Parameters**:
- `original_query`: The original user query
- `process_function`: Async function that processes each query (e.g., sends to an agent)
- `is_adequate_function`: Function that determines if a response is adequate
- `history`: Optional conversation history
- `max_attempts`: Maximum number of queries to try (default: 3)

**Returns**:
- Tuple of (best_response, queries_tried)

### 3.3 Key Methods for Content Enhancement

#### `extract_keywords_from_qa() -> Dict[str, List[str]]`
Extracts keywords from Q&A pairs to use for tagging content.

**Returns**:
- Dictionary mapping keyword categories to related terms

#### `create_embedding_prompt(text, metadata) -> str`
Creates an optimized prompt for embedding that highlights attribution terms and technical concepts.

**Parameters**:
- `text`: Original text to embed
- `metadata`: Metadata associated with the text

**Returns**:
- Enhanced prompt for embedding with additional context

#### `enrich_attribution_metadata(content) -> Dict[str, Any]`
Analyzes content for attribution terminology and creates enhanced metadata.

**Parameters**:
- `content`: Markdown or text content to analyze

**Returns**:
- Dictionary of attribution-related metadata

#### `enhance_records_with_keywords(records, keyword_map) -> List[Dict[str, Any]]`
Enhances content records with keywords based on content analysis.

**Parameters**:
- `records`: List of content records
- `keyword_map`: Dictionary of keywords and related terms

**Returns**:
- Enhanced records with keywords added

## 4. ChatCacheService
The `ChatCacheService` provides caching functionality to avoid redundant API calls.

### 4.1 Key Components
- **SQLite Database**: Stores cached responses and usage statistics
- **Hashing Algorithm**: Creates unique identifiers for queries
- **TTL Management**: Manages cache entry expiration
- **Statistics Collection**: Tracks cache hit rates and performance metrics

### 4.2 Key Methods
#### `generate_query_hash(query, history, session_id, system_prompt) -> str`

Generates a hash to uniquely identify a query with its context.

**Parameters**:
- `query`: The user's query text
- `history`: Optional chat history
- `session_id`: Optional session ID
- `system_prompt`: Optional custom system prompt

**Returns**:
- String hash that uniquely identifies this query in its context

#### `get_cached_response(query_hash) -> Tuple[Optional[Dict], bool]`
Retrieves a cached response for the given query hash.

**Parameters**:
- `query_hash`: Hash identifying the query

**Returns**:
- Tuple containing (cached_response, cache_hit_bool)

#### `cache_response(query_hash, user_input, rag_response, no_rag_response, sources, system_prompt) -> bool`
Caches a response for future retrieval.

**Parameters**:
- `query_hash`: Hash identifying the query
- `user_input`: Original user input
- `rag_response`: Response with RAG
- `no_rag_response`: Response without RAG
- `sources`: Optional list of sources
- `system_prompt`: Optional custom system prompt

**Returns**:
- Boolean indicating success/failure

#### `get_cache_stats() -> Dict`
Get statistics about the cache usage.

**Returns**:
- Dictionary with cache statistics including:
  - Total entries
  - Total hits
  - Hit rate percentage
  - Average hit/miss times
  - Most frequent queries
  - Cache settings information

#### `clear_cache(older_than_days) -> int`
Clears cache entries optionally based on age.

**Parameters**:
- `older_than_days`: Optional, only clear entries older than this many days

**Returns**:
- Number of entries cleared

### 4.3 Configuration

The ChatCacheService relies on:
- `app.config.cache_config`: Contains cache settings like:
  - `CACHE_ENABLED`: Toggle for cache functionality
  - `CACHE_TTL`: Time-to-live for cache entries
  - `CACHE_SIZE_LIMIT`: Maximum number of entries
  - `MAX_HISTORY_FOR_HASH`: How much chat history to include in hash
  - `CONSIDER_SESSION_IN_HASH`: Whether to include session ID in cache key
  - `CACHE_DB_PATH`: Path to SQLite database

## 5. AgentService

The `AgentService` provides a simplified interface for direct agent interactions without chat history management.

### 5.1 Key Methods

#### `process_query(query, history, use_rag, use_dual_response) -> ResponseMessage`

Processes a user query through the appropriate agent(s).

**Parameters**:
- `query`: The user's query text
- `history`: Optional chat history
- `use_rag`: Whether to use the RAG-enabled agent
- `use_dual_response`: Whether to generate both RAG and non-RAG responses

**Returns**:
- A ResponseMessage object containing the response(s) and sources

## 6. Usage Examples

### 6.1 Basic Chat Interaction

```python
from app.services.chat_service import ChatService
from app.services.enhancement_service import enhancement_service
from app.models.chat_models import Message

# Initialize the chat service
chat_service = ChatService()

# Create a message
message = Message(
    message="What is marketing attribution?",
    session_id="user_123",
    mode="rag"  # Use RAG mode
)

# Get response with enhanced query processing
response = await chat_service.chat(message)

# Access the response text
rag_response = response.rag_response

# Access sources
sources = response.sources

# Get alternative queries that were tried
alt_queries = response.metadata.get("queries_tried", [])
```

### 6.2 Using Custom System Prompt

```python
# Create a message with custom system prompt
message = Message(
    message="What is marketing attribution?",
    session_id="user_123",
    mode="rag",
    system_prompt="You are a marketing expert specializing in attribution models. Provide concise, technical answers."
)

# Get response with custom system prompt
response = await chat_service.chat(message)
```

### 6.3 Getting Both RAG and Non-RAG Responses

```python
# Request both RAG and non-RAG responses
message = Message(
    message="What is marketing attribution?",
    session_id="user_123",
    mode="both"  # Get both RAG and non-RAG responses
)

# Get dual response
response = await chat_service.chat(message)

# Access both responses
rag_response = response.rag_response
no_rag_response = response.no_rag_response
```

### 6.4 Cache Management

```python
from app.services.cache_service import chat_cache

# Get cache statistics
stats = chat_cache.get_cache_stats()
print(f"Cache hit rate: {stats['hit_rate_percent']}%")
print(f"Total hits: {stats['total_hits']}")

# Clear old cache entries
cleared = chat_cache.clear_cache(older_than_days=7)
print(f"Cleared {cleared} old cache entries")
```

## 7. Dependencies

The Chat Service components depend on:

- **LangChain**: For agent construction and execution
- **OpenAI API**: For language model access
- **Pinecone**: For vector search in RAG mode
- **SQLite**: For response caching

## 8. Troubleshooting

### 8.1 Common Issues
- **Missing Responses**: Check that the appropriate mode is set ("rag", "no_rag", or leave empty for both)
- **Cache Misses**: Verify that cache is enabled and that the query hash is consistent
- **RAG Quality Issues**: Try different system prompts or use query rewriting

### 8.2 Logging

The Chat Service components use the `app.utils.logging_utils` module for logging:

- `DEBUG`: Detailed information about query processing
- `INFO`: High-level flow information
- `WARNING`: Potential issues that don't prevent operation
- `ERROR`: Issues that prevent successful response generation

Logs can be used to debug issues with query routing, agent selection, and caching behavior.