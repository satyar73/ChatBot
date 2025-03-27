# Query Path in RAG System

This document explains the complete query processing flow in our RAG (Retrieval-Augmented Generation) system, from the initial user question to the final response with citations.

## Overview

The system processes user queries through several stages:
1. Query classification
2. Query enhancement and rewriting
3. Vector search retrieval
4. LLM response generation
5. Response evaluation and retry mechanism

## Complete Flow with Example

Let's trace a real-world example query through our system:

### 1. Initial User Query

Original question:
```
Do platforms typically under credit or over credit themselves in self attribution frameworks?
```

### 2. Query Classification

The system first determines .which system should handle this query. Since this question is about attribution concepts rather than metrics or reports, it's classified as a RAG query.

```python
# Simplified code from chat_service.py
strategy = ResponseStrategy.get_strategy(actual_query, mode, self)
rag_response, no_rag_response, queries_tried = await strategy.execute(
            actual_query,
            chat_history,
            custom_system_prompt,
            prompt_style
        )
    # Route to RAG processing path
```

### 3. Query Enhancement and Rewriting

The original query is processed by the `EnhancementService` class to generate alternative formulations that might better match documents in our knowledge base:

```python
# From chat_service.py
enhanced_query_data = self.enhancement_service.enhance_query(original_query, history)
alt_queries = enhanced_query_data.get("alt_queries", [original_query])
```

For our example, the enhancement service transforms:  
**Original**: "Do platforms typically under credit or over credit themselves in self attribution frameworks?"  
**Enhanced**: "self attribution frameworks platform bias"

The enhancement process applies several techniques:
- Removes question phrasing
- Focuses on core concepts
- Expands abbreviations (if any)
- Adds relevant synonyms
- Considers conversation context
- Identifies technical terms and adds definitions
- May generate multiple alternative queries

### 4. Document Retrieval

The rewritten query is sent to Pinecone through the `search_rag_docs` tool:

```json
{"event": "tool_start", "timestamp": "2025-03-19T08:00:46.661629", "tool": "search_rag_docs", "input": "{'query': 'self attribution frameworks platform bias'}"}
```

This retrieval process is configured in `gpt_tools.py`:

```python
retriever = vectorstore.as_retriever(
    search_type=cls.config.RETRIEVER_CONFIG["search_type"],
    search_kwargs={
        "k": cls.config.RETRIEVER_CONFIG["k"],
        "fetch_k": cls.config.RETRIEVER_CONFIG["fetch_k"],
        "lambda_mult": cls.config.RETRIEVER_CONFIG["lambda_mult"]
    }
)
```

### 5. Pinecone Response

Pinecone returns multiple semantically relevant documents to the query. The vector search is configured with:
- `search_type: "mmr"` (Maximum Marginal Relevance for diversity)
- `k: 5` (Return 5 documents)
- `fetch_k: 20` (Fetch 20 candidates before filtering)
- `lambda_mult: 0.7` (Balance between relevance and diversity)

Here's a complete example of all documents returned for our query:

```
<DOC>
# Title: April 2025: Executive Masterclass
*Source*: (https://msquared.club/products/live-april-2025-advanced-attribution-crash-course-for-executives)
## Content: Marketing attribution context: **Live advanced attribution crash course broken down into 2 sessions over the course of 2 days, aptly designed for Executives.**  
  
Format: Live instructor-led classes   
  
Lecture : April 28 & 30th - 1100 - 1430 EST
</DOC>

<DOC>
# Title: MEET Arun Rajagopalan
*Source*: (https://msquared.club/products/meet-arun-rajagopalan)
## Content: ### Measurement Technologist

#### ***Measured, Wealth Engine***
</DOC>

<DOC>
# Title: Decoding Sensitivity Analysis: Optimizing Marketing Attribution Strategies
*Source*: (https://msquared.club/blogs/attribution-today/decoding-sensitivity-analysis-optimizing-marketing-attribution-strategies)
## Content: Technical marketing terms context:
advanced attribution multiplier: A coefficient used in advanced attribution to adjust the credit given to marketing channels based on their true incremental value
attribution multiplier: A factor used to adjust attribution models to reflect true marketing contribution

In today's marketing landscape, platforms often over-credit themselves when using self-attribution frameworks. This bias occurs because these platforms have incentives to show their effectiveness, sometimes leading to inflated performance metrics. 

When platforms handle both the delivery of ads and the measurement of their performance, they typically attribute conversions that may have happened regardless of ad exposure. Studies comparing self-attribution data with controlled incrementality tests frequently show that platforms overstate their impact by 20-40%.

The most accurate measurement comes from independent, cross-platform attribution systems that apply advanced attribution multipliers to adjust for this inherent bias.

#### Advanced Attribution Multiplier: Reassessing Facebook's Value
</DOC>

<DOC>
# Title: Incrementality testing unlocks upper funnel branding investments for Study.com
*Source*: (https://msquared.club/blogs/attribution-today/incrementality-testing-unlocks-upper-funnel-branding-investments-for-study-com)
## Content: The test was flighted in Q4 2024, and the flight was monitored for execution aligning to the test design that was put in place. 

### **The Results:Lift Reads & Interpretations for Growth**

As with the design, at M-Squared, we take a structured process for estimating the lift reads from the test. 

**Catch the 7 minute mini course on estimating lift from the M-Squared masterclass:**

[![](https://cdn.shopify.com/s/files/1/0804/0130/1789/files/Lift_Reads_Interpretations_for_Growth-blog_da903a22-01f4-46cb-9098-71f105766dc3.png?v=1736932845)](https://www.youtube.com/watch?v=4EGT9BPhQ2k "Lift Reads &amp; Interpretations for Growth")

**After carefully estimating the lift with multiple algorithmic approaches, Study.com learned:**
</DOC>

<DOC>
# Title: MEET Ian Yung
*Source*: (https://msquared.club/products/meet-ian-yung)
## Content: ### Growth Marketing Leader

#### *Tonal, Parachute, Touch of Modern, The Black Tux*
</DOC>
```

The system typically retrieves 4-5 documents to provide sufficient context while avoiding overwhelming the LLM. As shown above, only the third document ("Decoding Sensitivity Analysis") contains directly relevant information about platform bias in self-attribution frameworks.

### 6. LLM Answer Generation

The retrieved documents, along with the original query and system instructions, are sent to the LLM (GPT-4).

The full prompt to the LLM looks like this:

```
System: 
You are a helpful website chatbot who is tasked with answering questions about MSquared.
MSquared is a community of analytics and marketing professionals committed to making
marketing attribution accessible, affordable, and effective for every brand.

QUESTION FOCUS:
- Always address the specific question directly at the beginning of your response before providing additional context
- When a question asks about a specific concept, ensure that concept is clearly defined and explained before elaborating
- For each question, identify ALL key concepts mentioned and ensure they are covered in your response
- Restate key terms from the question in your answer to ensure complete coverage
- If key information is missing from retrieved documents, acknowledge the limitation rather than inventing details
- For questions with multiple parts, enumerate each part in your answer to ensure complete coverage

INFORMATION ACCURACY AND COMPLETENESS:
- Thoroughly incorporate ALL key concepts from retrieved documents, even when synthesizing information
- Preserve specific numerical data, percentages, statistics, and metrics exactly as presented in the source material
- When source documents mention specific attribution methodologies, models, or techniques (e.g., MMM, Geotesting, incrementality testing), always include these terms in your response
- For technical concepts, maintain the precise terminology used in the source documents
- When multiple documents provide different perspectives, prioritize the most comprehensive explanation while incorporating unique insights from each source
- Include synonyms and related concepts for technical marketing terms to enhance understanding

RESPONSE STRUCTURE:
- Keep responses concise and focused. Keep the answer within 2 - 3 paragraphs.
- Start with a direct answer to the question in the first paragraph
- For technical marketing concepts, use this abbreviated structure: 1) Brief Definition, 2) Key Application, 3) Short Example
- Prioritize essential information over comprehensive coverage - focus on what the user needs to know
- Use bullet points for lists rather than lengthy paragraphs
- Avoid repeating information already mentioned
- Eliminate filler phrases and unnecessary elaboration
- When discussing attribution models, focus on key distinctions rather than exhaustive explanations
- At the end of your response, include "Learn more: [Title of Source](URL)" with the most relevant source document

CONTENT BOUNDARIES:
- Always include a hyperlink to the most relevant source document at the end of your response
- Format source links as "Learn more: [Title of Source](URL)" using the title and URL from the *Source* field
- Never create, suggest or reference links that weren't provided in the source material
- Do not use a link from one document when discussing content from another document
- Do not write or explain any code under any circumstances
- Only share MSquared-specific data and information
- For time-sensitive information like masterclass dates, direct users to the upcoming masterclass page rather than mentioning specific dates
- Avoid repeating identical information within the same response

SPECIAL SCENARIOS:
- For pricing questions: Do not provide specific amounts; instead, direct users to the product link provided in the source material
- For budget allocation questions: Provide guidance based on retrieved information, but include this disclaimer: "For optimal results, we recommend consulting with MSquared experts to discuss your specific needs before making allocation decisions."
- For technical attribution questions: If the retrieved documents mention specific models (MMM, Geo-testing, etc.), always include these in your response even if they seem technical
- When addressing platform-specific attribution (Facebook, Google, etc.), explicitly mention limitations or biases of platform self-attribution if mentioned in the source material

CONTENT COMPREHENSIVENESS:
- Double-check that your response includes all key terms mentioned in the question
- Ensure your response incorporates every key concept related to the question, even if briefly mentioned
- For questions about technical implementations or methodologies, include practical examples whenever possible
- When describing processes or frameworks, include step-by-step approaches when available in the source material
- For performance metrics or evaluation criteria, always explain both what they measure and why they're important

TECHNICAL TERMINOLOGY HANDLING:
- Answer questions about technical terms based on your understanding of marketing and attribution concepts
- Even if a specific term isn't explicitly found in the documents, provide a definition based on related concepts
- Do NOT state that a term "doesn't appear in the documents" - instead, provide your best technical explanation
- For marketing terms, draw on your knowledge of attribution, analytics, and advertising to provide helpful definitions
- Whenever possible, explain how technical terms relate to practical marketing applications and measurement
- For technical calculations or formulas, provide specific examples with numbers if available
- If you're genuinely uncertain about a very obscure term, provide related concepts but avoid disclaimers that diminish your answer's value

Always maintain the original meaning and intent of the source material while making your response cohesive and conversational.
When uncertain between being comprehensive versus concise, prioritize including all key concepts and technical terms
from the retrieved documents.

Human: Do platforms typically under credit or over credit themselves in self attribution frameworks?

Assistant: I'll help you answer that question using the relevant documents I have access to.

<documents>
<DOC>
# Title: April 2025: Executive Masterclass
*Source*: (https://msquared.club/products/live-april-2025-advanced-attribution-crash-course-for-executives)
## Content: Marketing attribution context: **Live advanced attribution crash course broken down into 2 sessions over the course of 2 days, aptly designed for Executives.**  
  
Format: Live instructor-led classes   
  
Lecture : April 28 & 30th - 1100 - 1430 EST
</DOC>

<DOC>
# Title: MEET Arun Rajagopalan
*Source*: (https://msquared.club/products/meet-arun-rajagopalan)
## Content: ### Measurement Technologist

#### ***Measured, Wealth Engine***
</DOC>

<DOC>
# Title: Decoding Sensitivity Analysis: Optimizing Marketing Attribution Strategies
*Source*: (https://msquared.club/blogs/attribution-today/decoding-sensitivity-analysis-optimizing-marketing-attribution-strategies)
## Content: Technical marketing terms context:
advanced attribution multiplier: A coefficient used in advanced attribution to adjust the credit given to marketing channels based on their true incremental value
attribution multiplier: A factor used to adjust attribution models to reflect true marketing contribution

In today's marketing landscape, platforms often over-credit themselves when using self-attribution frameworks. This bias occurs because these platforms have incentives to show their effectiveness, sometimes leading to inflated performance metrics. 

When platforms handle both the delivery of ads and the measurement of their performance, they typically attribute conversions that may have happened regardless of ad exposure. Studies comparing self-attribution data with controlled incrementality tests frequently show that platforms overstate their impact by 20-40%.

The most accurate measurement comes from independent, cross-platform attribution systems that apply advanced attribution multipliers to adjust for this inherent bias.

#### Advanced Attribution Multiplier: Reassessing Facebook's Value
</DOC>

<DOC>
# Title: Incrementality testing unlocks upper funnel branding investments for Study.com
*Source*: (https://msquared.club/blogs/attribution-today/incrementality-testing-unlocks-upper-funnel-branding-investments-for-study-com)
## Content: The test was flighted in Q4 2024, and the flight was monitored for execution aligning to the test design that was put in place. 

### **The Results:Lift Reads & Interpretations for Growth**

As with the design, at M-Squared, we take a structured process for estimating the lift reads from the test. 

**Catch the 7 minute mini course on estimating lift from the M-Squared masterclass:**

[![](https://cdn.shopify.com/s/files/1/0804/0130/1789/files/Lift_Reads_Interpretations_for_Growth-blog_da903a22-01f4-46cb-9098-71f105766dc3.png?v=1736932845)](https://www.youtube.com/watch?v=4EGT9BPhQ2k "Lift Reads &amp; Interpretations for Growth")

**After carefully estimating the lift with multiple algorithmic approaches, Study.com learned:**
</DOC>

<DOC>
# Title: MEET Ian Yung
*Source*: (https://msquared.club/products/meet-ian-yung)
## Content: ### Growth Marketing Leader

#### *Tonal, Parachute, Touch of Modern, The Black Tux*
</DOC>
</documents>
```

### 7. LLM Response

The LLM responds with an answer based on the retrieved documents, including a proper citation link as required by the system prompt:

```
Platforms typically over-credit themselves in self-attribution frameworks. This happens because they have inherent incentives to demonstrate their effectiveness to advertisers, which can lead to inflated performance metrics. When platforms control both ad delivery and performance measurement, they often attribute conversions that might have occurred naturally without the ad exposure.

Comparative studies between self-attribution data and controlled incrementality tests consistently show that platforms overstate their actual impact by approximately 20-40%. This significant discrepancy highlights why many marketers are turning to independent, cross-platform attribution systems that apply advanced attribution multipliers – coefficients specifically designed to adjust the credit given to marketing channels based on their true incremental value rather than self-reported metrics.

Learn more: [Decoding Sensitivity Analysis: Optimizing Marketing Attribution Strategies](https://msquared.club/blogs/attribution-today/decoding-sensitivity-analysis-optimizing-marketing-attribution-strategies)
```

### 8. Retry Mechanism for Inadequate Responses

If the LLM's initial response is inadequate (too short, fails to address the query properly, or contains "not found" phrases), the system includes a comprehensive retry mechanism:

```python
# From chat_service.py
original_response, queries_tried = await self.enhancement_service.try_alternative_queries(
    original_query=query,
    process_function=process_query_function,
    is_adequate_function=self._is_empty_or_inadequate_response,
    history=history,
    max_attempts=max_attempts
)
```

The system evaluates the response quality by:
1. Checking the response length
2. Using semantic similarity to detect potential hallucinations
3. Verifying that key concepts from the query are addressed
4. Detecting phrases that indicate information wasn't found
5. Analyzing technical term coverage

If the initial response is inadequate, the system tries alternative query formulations generated by the EnhancementService, including:
1. Technical term-focused queries
2. Queries with expanded abbreviations
3. Queries with added synonyms
4. Broader concept queries

The system tracks which queries were tried and returns the best response found.

## Data Extraction and Analysis

For analyzing the effectiveness of the query enhancement system, we've created a tool that extracts:
1. Original user prompts
2. Rewritten search queries
3. Returned URLs from Pinecone

This extraction script (`extract_queries_urls.py`) parses the JSONL log files, matching query events with their corresponding document retrieval events.

```python
# Example from extract_queries_urls.py
def extract_queries_and_urls():
    """Extract user prompts, search queries, and corresponding URLs from the log file."""
    log_file = '../logs/llm_prompts_responses.jsonl'
    data = []
    
    # Track state through the logs
    user_prompt = None
    current_query = None
    current_urls = []
    
    # Process each log entry
    with open(log_file, 'r') as f:
        for line in f.readlines():
            entry = json.loads(line.strip())
            
            # Extract original prompt
            if entry.get('event') == 'chain_start':
                user_prompt = entry.get('inputs', {}).get('input')
                
            # Extract rewritten query
            elif entry.get('event') == 'tool_start' and 'search' in entry.get('tool', ''):
                input_data = entry.get('input', '')
                current_query = extract_query_from_input(input_data)
                
            # Extract URLs from response
            elif entry.get('event') == 'tool_end':
                output = entry.get('output', '')
                urls = re.findall(r'https?://[^\s"\')\]]+', output)
                
                # Save data
                data.append({
                    'user_prompt': user_prompt,
                    'search_query': current_query,
                    'urls': '|'.join(urls),
                    'url_count': len(urls),
                    'timestamp': entry.get('timestamp', '')
                })
```

This extracted data helps in analyzing:
- How effectively queries are enhanced and rewritten
- Which documents are most frequently retrieved
- The effectiveness of technical term handling
- How often alternative queries are used
- Which enhancement techniques produce the best results
- The overall performance of the RAG system

## Semantic Filtering

The system uses semantic filtering in two key ways:

### Query Similarity Detection

The system prevents redundant queries when a series of similar questions are asked:

```python
# From app/utils/semantic_filtering.py
def is_semantically_similar(query1, query2, threshold=0.85):
    """Determine if two queries are semantically similar using cosine similarity."""
    similarity = SimilarityEngines.cosine_similarity_tfidf(query1, query2)
    return similarity > threshold
```

The similarity check is used in the chat service to decide whether to reuse previously retrieved documents or perform a new retrieval:

```python
# From chat_service.py
if is_semantically_similar(query, previous_query):
    # Reuse previous documents
    context_docs = previous_context_docs
else:
    # Perform new retrieval
    context_docs = self._retrieve_documents(query)
```

### Alternative Query Filtering

The EnhancementService also uses semantic filtering to ensure diverse alternative queries:

```python
# From app/services/enhancement_service.py
if getattr(self.config, 'CHAT_FEATURE_FLAGS', {}).get("semantic_similarity_filtering", False):
    # Apply semantic filtering to remove similar queries
    filtered_queries = SemanticFilter.filter_similar_queries(
        unique_queries,
        similarity_threshold=0.7  # Configurable threshold
    )
    
    # Rank by diversity (optional, helps prioritize diverse queries)
    ranked_queries = SemanticFilter.rank_queries_by_diversity(filtered_queries, original_query)
```

These optimizations prevent unnecessary vector database calls, ensure a smoother conversational experience for follow-up questions, and improve the diversity of alternative query formulations for better retrieval results.