# Testing Service Documentation

Documentation for the ChatBotExample's Testing Service, which evaluate and compare RAG and non-RAG chat responses.

## 1. Overview
The Testing Service evaluates the quality and accuracy of chat responses. The primary purpose is to 
compare RAG (Retrieval-Augmented Generation) and non-RAG responses.

It consists of two main components:
- **ChatTestService**: Core service for evaluating and comparing responses
- **ChatTesterCLI**: Command-line interface for running tests

These components support:
- Individual test execution with detailed analysis
- Batch testing from CSV files
- Semantic similarity evaluation
- LLM-based qualitative assessment
- Enhanced evaluation with multiple metrics
- RAG vs. non-RAG comparison

## 2. ChatTestService
The `ChatTestService` class is the core component that evaluates chat responses using multiple methods.

### 2.1 Architecture
The service is implemented as a set of agents using a directed graph workflow built with LangGraph to
process responses through multiple evaluation steps:

1. **Response Generation**: Obtains both RAG and non-RAG responses
then follows a graph approach to  
2. **Basic Similarity Testing**: Checks text similarity metrics
3. **LLM Evaluation**: Uses language models for qualitative assessment
4. **Enhanced Evaluation**: Applies multiple specialized metrics
5. **Comparison**: Compares RAG and non-RAG responses

### 2.2 Key Methods
#### `run_test(request: ChatTestRequest) -> ChatTestResponse`
Runs a single test on a prompt/expected result pair with RAG comparison.

**Parameters**:
- `request`: ChatTestRequest object with prompt, expected result, and similarity threshold

**Returns**:
- ChatTestResponse object with test results and detailed analysis

#### `run_batch_test(csv_file: str, similarity_threshold: float) -> ChatBatchTestResponse`
Runs tests from a CSV file with multiple test cases.

**Parameters**:
- `csv_file`: Path to CSV file containing test cases
- `similarity_threshold`: Threshold for similarity score to consider a test passed

**Returns**:
- ChatBatchTestResponse with batch results and statistics

#### `start_node(state: DualState) -> DualState`
GraphNode that calls MSquared API to get both RAG and non-RAG responses.

**Parameters**:
- `state`: Current state of the test

**Returns**:
- Updated state with RAG and non-RAG responses

#### `evaluate_rag_node(state: DualState) -> DualState`
Evaluates RAG response with quick similarity tests.

**Parameters**:
- `state`: Current state of the test

**Returns**:
- Updated state with RAG evaluation results

#### `evaluate_no_rag_node(state: DualState) -> DualState`
Evaluates non-RAG response with quick similarity tests.

**Parameters**:
- `state`: Current state of the test

**Returns**:
- Updated state with non-RAG evaluation results

#### `llm_evaluate_rag_node(state: DualState) -> DualState`
Uses LLM to evaluate RAG response that failed the quick check.

**Parameters**:
- `state`: Current state of the test

**Returns**:
- Updated state with LLM-based RAG evaluation

#### `llm_evaluate_no_rag_node(state: DualState) -> DualState`
Uses LLM to evaluate non-RAG response that failed the quick check.

**Parameters**:
- `state`: Current state of the test

**Returns**:
- Updated state with LLM-based non-RAG evaluation

#### `enhance_evaluation_node(state: DualState) -> DualState`
Applies enhanced evaluation metrics to both RAG and non-RAG responses.

**Parameters**:
- `state`: Current state of the test

**Returns**:
- Updated state with enhanced evaluation metrics

#### `compare_node(state: DualState) -> DualState`
Compares RAG and non-RAG responses to determine RAG value.

**Parameters**:
- `state`: Current state of the test

**Returns**:
- Updated state with comparison results

### 2.3 Evaluation Metrics

The service uses multiple evaluation metrics:

1. **Basic Similarity**:
   - Text similarity (cosine similarity)
   - Jaccard similarity (word overlap)
   - N-gram overlap (bigram, trigram)
   
2. **LLM Evaluation**:
   - Semantic score (0-10)
   - Marketing accuracy (0-10)
   - Key differences identification
   - Pass/fail judgment
   
3. **Enhanced Evaluation**:
   - Concept coverage (key concept inclusion)
   - Semantic similarity (understanding)
   - Factual accuracy (correct information)
   - Specificity (details, numbers, examples)
   
4. **RAG Value Rating**:
   - High: RAG significantly outperforms non-RAG
   - Medium: RAG moderately outperforms non-RAG
   - Low: RAG slightly outperforms non-RAG
   - None: RAG and non-RAG perform similarly
   - Negative: Non-RAG outperforms RAG

### 2.4 Configuration
The ChatTestService uses configuration from:
- `app.config.chat_test_config.ChatTestConfig`: Contains LLM settings and prompt templates

## 3. ChatTesterCLI
The `ChatTesterCLI` class provides a command-line interface for running tests.

### 3.1 Key Methods
#### `__init__(api_url: str, csv_path: str, similarity_threshold: float)`

Initializes the chat tester CLI.

**Parameters**:
- `api_url`: Base URL for the API
- `csv_path`: Path to CSV file containing test cases
- `similarity_threshold`: Threshold for similarity score to consider a test passed

#### `async load_test_cases() -> pd.DataFrame`
Loads test cases from the CSV file.

**Returns**:
- DataFrame containing test cases

#### `async run_single_test(prompt: str, expected_result: str) -> Dict`
Runs a single test and returns results as a dictionary.

**Parameters**:
- `prompt`: The test prompt
- `expected_result`: The expected result

**Returns**:
- Dictionary with test results

#### `async run_all_tests() -> pd.DataFrame`

Runs all test cases and returns results as a DataFrame.

**Returns**:
- DataFrame with test results

#### `async run_batch_test() -> None`
Runs batch test using the TestService's batch test functionality.

**Returns**:
- None (prints results and saves to file)

### 3.2 Command-line Interface
The CLI supports the following usage:

```
python chat_evaluator.py <api_url> <csv_file> [similarity_threshold] [batch]
```

**Arguments**:
- `api_url`: The base URL for the API (e.g., "http://localhost:8005")
- `csv_file`: Path to CSV file containing test cases
- `similarity_threshold`: (Optional) Threshold for similarity score (default: 0.7)
- `batch`: (Optional) Use "batch" to enable batch processing mode

## 4. Test Models
The service uses several data models:

### 4.1 Input Models
#### `ChatTestRequest`

Represents a single test request.

**Fields**:
- `prompt`: The user prompt to test
- `expected_result`: The expected response
- `similarity_threshold`: Score threshold for passing (0.0-1.0)
- `test_id`: Optional test identifier

#### `ChatBatchTestRequest`
Represents a batch test request.

**Fields**:
- `csv_file`: Path to CSV file with test cases
- `similarity_threshold`: Score threshold for passing

### 4.2 Output Models
#### `ChatTestResponse`

Represents a single test result.

**Fields**:
- `test_id`: Test identifier
- `prompt`: The original prompt
- `expected_result`: The expected result
- `actual_result`: The actual result
- `passed`: Whether the test passed
- `reasoning`: Explanation of the test result
- `similarity_score`: Calculated similarity score
- `detailed_analysis`: Detailed information about the test

#### `ChatBatchTestResponse`
Represents batch test results.

**Fields**:
- `total_tests`: Number of tests run
- `passed`: Number of tests passed
- `failed`: Number of tests failed
- `pass_rate`: Percentage of tests passed
- `output_file`: Path to output CSV file
- `results`: List of individual test results

### 4.3 State Models
#### `DualState`

Internal state model for the testing workflow.

**Fields**:
- Test input information (prompt, expected result)
- RAG test results and evaluation
- Non-RAG test results and evaluation
- Enhanced evaluation metrics
- Comparison data
- Workflow control fields

## 5. Usage Examples
### 5.1 Running a Single Test

```python
from app.services.chat_test_service import ChatTestService
from app.models.chat_test_models import ChatTestRequest

# Initialize test service
test_service = ChatTestService(chatbot_api_url="http://localhost:8005")

# Create a test request
request = ChatTestRequest(
   prompt="What is marketing attribution?",
   expected_result="Marketing attribution is the process of identifying which marketing actions contribute to sales or conversions.",
   similarity_threshold=0.7
)

# Run the test
result = await test_service.run_test(request)

# Check if the test passed
if result.passed:
   print("Test passed with similarity score:", result.similarity_score)
else:
   print("Test failed:", result.reasoning)

# Access detailed analysis
rag_score = result.detailed_analysis["rag_test"]["weighted_similarity"]
no_rag_score = result.detailed_analysis["no_rag_test"]["weighted_similarity"]
print(f"RAG score: {rag_score}, Non-RAG score: {no_rag_score}")

# Check RAG value rating
value_rating = result.detailed_analysis["comparison"]["rag_value_rating"]
print(f"RAG value rating: {value_rating}")
```

### 5.2 Running Batch Tests from CSV

```python
import asyncio
from app.services.chat_test_service import ChatTestService


async def run_batch_tests():
   # Initialize test service
   test_service = ChatTestService(chatbot_api_url="http://localhost:8005")

   # Run batch test
   results = await test_service.run_batch_test(
      csv_file="tests/test_cases.csv",
      similarity_threshold=0.7
   )

   # Print summary
   print(f"Ran {results.total_tests} tests")
   print(f"Passed: {results.passed} ({results.pass_rate:.2f}%)")
   print(f"Failed: {results.failed}")
   print(f"Results saved to: {results.output_file}")


# Run the batch test
asyncio.run(run_batch_tests())
```

### 5.3 Using from Command Line

```bash
# Run tests from CSV file with default threshold
python -m app.services.chat_evaluator http://localhost:8005 app/services/chattests.csv

# Run tests with custom threshold
python -m app.services.chat_evaluator http://localhost:8005 app/services/chattests.csv 0.8

# Run tests in batch mode
python -m app.services.chat_evaluator http://localhost:8005 app/services/chattests.csv 0.7 batch
```

## 6. Test CSV Format

The service expects test cases in a CSV file with the following format:

```csv
Prompt,Expected Result
"What is marketing attribution?","Marketing attribution is the process of identifying which marketing actions contribute to sales or conversions."
"Explain incrementality testing","Incrementality testing is a method to measure the true impact of marketing efforts by comparing test and control groups."
```

Required columns:
- `Prompt`: The test query
- `Expected Result`: The expected response

## 7. Output Files

The service generates two types of output files:

### 7.1 Semantic Test Results

Contains detailed results for each test case:
- Basic test information
- RAG and non-RAG responses
- Similarity scores
- Pass/fail status
- Missing concepts
- LLM evaluation results

### 7.2 RAG Comparison Report

Focuses on comparing RAG and non-RAG performance:
- Score differences
- Value rating
- Strengths and weaknesses
- Performance statistics

## 8. Dependencies

The Testing Service components depend on:
- **LangGraph**: For workflow orchestration
- **LangChain**: For LLM integration
- **OpenAI API**: For LLM evaluation
- **Pandas**: For data processing and CSV handling
- **MSquaredClient**: Internal client for API access
- **SimilarityEngines**: Internal similarity calculation

## 9. Troubleshooting

### 9.1 Common Issues

- **API Connection Errors**: Verify the API URL and ensure the service is running
- **CSV Format Issues**: Check that CSV file has required columns
- **LLM Evaluation Errors**: Check API keys and rate limits
- **Low Similarity Scores**: Adjust the similarity threshold or review expected results

### 9.2 Logging

The service uses logging to track its operation:
- `DEBUG`: Detailed workflow information
- `INFO`: General progress updates
- `WARNING`: Potential issues
- `ERROR`: Errors that impact testing

## 10. Extending the Testing Framework

To extend the testing framework:

### 10.1 Adding New Metrics

1. Define the metric calculation in a method
2. Add it to the `_evaluate_response_quality` method
3. Update the `evaluation_weights` dictionary

### 10.2 Adding New Evaluation Nodes

1. Create a new node method in `ChatTestService`
2. Add it to the graph in `build_test_graph`
3. Update the router to handle the new node

### 10.3 Creating Custom Test Types

1. Define new request and response models
2. Create specialized workflow nodes
3. Build a custom graph for the new test type