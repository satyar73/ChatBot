from pydantic import BaseModel
from typing import Dict, List, TypedDict, Literal, Union, Optional

class ChatTestRequest(BaseModel):
    prompt: str
    expected_result: str
    similarity_threshold: float = 0.5
    test_id: Optional[str] = None

class ChatTestResponse(BaseModel):
    test_id: str
    prompt: str
    expected_result: str
    actual_result: str
    passed: bool
    reasoning: str
    similarity_score: float
    detailed_analysis: Dict

class ChatBatchTestRequest(BaseModel):
    csv_file: str
    similarity_threshold: float = 0.7

class ChatBatchTestResponse(BaseModel):
    total_tests: int
    passed: int
    failed: int
    pass_rate: float
    output_file: str
    rag_report_file: Optional[str] = None  # Added this field
    results: List[Dict]  # Changed to List[Dict] to handle raw dict results

class ChatMarketingTestState(TypedDict):
    prompt: str
    expected_result: str
    actual_result: Optional[str]
    similarity_threshold: float
    quick_test_results: Optional[Dict]
    llm_test_results: Optional[Dict]
    advanced_test_results: Optional[Dict]
    passed: bool
    final_reasoning: Optional[str]
    messages: List  # Required for MessagesState

class RouterOutput(TypedDict):
    """Router decision for next step in workflow"""
    reasoning: str
    next: Union[
        Literal["quick_test"],
        Literal["llm_test"],
        Literal["advanced_test"],
        Literal["FINISH"]
    ]

class ChatLLMTestOutput(TypedDict):
    semantic_score: float
    marketing_accuracy: float
    key_differences: List[str]
    would_mislead_marketer: str
    overall_assessment: str
    pass_fail: str

class ChatAdvancedTestOutput(TypedDict):
    attribution_model_accuracy: float
    technical_precision: float
    measurement_accuracy: float
    best_practices_score: float
    decision_impact: str
    specific_errors: List[str]
    overall_assessment: str
    final_verdict: str
