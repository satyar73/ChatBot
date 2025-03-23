from dotenv import load_dotenv

class ChatTestConfig:


    # Default API URL
    def __init__(self):
        # Load environment variables
        load_dotenv()

        self.DEFAULT_CHATBOT_API_URL = "http://localhost:8005"

        # LLM configuration
        self.LLM_MODEL = "gpt-4o"
        self.LLM_TEMPERATURE = 0

        # Threshold for similarity comparison
        self.DEFAULT_SIMILARITY_THRESHOLD = 0.7

        # Labels/Steps for state transition
        self.START = "START"
        self.END = "END"
        self.NEXT = "next"
        self.START_GET_CHAT_RESPONSE = "start_get_chat_response_node"
        self.EVALUATE_RAG = "evaluate_rag"
        self.EVALUATE_NO_RAG = "evaluate_no_rag"
        self.EVALUATE_LLM_RAG = "evaluate_llm_rag"
        self.EVALUATE_LLM_NO_RAG = "evaluate_llm_no_rag"
        self.ENHANCED_EVALUATION = "enhance_evaluation"
        self.COMPARE = "compare"

        self.RAG_RESPONSE = "RAG"
        self.NO_RAG_RESPONSE = "NO RAG"

        # File timestamp format
        self.FILE_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

        # Test prompts (examples based on the selected code)
        self.RAG_PROMPT_TEMPLATE = """
            As an expert in marketing attribution, evaluate the semantic similarity between these two responses:

            QUESTION: {{prompt}}
        
            EXPECTED RESPONSE: 
            {{expected_result}}

            ACTUAL RESPONSE (WITH RAG): 
            {{actual_result}}

            Evaluate based on marketing attribution expertise. Focus on whether the core marketing concepts, 
            attribution models, and recommendations align - not on exact wording or structure.

            Return a JSON object with these fields:
            1. semantic_score (0-10): How semantically similar the content is
            2. marketing_accuracy (0-10): How accurate the marketing attribution concepts are
            3. key_differences: List the key conceptual differences (if any)
            4. would_mislead_marketer (Yes/No): Would the actual response mislead a marketer
            5. overall_assessment: Brief analysis of the response quality
            6. pass_fail: "PASS" or "FAIL" based on threshold of {{threshold}}
            """

        self.NON_RAG_PROMPT_TEMPLATE = """
            As an expert in marketing attribution, evaluate the semantic similarity between these two responses:
    
            QUESTION: {{prompt}}
    
            EXPECTED RESPONSE: 
            {{expected_result}}
    
            ACTUAL RESPONSE (WITHOUT RAG): 
            {{actual_result}}

            Evaluate based on marketing attribution expertise. Focus on whether the core marketing concepts, 
            attribution models, and recommendations align - not on exact wording or structure.

            Return a JSON object with these fields:
            1. semantic_score (0-10): How semantically similar the content is
            2. marketing_accuracy (0-10): How accurate the marketing attribution concepts are
            3. key_differences: List the key conceptual differences (if any)
            4. would_mislead_marketer (Yes/No): Would the actual response mislead a marketer
            5. overall_assessment: Brief analysis of the response quality
            6. pass_fail: "PASS" or "FAIL" based on threshold of {{threshold}}
            """

        self.COMPARISON_PROMPT_TEMPLATE = """
            As an expert in marketing attribution, compare these two responses to the same question.
            One response was generated with RAG (retrieval augmented generation) and one without.
            
            QUESTION: {{prompt}}
            
            EXPECTED RESPONSE:
            {{expected_result}}
        
            RESPONSE WITH RAG:
            {{rag_response}}
        
            RESPONSE WITHOUT RAG:
            {{no_rag_response}}
        
            Compare the two responses in terms of:
            1. Accuracy of marketing attribution concepts
            2. Completeness of information
            3. Specific knowledge and examples provided
            4. Practical usefulness to a marketer
        
            Return a JSON object with these fields:
            1. rag_strengths: List of ways the RAG response is better
            2. rag_weaknesses: List of ways the RAG response is worse or missing elements
            3. no_rag_strengths: List of ways the non-RAG response is better
            4. no_rag_weaknesses: List of ways the non-RAG response is worse or missing elements
            5. value_rating: Rate the value added by RAG as "None", "Low", "Medium", or "High"
            6. overall_assessment: Overall assessment of the value added by RAG for this query
            """


    def update_setting(self, setting_name, value) -> bool:
        """Update a setting value if it exists
        @param setting_name:
        @param value:
        @return:
        """
        if hasattr(self, setting_name):
            setattr(self, setting_name, value)
            return True

        return False

    def get_all_settings(self):
        """Get all settings as a dictionary (excluding private attributes)"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def validate_settings(self):
        """Validate that essential settings are present"""

        missing_settings = []

        return missing_settings