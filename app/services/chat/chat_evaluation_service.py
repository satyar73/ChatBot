import os
import pandas as pd
from typing import Dict, Tuple, Literal
from datetime import datetime
from uuid import uuid4
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END

from app.models.chat_test_models import (
    ChatTestRequest,
    ChatTestResponse,
    ChatBatchTestResponse,
    ChatLLMTestOutput,
    DualState, ComparisonOutput
)
from app.config.chat_test_config import ChatTestConfig
from app.utils.chatbot_client import ChatBotClient
from app.utils.similarity_engines import SimilarityEngines
from app.utils.llm_client import LLMClientManager  # Import the LLMClientManager
from app.utils.logging_utils import get_logger

class ChatEvaluationService:
    def __init__(self, chatbot_api_url: str = "http://localhost:8005"):
        self.config = ChatTestConfig()
        self.chatbot_client = ChatBotClient(chatbot_api_url)
        self.logger = get_logger(f"{__name__}.ChatTestService", "DEBUG")
        self.logger.info("ChatTestService initialized")

        # Define weights for enhanced evaluation
        self.evaluation_weights = {
            "concept_coverage": 0.35,  # Increased from default - key concepts are critical
            "semantic_similarity": 0.30,  # Semantic understanding
            "factual_accuracy": 0.20,  # Correct information
            "specificity": 0.15,  # Specific details, numbers, examples
        }

        if not hasattr(self.config, 'ENHANCE_EVALUATION'):
            self.config.ENHANCE_EVALUATION = "enhance_evaluation"

    def get_llm(self):
        """Create LLM instance using LLMClientManager"""
        return LLMClientManager.get_chat_llm(
            model=self.config.LLM_MODEL,
            temperature=self.config.LLM_TEMPERATURE
        )

    # Helper method get the chat responses

    async def _get_chat_response_rag_no_rag(self,
                                    state: DualState):
        """Node that calls Chatbot API to get both RAG and non-RAG responses"""
        prompt = state["prompt"]

        try:
            # Get both RAG and non-RAG responses in a single API call
            self.logger.debug("Calling Chatbot API for responses")
            rag_response, no_rag_response = await self.chatbot_client.get_response(prompt)
            
            # Ensure responses are not None
            rag_response = rag_response or "No RAG response received"
            no_rag_response = no_rag_response or "No non-RAG response received"
            
            self.logger.debug(f"Received responses - RAG: {len(rag_response)} chars, "
                              f"non-RAG: {len(no_rag_response)} chars")

            # Return updated state with both responses
            self.logger.debug(f"start_node completed successfully, "
                              f"next step: {self.config.EVALUATE_RAG}")
            return {
                **state,
                "rag_response": rag_response,
                "no_rag_response": no_rag_response,
                self.config.NEXT: self.config.EVALUATE_RAG
            }
        except Exception as e:
            self.logger.error(f"Error getting ChatBot responses: {str(e)}")
            # Return error state
            return {
                **state,
                "rag_response": f"Error getting response: {str(e)}",
                "no_rag_response": f"Error getting response: {str(e)}",
                "reasoning": f"Error communicating with ChatBot API: {str(e)}",
                self.config.NEXT: self.config.END
            }

    #helper methods for internal tests
    def _compare_response_using_internal_tests(self,
                                    state: DualState,
                                    actual: str,
                                    expected: str,
                                    module: str,
                                    next_step_pass: str,
                                    next_step_fail: str,
                                    threshold: float):
        try:
            self.logger.debug("Running similarity tests")
            results = SimilarityEngines.quick_test(actual, expected)
            self.logger.debug(f"Similarity test results: "
                              f"weighted_similarity={results['weighted_similarity']:.4f}")

            # Determine if passed based on weighted similarity
            passed = results["weighted_similarity"] >= threshold

            # Generate basic reasoning
            if passed:
                self.logger.debug(f"{module} response PASSED with score: "
                                  f"{results['weighted_similarity']:.4f}")
                reasoning = (f"{module} response: Quick test PASSED with similarity score "
                             f"of {results['weighted_similarity']:.4f}")
                next_step = next_step_pass
            else:
                self.logger.debug(f"{module} response FAILED with score: "
                                  f"{results['weighted_similarity']:.4f}")
                reasoning = (f"{module} response: Quick test FAILED with similarity "
                             f"score of {results['weighted_similarity']:.4f}")
                if results.get("concepts_missing"):
                    concepts_str = ", ".join(results["concepts_missing"][:5])
                    self.logger.debug(f"Missing key concepts: {concepts_str}")
                    reasoning += f"\nMissing key concepts: {concepts_str}"
                next_step = next_step_fail

            self.logger.debug(f"Evaluation {module} completed, next step: {next_step}")
            # Return updated state with results
            return {
                **state,
                f"{module}_test_results": results,
                f"{module}_passed": passed,
                "reasoning": state.get("reasoning", "") + reasoning
                                if state.get("reasoning")
                                else reasoning,
                self.config.NEXT: next_step_fail
            }
        except Exception as e:
            error_msg = f"Error in {module} evaluation: {str(e)}"
            self.logger.error(error_msg)
            return {
                **state,
                f"{module}_test_results": {"error": str(e)},
                f"{module}_passed": False,
                "reasoning": (state.get("reasoning", "") + "\n" + error_msg)
                                        if state.get("reasoning")
                                        else error_msg,
                self.config.NEXT: next_step_fail
            }

    def _evaluate_response_using_llm(self,
                                     state: DualState,
                                     module: str,
                                     prompt: str,
                                     expected: str,
                                     response: str,
                                     similarity_score: float,
                                     next_step_pass: str,
                                     next_step_fail: str) -> DualState:
        """Use LLM to evaluate RAG response that failed the quick check"""
        self.logger.debug("Starting llm_evaluate_rag_node")
        # Use LLMClientManager instead of direct LLM creation
        llm = LLMClientManager.with_structured_output(ChatLLMTestOutput)
        self.logger.debug("Created LLM client with structured output")

        test_prompt = self.config.RAG_PROMPT_TEMPLATE

        try:
            # Replace placeholders
            filled_prompt = test_prompt
            filled_prompt = filled_prompt.replace("{{prompt}}", prompt)
            filled_prompt = filled_prompt.replace("{{expected_result}}", expected)
            filled_prompt = filled_prompt.replace("{{actual_result}}", response)
            filled_prompt = filled_prompt.replace("{{threshold}}",
                                            str(similarity_score * 10))  # Convert 0.7 to 7
            self.logger.debug(f"Prepared LLM prompt for {module} evaluation")

            messages = [
                SystemMessage(content=filled_prompt)
            ]

            # Get LLM evaluation
            self.logger.debug(f"Invoking LLM for {module} evaluation")
            results = llm.invoke(messages)
            self.logger.debug(
                f"LLM evaluation result: pass_fail={results['pass_fail']}, "
                f"semantic_score={results['semantic_score']}/10")

            # Determine if passed based on LLM judgment
            passed = results["pass_fail"] == "PASS"
            normalized_score = (results["semantic_score"] +
                                results["marketing_accuracy"]) / 20  # Convert to 0-1 scale
            self.logger.debug(f"Normalized score: {normalized_score:.4f}")

            # Generate reasoning
            if passed:
                self.logger.debug("LLM evaluation PASSED")
                reasoning = (f"\n\n{module} response - LLM test: PASSED "
                             f"with semantic score {results['semantic_score']}/10 "
                             f"and marketing accuracy {results['marketing_accuracy']}/10")
                final_passed = True
            else:
                self.logger.debug("LLM evaluation FAILED")
                reasoning = (f"\n\n{module} response - LLM test: FAILED "
                             f"with semantic score {results['semantic_score']}/10 "
                             f"and marketing accuracy {results['marketing_accuracy']}/10")
                if results["key_differences"]:
                    differences_str = "\n- " + "\n- ".join(results["key_differences"][:3])
                    self.logger.debug(f"Key differences identified: {len(results['key_differences'])}")
                    reasoning += f"\nKey differences: {differences_str}"
                final_passed = state.get("rag_passed", False)
                self.logger.debug(f"Final {module} evaluation: {final_passed}")

            self.logger.debug(f"Evaluation completed, next step: {next_step_pass}")
            # Store everything in state
            return {
                **state,
                f"{module}_llm_results": {**results, "normalized_score": normalized_score},
                f"{module}_passed": final_passed,
                "reasoning": state.get("reasoning", "") + reasoning
                                if state.get("reasoning")
                                else reasoning,
                self.config.NEXT: next_step_pass
            }
        except Exception as e:
            error_msg = f"\n\nError in RAG LLM evaluation: {str(e)}"
            self.logger.error(error_msg)
            # Return state with error information, keeping previous test results
            return {
                **state,
                f"{module}_llm_results": {"error": str(e)},
                "reasoning": (state.get("reasoning", "") + error_msg)
                            if state.get("reasoning")
                    else error_msg,
                self.config.NEXT: next_step_fail
            }

    def _evaluate_response_quality(self,
                                   expected: str,
                                   response: str,
                                   similarity_score: float,
                                   concept_coverage: float,
                                   test_details: Dict) -> Tuple[float, Dict]:
        """
            Enhanced evaluation function that goes beyond simple
            similarity to assess response quality.
        """
        # 1. Concept coverage - are all key concepts present?
        # 2. Semantic similarity - already calculated
        evaluation = {"concept_coverage": concept_coverage,
                      "semantic_similarity": similarity_score}

        # 3. Factual accuracy - check if missing numerical values
        numbers_missing = test_details.get("numbers_missing", [])
        factual_accuracy = 1.0 - (len(numbers_missing) * 0.1)
        factual_accuracy = max(0.0, min(1.0, factual_accuracy))
        evaluation["factual_accuracy"] = factual_accuracy

        # 4. Specificity - check for specific metrics, examples, or technical terms
        # This is a heuristic - check for numbers, percentages, specific terms
        has_numbers = any(c.isdigit() for c in response)
        has_percentages = "%" in response
        words_in_response = len(response.split())
        words_in_expected = len(expected.split())

        # Higher specificity if response has similar or more content than expected
        length_ratio = min(1.0, words_in_response / max(1, words_in_expected))

        specificity = 0.5  # Base value
        if has_numbers:
            specificity += 0.2
        if has_percentages:
            specificity += 0.1

        # Adjust based on length ratio - longer answers often contain more specifics
        specificity = specificity * (0.5 + 0.5 * length_ratio)
        specificity = min(1.0, specificity)
        evaluation["specificity"] = specificity

        # Calculate weighted score
        refined_score = sum(
            self.evaluation_weights[metric] * score
            for metric, score in evaluation.items()
        )

        return refined_score, evaluation

    @staticmethod
    def _evaluate_rag_value(rag_score: float,
                            no_rag_score: float,
                            rag_eval: Dict,
                            no_rag_eval: Dict) -> Tuple[str, str, float]:
        """
        Enhanced function to evaluate the value added by RAG compared to non-RAG.
        """
        # Calculate score difference
        score_diff = rag_score - no_rag_score

        # Determine value rating
        if score_diff > 0.15:
            value_rating = "High"
        elif score_diff > 0.05:
            value_rating = "Medium"
        elif score_diff > -0.05:
            value_rating = "Low"
        elif score_diff > -0.15:
            value_rating = "None"
        else:
            value_rating = "Negative"

        # Check specific dimensions for a more nuanced assessment
        specificity_diff = (rag_eval.get("specificity", 0)
                            - no_rag_eval.get("specificity", 0))
        factual_diff = (rag_eval.get("factual_accuracy", 0)
                        - no_rag_eval.get("factual_accuracy", 0))
        concept_diff = (rag_eval.get("concept_coverage", 0)
                        - no_rag_eval.get("concept_coverage", 0))

        # Generate value assessment
        assessment_parts = []

        if abs(score_diff) < 0.05:
            assessment_parts.append("RAG and non-RAG perform similarly "
                                    "(score difference: {:.4f})".format(score_diff))
        elif score_diff > 0:
            assessment_parts.append("RAG " +
                                    ("significantly "
                                        if score_diff > 0.15
                                        else "moderately ") +
                                    "outperforms non-RAG "
                                    "(score difference: +{:.4f})".format(score_diff))
        else:
            assessment_parts.append("Non-RAG outperforms RAG "
                                    "(score difference: {:.4f})".format(score_diff))

        # Add specific strengths/weaknesses
        if concept_diff > 0.1:
            assessment_parts.append("RAG provides better concept coverage "
                                    "(+{:.2f})".format(concept_diff))
        elif concept_diff < -0.1:
            assessment_parts.append("RAG misses key concepts "
                                    "({:.2f})".format(concept_diff))

        if specificity_diff > 0.1:
            assessment_parts.append("RAG provides more specific information "
                                    "(+{:.2f})".format(specificity_diff))
        elif specificity_diff < -0.1:
            assessment_parts.append("RAG lacks specificity compared to non-RAG "
                                    "({:.2f})".format(specificity_diff))

        if factual_diff > 0.1:
            assessment_parts.append("RAG is more factually accurate "
                                    "(+{:.2f})".format(factual_diff))
        elif factual_diff < -0.1:
            assessment_parts.append("RAG has factual inaccuracies "
                                    "({:.2f})".format(factual_diff))

        value_assessment = "\n".join(assessment_parts)

        return value_rating, value_assessment, score_diff

    def _generate_comparison(self,
                             state: DualState) :
        """Compare RAG and non-RAG responses to determine RAG value"""
        self.logger.debug("Starting comparison of rag and non-rag responses")

        # Check if we have enhanced evaluation results
        if state.get("rag_enhanced_results") and state.get("no_rag_enhanced_results"):
            # Use enhanced scores
            rag_score = state["rag_enhanced_results"]["refined_score"]
            no_rag_score = state["no_rag_enhanced_results"]["refined_score"]
            rag_evaluation = state["rag_enhanced_results"]["evaluation"]
            no_rag_evaluation = state["no_rag_enhanced_results"]["evaluation"]

            # Use enhanced evaluation to determine value rating
            value_rating, value_assessment, score_diff = self._evaluate_rag_value(
                rag_score, no_rag_score, rag_evaluation, no_rag_evaluation
            )
            self.logger.debug(f"Enhanced evaluation - RAG value rating: {value_rating}")
        else:
            # Fall back to original approach
            self.logger.debug("No enhanced evaluation results found, "
                              "using original comparison method")

        llm = LLMClientManager.with_structured_output(ComparisonOutput)
        self.logger.debug("Created LLM client with structured output for comparison")

        # Get the best scores for each approach
        # Safely calculate rag_score
        rag_score = (state["rag_llm_results"].get("normalized_score", 0)
                                   if state.get("rag_llm_results")
                                   else state.get("rag_test_results", {}).get("weighted_similarity", 0))

        # Safely calculate no_rag_score
        no_rag_score = (state["no_rag_llm_results"].get("normalized_score", 0)
                                if state.get("no_rag_llm_results")
                                else state.get("no_rag_test_results", {}).get("weighted_similarity", 0))

        self.logger.debug(f"Best scores - RAG: {rag_score:.4f}, non-RAG: {no_rag_score:.4f}")

        # Simple score-based comparison
        score_diff = rag_score - no_rag_score
        self.logger.debug(f"Score difference (RAG - non-RAG): {score_diff:.4f}")

        # Generate comparison text
        if score_diff > 0.2:
            self.logger.debug("RAG significantly outperforms non-RAG")
            comparison_text = f"RAG significantly outperforms non-RAG (score difference: +{score_diff:.4f})"
            value_rating = "High"
        elif score_diff > 0.05:
            self.logger.debug("RAG moderately outperforms non-RAG")
            comparison_text = f"RAG moderately outperforms non-RAG (score difference: +{score_diff:.4f})"
            value_rating = "Medium"
        elif score_diff > 0.001:  # Use a small threshold to avoid rounding errors
            self.logger.debug("RAG slightly outperforms non-RAG")
            comparison_text = f"RAG slightly outperforms non-RAG (score difference: +{score_diff:.4f})"
            value_rating = "Low"
        elif abs(score_diff) <= 0.001:  # If difference is essentially zero
            self.logger.debug("RAG and non-RAG perform similarly")
            comparison_text = f"RAG and non-RAG perform similarly (score difference: {score_diff:.4f})"
            value_rating = "None"
        else:
            self.logger.debug("Non-RAG outperforms RAG")
            comparison_text = f"Non-RAG outperforms RAG (score difference: {score_diff:.4f})"
            value_rating = "Negative"

        self.logger.debug(f"RAG value rating: {value_rating}")

        # Create detailed comparison
        comparison = {
                "rag_score": rag_score,
                "no_rag_score": no_rag_score,
                "score_difference": score_diff,
                "rag_passed": state.get("rag_passed", False),
                "no_rag_passed": state.get("no_rag_passed", False),
                "value_assessment": comparison_text,
                "rag_value_rating": value_rating
        }

        # Add enhanced evaluation metrics if available
        if state.get("rag_enhanced_results") and state.get("no_rag_enhanced_results"):
            comparison.update({
                "rag_specificity": state["rag_enhanced_results"]["evaluation"].get("specificity", 0),
                "no_rag_specificity": state["no_rag_enhanced_results"]["evaluation"].get("specificity", 0),
                "rag_factual_accuracy": state["rag_enhanced_results"]["evaluation"].get("factual_accuracy", 0),
                "no_rag_factual_accuracy": state["no_rag_enhanced_results"]["evaluation"].get("factual_accuracy", 0),
                "enhanced_evaluation": True
            })

        # Also run an LLM comparison for more detailed analysis
        try:
            compare_prompt = self.config.COMPARISON_PROMPT_TEMPLATE
            self.logger.debug("Preparing LLM comparison prompt")

            # Fill in the values
            compare_prompt = compare_prompt.replace("{{prompt}}", state["prompt"])
            compare_prompt = compare_prompt.replace("{{expected_result}}", state["expected_result"])
            compare_prompt = compare_prompt.replace("{{rag_response}}", state["rag_response"])
            compare_prompt = compare_prompt.replace("{{no_rag_response}}", state["no_rag_response"])

            messages = [
                SystemMessage(content=compare_prompt)
            ]

            # Get LLM evaluation
            self.logger.debug("Invoking LLM for detailed comparison")
            llm_comparison = llm.invoke(messages)
            self.logger.debug(f"LLM comparison value rating: {llm_comparison['value_rating']}")

            # Add LLM comparison to our comparison object
            comparison.update({
                "llm_comparison": llm_comparison
            })

            # Add the LLM's detailed assessment to our reasoning
            detailed_comparison = f"\n\n## RAG vs Non-RAG Comparison:\n\n"
            detailed_comparison += f"Value added by RAG: {llm_comparison['value_rating']}\n\n"
            detailed_comparison += f"Overall assessment: {llm_comparison['overall_assessment']}\n\n"

            detailed_comparison += "### RAG Strengths:\n"
            for strength in llm_comparison["rag_strengths"]:
                detailed_comparison += f"- {strength}\n"

            detailed_comparison += "\n### RAG Weaknesses:\n"
            for weakness in llm_comparison["rag_weaknesses"]:
                detailed_comparison += f"- {weakness}\n"

            detailed_comparison += "\n### Non-RAG Strengths:\n"
            for strength in llm_comparison["no_rag_strengths"]:
                detailed_comparison += f"- {strength}\n"

            detailed_comparison += "\n### Non-RAG Weaknesses:\n"
            for weakness in llm_comparison["no_rag_weaknesses"]:
                detailed_comparison += f"- {weakness}\n"

            self.logger.debug("Generated detailed comparison from LLM results")

        except Exception as e:
            self.logger.error(f"Error in LLM comparison: {str(e)}")
            detailed_comparison = f"\n\nError in detailed LLM comparison: {str(e)}"
            # Keep the basic comparison without LLM input

        # Update state with comparison and detailed reasoning
        return {
            **state,
            "comparison": comparison,
            "reasoning": state.get("reasoning",
                                   "") + f"\n\n## RAG vs Non-RAG Comparison:\n{comparison_text}" + detailed_comparison,
            self.config.NEXT: self.config.END
        }

    async def start_get_chat_response_node(self,
                                           state: DualState):
        self.logger.debug(f"Getting rag and no responses prompt: {state['prompt'][:50]}...")

        return await self._get_chat_response_rag_no_rag(state)

    def evaluate_rag_node(self, state: DualState) -> DualState:
        """Evaluate RAG response with quick similarity test"""
        self.logger.debug("Starting evaluate_rag_node")
        actual = state["rag_response"]
        expected = state["expected_result"]
        threshold = state["similarity_threshold"]
        self.logger.debug(f"Evaluating RAG response against threshold: {threshold}")

        return self._compare_response_using_internal_tests(state, actual, expected,
                                                           "rag",
                                                           self.config.EVALUATE_NO_RAG,
                                                           self.config.EVALUATE_LLM_RAG,
                                                           threshold)

    def evaluate_no_rag_node(self, state: DualState) -> DualState:
        """Evaluate non-RAG response with quick similarity test"""
        self.logger.debug("Starting evaluate_no_rag_node")
        actual = state["no_rag_response"]
        expected = state["expected_result"]
        threshold = state["similarity_threshold"]
        self.logger.debug(f"Evaluating non-RAG response against threshold: {threshold}")

        return self._compare_response_using_internal_tests(state, actual, expected,
                                                           "no_rag",
                                                           self.config.ENHANCED_EVALUATION,
                                                           self.config.EVALUATE_LLM_NO_RAG,
                                                           threshold)

    def llm_evaluate_rag_node(self, state: DualState) -> DualState:
        return self._evaluate_response_using_llm(state, "rag",
                                                 state["prompt"],
                                                 state["expected_result"],
                                                 state["rag_response"],
                                                 state["similarity_threshold"],
                                                 self.config.EVALUATE_NO_RAG,
                                                 self.config.EVALUATE_NO_RAG)

    def llm_evaluate_no_rag_node(self, state: DualState) -> DualState:
        return self._evaluate_response_using_llm(state, "no_rag",
                                                 state["prompt"],
                                                 state["expected_result"],
                                                 state["no_rag_response"],
                                                 state["similarity_threshold"],
                                                 self.config.ENHANCED_EVALUATION,
                                                 self.config.COMPARE)

    def enhance_evaluation_node(self, state: DualState) -> DualState:
        """Apply enhanced evaluation metrics to both RAG and non-RAG responses"""
        self.logger.debug("Starting enhance_evaluation_node")

        expected = state["expected_result"]

        # Process RAG response
        rag_test_details = state.get("rag_test_results", {})
        rag_llm_results = state.get("rag_llm_results", {})
        # Get the base scores
        rag_base_score = max(
            rag_test_details.get("weighted_similarity", 0),
            rag_llm_results.get("normalized_score", 0) if rag_llm_results else 0
        )

        # Apply enhanced evaluation
        rag_refined_score, rag_evaluation = self._evaluate_response_quality(
                expected, state["rag_response"],
                rag_base_score,
                rag_test_details.get("concept_coverage", 0),
                rag_test_details
            )
        self.logger.debug(f"RAG refined score: {rag_refined_score:.4f}")

        # Process non-RAG response
        no_rag_test_details = state.get("no_rag_test_results", {})
        no_rag_llm_results = state.get("no_rag_llm_results", {})
        # Get the base scores
        no_rag_base_score = max(
            no_rag_test_details.get("weighted_similarity", 0),
            no_rag_llm_results.get("normalized_score", 0) if no_rag_llm_results else 0
        )
        no_rag_concept_coverage = no_rag_test_details.get("concept_coverage", 0)

        # Apply enhanced evaluation
        no_rag_refined_score, no_rag_evaluation = (
                        self._evaluate_response_quality(
                                expected,
                                state["no_rag_response"],
                                no_rag_base_score,
                                no_rag_concept_coverage,
                                no_rag_test_details
        ))
        self.logger.debug(f"Non-RAG refined score: {no_rag_refined_score:.4f}")

        # Determine pass/fail based on refined scores
        rag_passed = (rag_refined_score >= state["similarity_threshold"]
                        or rag_refined_score > no_rag_refined_score)
        no_rag_passed = no_rag_refined_score >= state["similarity_threshold"]

        # Add enhanced evaluations to state
        enhanced_state = {
            **state,
            "rag_enhanced_results": {
                "refined_score": rag_refined_score,
                "evaluation": rag_evaluation,
                "original_score": rag_base_score
            },
            "no_rag_enhanced_results": {
                "refined_score": no_rag_refined_score,
                "evaluation": no_rag_evaluation,
                "original_score": no_rag_base_score
            },
            "rag_passed": rag_passed,
            "no_rag_passed": no_rag_passed,
            self.config.NEXT: self.config.COMPARE
        }

        # Add reasoning about enhanced evaluation
        reasoning = "\n\n## Enhanced Evaluation Metrics:\n"
        reasoning += (f"RAG refined score: {rag_refined_score:.4f} "
                      f"(original: {rag_base_score:.4f})\n")
        reasoning += (f"Non-RAG refined score: {no_rag_refined_score:.4f} "
                      f"(original: {no_rag_base_score:.4f})\n")
        reasoning += "\nMetric breakdown:\n"

        # Add individual metrics
        for metric, weight in self.evaluation_weights.items():
            rag_value = rag_evaluation.get(metric, 0)
            no_rag_value = no_rag_evaluation.get(metric, 0)
            reasoning += (f"- {metric} (weight: {weight:.2f}): "
                          f"RAG={rag_value:.4f}, Non-RAG={no_rag_value:.4f}\n")

        enhanced_state["reasoning"] = (state.get("reasoning", "") + reasoning) \
                                            if state.get("reasoning") \
                                            else reasoning

        self.logger.debug(f"enhance_evaluation_node completed, "
                          f"next step: {self.config.COMPARE}")
        return enhanced_state

    def compare_node(self, state: DualState) -> DualState:
        return self._generate_comparison(state)

    def router(self, state: DualState) -> Literal[
        "evaluate_rag", "evaluate_no_rag", "evaluate_llm_rag",
        "evaluate_llm_no_rag", "enhance_evaluation", "compare", "END"]:
        """Router function to determine next step based on state"""

        # noinspection PyTypedDict
        next_step = state.get(self.config.NEXT, self.config.END)
        self.logger.debug(f"Router determining next step: {next_step}")
        return next_step

    def build_simple_test_graph(self):
        try:
            builder = StateGraph(DualState)

            builder.add_node(self.config.START_GET_CHAT_RESPONSE, self.start_get_chat_response_node)
            builder.add_node(self.config.EVALUATE_LLM_RAG, self.llm_evaluate_rag_node)
            builder.add_node(self.config.EVALUATE_LLM_NO_RAG, self.llm_evaluate_no_rag_node)
            builder.add_node(self.config.COMPARE, self.compare_node)

            builder.add_edge(START, self.config.START_GET_CHAT_RESPONSE)
            builder.add_edge(self.config.START_GET_CHAT_RESPONSE, self.config.EVALUATE_LLM_RAG)
            builder.add_edge(self.config.EVALUATE_LLM_RAG, self.config.EVALUATE_LLM_NO_RAG)
            builder.add_edge(self.config.EVALUATE_LLM_NO_RAG, self.config.COMPARE)

            builder.add_conditional_edges(
                self.config.COMPARE,
                self.router,
                {
                    self.config.END: END,
                }
            )

            # Compile and return
            state_graph = builder.compile()
        except Exception as e:
           state_graph = None

        return state_graph

    def build_test_graph(self):
        """Build a testing workflow graph with dual evaluation"""
        # Create the graph
        builder = StateGraph(DualState)

        if not hasattr(self.config, 'ENHANCE_EVALUATION'):
            self.config.ENHANCE_EVALUATION = "enhance_evaluation"

        # Add nodes
        builder.add_node(self.config.START_GET_CHAT_RESPONSE, self.start_get_chat_response_node)
        builder.add_node(self.config.EVALUATE_RAG, self.evaluate_rag_node)
        builder.add_node(self.config.EVALUATE_NO_RAG, self.evaluate_no_rag_node)
        builder.add_node(self.config.EVALUATE_LLM_RAG, self.llm_evaluate_rag_node)
        builder.add_node(self.config.EVALUATE_LLM_NO_RAG, self.llm_evaluate_no_rag_node)
        builder.add_node(self.config.ENHANCE_EVALUATION, self.enhance_evaluation_node)
        builder.add_node(self.config.COMPARE, self.compare_node)

        # Set up the flow
        builder.add_edge(START, self.config.START_GET_CHAT_RESPONSE)
        builder.add_edge(self.config.START_GET_CHAT_RESPONSE, self.config.EVALUATE_RAG)

        # Add conditional edges
        builder.add_conditional_edges(
            self.config.EVALUATE_RAG,
            self.router,
            {
                self.config.EVALUATE_LLM_RAG : self.config.EVALUATE_LLM_RAG,
                self.config.EVALUATE_NO_RAG : self.config.EVALUATE_NO_RAG,
            }
        )

        builder.add_conditional_edges(
            self.config.EVALUATE_LLM_RAG,
            self.router,
            {
                self.config.EVALUATE_NO_RAG : self.config.EVALUATE_NO_RAG,
            }
        )

        builder.add_conditional_edges(
            self.config.EVALUATE_NO_RAG,
            self.router,
            {
                self.config.EVALUATE_LLM_NO_RAG : self.config.EVALUATE_LLM_NO_RAG,
                self.config.ENHANCE_EVALUATION : self.config.ENHANCE_EVALUATION,
            }
        )

        builder.add_conditional_edges(
            self.config.EVALUATE_LLM_NO_RAG,
            self.router,
            {
                self.config.ENHANCE_EVALUATION : self.config.ENHANCE_EVALUATION,
            }
        )

        builder.add_conditional_edges(
            self.config.ENHANCE_EVALUATION,
            self.router,
            {
                self.config.COMPARE: self.config.COMPARE,
            }
        )

        builder.add_conditional_edges(
           self.config.COMPARE,
            self.router,
            {
                self.config.END: END,
            }
        )

        # Compile and return
        return builder.compile()

    async def run_test(self, request: ChatTestRequest) -> ChatTestResponse | None:
        """Run a test on a prompt/expected result pair with RAG comparison"""
        test_id = request.test_id or str(uuid4())

        # Initialize the graph
        graph = self.build_simple_test_graph()
        if not graph:
            return ChatTestResponse(
                    test_id = test_id,
                    prompt = request.prompt,
                    expected_result = request.expected_result,
                    actual_result = "Unable to build the graph",
                    passed = False,
                    reasoning = "Unable to build the graph",
                    similarity_score = 0,
                    detailed_analysis = {"results": "Test results"}
                )

        # Initialize the state
        initial_state = {
            "prompt": request.prompt,
            "expected_result": request.expected_result,
            "similarity_threshold": request.similarity_threshold,
            "rag_response": None,
            "rag_test_results": None,
            "rag_llm_results": None,
            "rag_passed": None,
            "rag_enhanced_results": None,
            "no_rag_response": None,
            "no_rag_test_results": None,
            "no_rag_llm_results": None,
            "no_rag_enhanced_results": None,
            "no_rag_passed": None,
            "reasoning": None,
            "comparison": None,
            "next": None
        }

        try:
            # Run the graph
            final_state = await graph.ainvoke(initial_state)

            # Determine overall passed status (if either passed)
            overall_passed = (final_state.get("rag_passed", False)
                              or final_state.get("no_rag_passed", False) or False)

            weighted_similarity = (final_state.get("rag_test_results") or {}).get("weighted_similarity", 0)

            if final_state.get("rag_llm_results"):  # Check if rag_llm_results exists and is not None
                normalized_score = final_state["rag_llm_results"].get("normalized_score", 0)
                rag_score = max(weighted_similarity,normalized_score )
            else:
                # If rag_llm_results does not exist, fall back to rag_test_results only
                rag_score = weighted_similarity

            # Safely calculate no_rag_score
            weighted_similarity = (final_state.get("no_rag_test_results") or {}).get("weighted_similarity", 0)
            if final_state.get("no_rag_llm_results"):
                normalized_score = final_state["no_rag_llm_results"].get("normalized_score", 0)
                no_rag_score = max(weighted_similarity, normalized_score)
            else:
                # If no_rag_llm_results does not exist, fall back to no_rag_test_results only
                no_rag_score = weighted_similarity

            # Get the best similarity score from all tests
            similarity_score = max(rag_score, no_rag_score)

            overall_passed = overall_passed or similarity_score >= request.similarity_threshold

            # Include enhanced evaluation results if available
            detailed_analysis = {
                "rag_response": final_state.get("rag_response", ""),
                "no_rag_response": final_state.get("no_rag_response", ""),
                "rag_test": final_state.get("rag_test_results", {}),
                "no_rag_test": final_state.get("no_rag_test_results", {}),
                "rag_llm_test": final_state.get("rag_llm_results", {}),
                "no_rag_llm_test": final_state.get("no_rag_llm_results", {}),
                "comparison": final_state.get("comparison", {})
            }

            # Add enhanced evaluation if available
            if final_state.get("rag_enhanced_results"):
                detailed_analysis["rag_enhanced"] = final_state.get("rag_enhanced_results", {})

            if final_state.get("no_rag_enhanced_results"):
                detailed_analysis["no_rag_enhanced"] = final_state.get("no_rag_enhanced_results", {})

            # Build response
            response = ChatTestResponse(
                test_id=test_id,
                prompt=request.prompt,
                expected_result=request.expected_result,
                actual_result=final_state.get("rag_response", ""),  # Use RAG response as the primary
                passed=overall_passed,
                reasoning=final_state.get("reasoning", "No reasoning provided"),
                similarity_score=similarity_score,
                detailed_analysis= detailed_analysis
            )

            return response
        except Exception as e:
            # Return error response if something goes wrong
            return ChatTestResponse(
                test_id=test_id,
                prompt=request.prompt,
                expected_result=request.expected_result,
                actual_result="Error occurred during test execution",
                passed=False,
                reasoning=f"Error: {str(e)}",
                similarity_score=0,
                detailed_analysis={"error": str(e)}
            )
        finally:
            # Always ensure we clean up resources
            try:
                await self.chatbot_client.cleanup()
            except Exception as e:
                self.logger.error(f"Error cleaning up chatbot client: {str(e)}")

    # noinspection PyTypeChecker
    async def run_batch_test(self, csv_file: str, similarity_threshold: float = 0.7):
        """Run tests from a CSV file"""
        # Load CSV
        df = pd.read_csv(csv_file)

        # Initialize results
        results = []
        total = len(df)
        passed = 0
        failed = 0

        # Process each row
        for i, (_, row) in enumerate(df.iterrows()):
            print(f"Running test {i + 1}/{total}: {row['Prompt'][:50]}...")

            # Create test request
            test_request = ChatTestRequest(
                prompt=row["Prompt"],
                expected_result=row["Expected Result"],
                similarity_threshold=similarity_threshold
            )

            # Run test
            try:
                result = await self.run_test(test_request)
                results.append(result.dict())

                if result.passed:
                    passed += 1
                else:
                    failed += 1

                # Add RAG comparison metrics to results
                if "comparison" in result.detailed_analysis:
                    comp = result.detailed_analysis["comparison"]
                    print(f"  -> RAG value assessment: {comp.get('value_assessment', 'N/A')}")
                    if "llm_comparison" in comp:
                        print(f"  -> LLM value rating: {comp['llm_comparison'].get('value_rating', 'N/A')}")

            except Exception as e:
                print(f"Error on test {i + 1}: {str(e)}")
                failed += 1
                results.append({
                    "test_id": str(uuid4()),
                    "prompt": row["Prompt"],
                    "expected_result": row["Expected Result"],
                    "actual_result": "Error occurred",
                    "passed": False,
                    "reasoning": f"Error: {str(e)}",
                    "similarity_score": 0,
                    "detailed_analysis": {"error": str(e)}
                })

        # Save results to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Try multiple approaches to ensure we save results somewhere accessible

        # Approach 1: Use absolute path from project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        results_dir_1 = os.path.join(project_root, "test_results")

        # Approach 2: Use Docker volume path (if running in Docker)
        results_dir_2 = "/app/test_results"

        # Approach 3: Use current working directory
        results_dir_3 = os.path.join(os.getcwd(), "test_results")

        # Print debug info about directories
        print(f"DEBUG: Current working directory: {os.getcwd()}")
        print(f"DEBUG: Project root path: {project_root}")
        print(f"DEBUG: Results dir 1: {results_dir_1}")
        print(f"DEBUG: Results dir 2: {results_dir_2}")
        print(f"DEBUG: Results dir 3: {results_dir_3}")

        # Try to create all possible directories
        for results_dir in [results_dir_1, results_dir_2, results_dir_3]:
            try:
                os.makedirs(results_dir, exist_ok=True)
                print(f"DEBUG: Successfully created/verified directory: {results_dir}")
            except Exception as e:
                print(f"DEBUG: Error creating directory {results_dir}: {str(e)}")

        # Decide which directory to use - prioritize Docker volume if it exists
        if os.path.exists(results_dir_2) and os.access(results_dir_2, os.W_OK):
            results_dir = results_dir_2
        elif os.path.exists(results_dir_1) and os.access(results_dir_1, os.W_OK):
            results_dir = results_dir_1
        else:
            results_dir = results_dir_3

        print(f"DEBUG: Selected results directory: {results_dir}")

        # Define output file paths
        output_file = os.path.join(results_dir, f"attribution_test_results_{timestamp}.csv")
        rag_report_file = os.path.join(results_dir, f"rag_comparison_report_{timestamp}.csv")

        # Log the file paths
        print(f"Saving test results to: {output_file}")
        print(f"Saving RAG comparison report to: {rag_report_file}")

        # Also log to the Python logger
        self.logger.info(f"Saving test results to: {output_file}")
        self.logger.info(f"Saving RAG comparison report to: {rag_report_file}")

        # Convert results to DataFrame and save
        try:
            results_df = pd.DataFrame(results)
            print(f"DEBUG: Created DataFrame with {len(results_df)} rows")

            # Save main results file
            results_df.to_csv(output_file, index=False)
            print(f"DEBUG: Successfully saved results to {output_file}")

            # Verify file was created
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                print(f"DEBUG: Verified file exists at {output_file}, size: {file_size} bytes")
            else:
                print(f"DEBUG: ERROR - File was not created at {output_file}")

        except Exception as e:
            print(f"DEBUG: Error saving results file: {str(e)}")
            import traceback
            print(f"DEBUG: Traceback: {traceback.format_exc()}")

        try:
            return ChatBatchTestResponse(
                total_tests=total,
                passed=passed,
                failed=failed,
                pass_rate=(passed / total) * 100 if total > 0 else 0,
                output_file=output_file,
                rag_report_file=rag_report_file,  # Added rag_report_file attribute
                results=results
            )
        finally:
            # Ensure client is cleaned up at the end of the batch test
            try:
                await self.chatbot_client.cleanup()
                self.logger.info("Successfully cleaned up chatbot client")
            except Exception as e:
                self.logger.error(f"Error cleaning up chatbot client: {str(e)}")

    async def cleanup(self):
        """Cleanup resources"""
        await self.chatbot_client.cleanup()