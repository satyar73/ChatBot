import asyncio
import sys
import os
import pandas as pd
from datetime import datetime
from typing import Dict
from uuid import uuid4

# Import the TestService and models
from app.services.chat_test_service import ChatTestService
from app.models.chat_test_models import ChatTestRequest, ChatBatchTestRequest


class ChatTesterCLI:
    """
    Command-line interface for running attribution tests using the TestService
    """

    def __init__(self, api_url: str, csv_path: str, similarity_threshold: float = 0.7):
        """
        Initialize the chat tester CLI.

        Args:
            api_url: Base URL for the API
            csv_path: Path to CSV file containing test cases
            similarity_threshold: Threshold for similarity score to consider a test passed
        """
        self.api_url = api_url
        self.csv_path = csv_path
        self.similarity_threshold = similarity_threshold
        self.session_id = f"test_session_{int(datetime.now().timestamp())}"

        # Initialize test service with the API URL
        self.test_service = ChatTestService(chatbot_api_url=api_url)

    async def load_test_cases(self) -> pd.DataFrame:
        """Load test cases from the CSV file."""
        df = pd.read_csv(self.csv_path)

        # Ensure required columns exist
        required_columns = ["Prompt", "Expected Result"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"CSV file is missing required columns: {missing_columns}")

        # Filter out rows where either Prompt or Expected Result is empty
        df = df.dropna(subset=required_columns)

        return df

    async def run_single_test(self, prompt: str, expected_result: str) -> Dict:
        """
        Run a single test using the TestService with robust error handling
        """
        rag_test = {}
        try:
            # Create test request
            request = ChatTestRequest(
                prompt=prompt,
                expected_result=expected_result,
                similarity_threshold=self.similarity_threshold,
                test_id=str(uuid4())
            )

            # Run the test through the test service
            result = await self.test_service.run_test(request)
            
            # Initialize with default values
            rag_response = ""
            no_rag_response = ""
            rag_score = 0.0
            no_rag_score = 0.0
            rag_value_rating = "Unknown"
            value_assessment = ""
            rag_passed = False
            no_rag_passed = False
            
            # Safely process detailed_analysis if it exists
            if hasattr(result, 'detailed_analysis') and result.detailed_analysis is not None:
                detailed_analysis = result.detailed_analysis
                
                # Get RAG and non-RAG specific data
                rag_response = detailed_analysis.get("rag_response", "")
                no_rag_response = detailed_analysis.get("no_rag_response", "")

                # Safely access nested dictionaries
                rag_test = detailed_analysis.get("rag_test") or {}
                rag_llm_test = detailed_analysis.get("rag_llm_test") or {}
                no_rag_test = detailed_analysis.get("no_rag_test") or {}
                no_rag_llm_test = detailed_analysis.get("no_rag_llm_test") or {}
                
                # Get similarity scores for both
                rag_score = max(
                    rag_test.get("weighted_similarity", 0),
                    rag_llm_test.get("normalized_score", 0) 
                )

                no_rag_score = max(
                    no_rag_test.get("weighted_similarity", 0),
                    no_rag_llm_test.get("normalized_score", 0)
                )

                # Get comparison data
                comparison = detailed_analysis.get("comparison", {})
                rag_value_rating = comparison.get("rag_value_rating", "Unknown")
                value_assessment = comparison.get("value_assessment", "")

                # Get individual pass/fail status
                rag_passed = comparison.get("rag_passed", False)
                no_rag_passed = comparison.get("no_rag_passed", False)
            else:
                # Log if detailed_analysis is missing
                print(f"Warning: detailed_analysis is None for prompt: {prompt[:50]}...")
        except Exception as e:
            # Handle any errors gracefully
            print(f"Error processing test for prompt '{prompt[:50]}...': {str(e)}")
            # Return a minimal result with error information
            return {
                "Prompt": prompt,
                "Expected Result": expected_result,
                "RAG Response": f"Error: {str(e)}",
                "Non-RAG Response": f"Error: {str(e)}",
                "Overall Passed": False,
                "RAG Passed": False,
                "Non-RAG Passed": False,
                "Failure Reasons": str(e),
                "RAG Similarity": 0,
                "Non-RAG Similarity": 0,
                "RAG Value Rating": "Error",
                "Value Assessment": f"Test failed with error: {str(e)}"
            }

        # Convert to dictionary format for DataFrame compatibility
        return {
            "Prompt": getattr(result, 'prompt', prompt),
            "Expected Result": getattr(result, 'expected_result', expected_result),
            "RAG Response": rag_response,
            "Non-RAG Response": no_rag_response,
            "Overall Passed": getattr(result, 'passed', False),
            "RAG Passed": rag_passed,
            "Non-RAG Passed": no_rag_passed,
            "Failure Reasons": getattr(result, 'reasoning', ''),
            "RAG Similarity": rag_score,
            "Non-RAG Similarity": no_rag_score,
            "RAG Value Rating": rag_value_rating,
            "Value Assessment": value_assessment,
            "Text Similarity": rag_test.get("basic_similarity", 0),
            "Word Overlap": rag_test.get("jaccard_similarity", 0),
            "Bigram Overlap": rag_test.get("bigram_overlap", 0),
            "Trigram Overlap": rag_test.get("trigram_overlap", 0),
            "Concept Coverage": rag_test.get("concept_coverage", 0),
            "Weighted Similarity": getattr(result, 'similarity_score', rag_score),
            "Missing Key Concepts": ", ".join(rag_test.get("key_concepts_missing", []))[:100],
            "Missing Numerical Values": ", ".join(rag_test.get("numbers_missing", []))[:100]
        }

    async def run_all_tests(self) -> pd.DataFrame:
        """Run all test cases and return results as a DataFrame."""
        # Load test cases from CSV
        test_cases_df = await self.load_test_cases()
        results = []

        total_tests = len(test_cases_df)

        for i, (_, row) in enumerate(test_cases_df.iterrows()):
            prompt = row["Prompt"]
            expected = row["Expected Result"]

            print(f"Running test {i + 1}/{total_tests}: {prompt[:50]}...")

            # Run single test
            result = await self.run_single_test(prompt, expected)
            results.append(result)

            # Print RAG value information
            value_rating = result.get("RAG Value Rating", "Unknown")
            if "Value Assessment" in result:
                print(f"   RAG Value Rating: {value_rating}")
                print(f"   Value Assessment: {result['Value Assessment']}")

        # Create DataFrame from results
        results_df = pd.DataFrame(results)

        # Calculate summary statistics
        total = len(results)
        passed = results_df["Overall Passed"].sum()
        rag_passed = results_df["RAG Passed"].sum()
        no_rag_passed = results_df["Non-RAG Passed"].sum()

        failed = total - passed
        pass_rate = (passed / total) * 100 if total > 0 else 0
        rag_pass_rate = (rag_passed / total) * 100 if total > 0 else 0
        no_rag_pass_rate = (no_rag_passed / total) * 100 if total > 0 else 0

        avg_rag_similarity = results_df["RAG Similarity"].mean()
        avg_no_rag_similarity = results_df["Non-RAG Similarity"].mean()

        high_value, medium_value, low_value, value_counts, negative_value = (0, 0, 0, None, 0)
        # Count by RAG value rating
        if "RAG Value Rating" in results_df.columns:
            value_counts = results_df["RAG Value Rating"].value_counts()
            high_value = value_counts.get("High", 0)
            medium_value = value_counts.get("Medium", 0)
            low_value = value_counts.get("Low", 0)
            negative_value = value_counts.get("Negative", 0)

        # Print summary
        print(f"\nSemantic Test Summary:")
        print(f"Total tests: {total}")
        print(f"Overall passed: {passed} ({pass_rate:.2f}%)")
        print(f"RAG passed: {rag_passed} ({rag_pass_rate:.2f}%)")
        print(f"Non-RAG passed: {no_rag_passed} ({no_rag_pass_rate:.2f}%)")
        print(f"Failed: {failed}")
        print(f"Average RAG similarity: {avg_rag_similarity:.4f}")
        print(f"Average Non-RAG similarity: {avg_no_rag_similarity:.4f}")
        print(f"Similarity delta (RAG - Non-RAG): {(avg_rag_similarity - avg_no_rag_similarity):.4f}")

        # Print RAG value rating summary
        if "RAG Value Rating" in results_df.columns:
            print("\nRAG Value Rating Summary:")
            print(f"High: {high_value} ({high_value / total * 100:.1f}%)")
            print(f"Medium: {medium_value} ({medium_value / total * 100:.1f}%)")
            print(f"Low: {low_value} ({low_value / total * 100:.1f}%)")
            print(f"Negative: {negative_value} ({negative_value / total * 100:.1f}%)")

        # Print failed tests
        if failed > 0:
            print("\nFailed Tests:")
            failed_tests = results_df[~results_df["Overall Passed"]]
            for i, (_, test) in enumerate(failed_tests.iterrows()):
                print(f"{i + 1}. Prompt: {test['Prompt'][:50]}...")
                print(f"   RAG Similarity: {test.get('RAG Similarity', 0):.4f}")
                print(f"   Non-RAG Similarity: {test.get('Non-RAG Similarity', 0):.4f}")
                print(f"   Failure reasons: {test['Failure Reasons']}")

                if "Missing Key Concepts" in test and test["Missing Key Concepts"]:
                    print(f"   Missing key concepts: {test['Missing Key Concepts']}")

                if "Missing Numerical Values" in test and test["Missing Numerical Values"]:
                    print(f"   Missing numerical values: {test['Missing Numerical Values']}")

                print(f"   Expected: {test['Expected Result'][:50]}...")
                print(f"   RAG Response: {test.get('RAG Response', '')[:50]}...")
                print(f"   Non-RAG Response: {test.get('Non-RAG Response', '')[:50]}...")
                print()

        return results_df

    async def run_batch_test(self) -> None:
        """Run batch test using the TestService's batch test functionality"""
        # Create batch test request
        request = ChatBatchTestRequest(
            csv_file=self.csv_path,
            similarity_threshold=self.similarity_threshold
        )

        # Run batch test
        result = await self.test_service.run_batch_test(
            request.csv_file,
            request.similarity_threshold
        )

        # Print summary
        print(f"\nBatch Test Results:")
        print(f"Total tests: {result.total_tests}")
        print(f"Passed: {result.passed} ({result.pass_rate:.2f}%)")
        print(f"Failed: {result.failed}")
        print(f"Results saved to: {result.output_file}")

        # Check if RAG comparison report is available
        if hasattr(result, 'rag_report_file') and result.rag_report_file:
            print(f"RAG comparison report saved to: {result.rag_report_file}")

        return None

    async def cleanup(self):
        """Clean up resources."""
        await self.test_service.cleanup()


async def main():
    # Check command line arguments
    if len(sys.argv) < 3:
        print("Usage: python chat_evaluator.py <api_url> <csv_file> [similarity_threshold]")
        print("Example: python chat_evaluator.py http://localhost:8005 test_cases.csv 0.7")
        sys.exit(1)

    api_url = sys.argv[1]
    csv_path = sys.argv[2]

    # Optional similarity threshold
    similarity_threshold = 0.7  # Default
    if len(sys.argv) > 3:
        try:
            similarity_threshold = float(sys.argv[3])
        except ValueError:
            print(f"Warning: Invalid similarity threshold '{sys.argv[3]}'. Using default (0.7).")

    # Check if we should use batch mode (using TestService's batch functionality)
    use_batch_mode = False
    if len(sys.argv) > 4 and sys.argv[4].lower() == 'batch':
        use_batch_mode = True

    # Get the full path assuming the file is in the current working directory
    full_path = os.path.join(os.getcwd(), csv_path)

    # Validate the file exists in the current working directory
    if not os.path.exists(full_path):
        print(f"Error: CSV file '{csv_path}' not found in the current working directory: {os.getcwd()}")
        sys.exit(1)

    print(f"Full path to the CSV file: {full_path}")
    print(f"Using similarity threshold: {similarity_threshold}")
    print(f"Mode: {'Batch' if use_batch_mode else 'Sequential'}")

    # Create the tester
    tester = ChatTesterCLI(api_url, csv_path, similarity_threshold)

    try:
        if use_batch_mode:
            # Run batch test using TestService's batch functionality
            await tester.run_batch_test()
        else:
            # Run tests in sequence
            results = await tester.run_all_tests()

            # Save results to CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"semantic_test_results_{timestamp}.csv"
            results.to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")

            # Also save a RAG-specific report with just the comparison data
            if "RAG Value Rating" in results.columns:
                rag_columns = [
                    "Prompt", "RAG Passed", "Non-RAG Passed",
                    "RAG Similarity", "Non-RAG Similarity",
                    "RAG Value Rating", "Value Assessment"
                ]

                rag_report = results[rag_columns] if all(col in results.columns for col in rag_columns) else results
                rag_report_file = f"rag_comparison_{timestamp}.csv"
                rag_report.to_csv(rag_report_file, index=False)
                print(f"RAG comparison report saved to {rag_report_file}")

    finally:
        await tester.cleanup()


if __name__ == "__main__":
    asyncio.run(main())