"""
CLI client for interacting with the ChatBot testing API.

This module provides a command-line interface for running tests against the ChatBot API.
It uses the test_routes API endpoints rather than directly instantiating the ChatTestService.
"""
import asyncio
import sys
import os
import json
import time
import pandas as pd
import aiohttp
from datetime import datetime
from typing import Dict, Any, List, Optional
from uuid import uuid4

from app.utils.logging_utils import get_logger


class ChatTesterCLI:
    """
    Command-line interface for running tests using the ChatBot testing API.
    """

    def __init__(self, api_url: str, csv_path: str, similarity_threshold: float = 0.7):
        """
        Initialize the chat tester CLI.

        Args:
            api_url: Base URL for the API (e.g., http://localhost:8005)
            csv_path: Path to CSV file containing test cases
            similarity_threshold: Threshold for similarity score to consider a test passed
        """
        self.api_url = api_url
        self.csv_path = csv_path
        self.similarity_threshold = similarity_threshold
        self.logger = get_logger(__name__)
        self.session_id = f"test_session_{int(datetime.now().timestamp())}"
        self.session = None
        
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an HTTP session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=60)  # Longer timeout for test operations
            )
        return self.session

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
        Run a single test using the testing API endpoint.
        
        Args:
            prompt: The test prompt
            expected_result: The expected result
            
        Returns:
            A dictionary with test results
        """
        session = await self._get_session()
        
        try:
            # Create the request payload
            payload = {
                "prompt": prompt,
                "expected_result": expected_result,
                "similarity_threshold": self.similarity_threshold,
                "test_id": str(uuid4())
            }
            
            # Call the single test endpoint
            async with session.post(f"{self.api_url}/test/single", json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API request failed with status {response.status}: {error_text}")
                
                result = await response.json()
                
                # Process the result
                detailed_analysis = result.get("detailed_analysis", {})
                
                # Extract RAG and non-RAG responses
                rag_response = detailed_analysis.get("rag_response", "")
                no_rag_response = detailed_analysis.get("no_rag_response", "")
                
                # Extract test results
                rag_test = detailed_analysis.get("rag_test", {})
                rag_llm_test = detailed_analysis.get("rag_llm_test", {})
                no_rag_test = detailed_analysis.get("no_rag_test", {})
                no_rag_llm_test = detailed_analysis.get("no_rag_llm_test", {})
                
                # Get scores
                rag_score = max(
                    rag_test.get("weighted_similarity", 0),
                    rag_llm_test.get("normalized_score", 0) if rag_llm_test else 0
                )
                
                no_rag_score = max(
                    no_rag_test.get("weighted_similarity", 0),
                    no_rag_llm_test.get("normalized_score", 0) if no_rag_llm_test else 0
                )
                
                # Get comparison data
                comparison = detailed_analysis.get("comparison", {})
                rag_value_rating = comparison.get("rag_value_rating", "Unknown")
                value_assessment = comparison.get("value_assessment", "")
                
                # Get pass/fail status
                rag_passed = comparison.get("rag_passed", False)
                no_rag_passed = comparison.get("no_rag_passed", False)
                
                # Return formatted result
                return {
                    "Prompt": result.get("prompt", prompt),
                    "Expected Result": result.get("expected_result", expected_result),
                    "RAG Response": rag_response,
                    "Non-RAG Response": no_rag_response,
                    "Overall Passed": result.get("passed", False),
                    "RAG Passed": rag_passed,
                    "Non-RAG Passed": no_rag_passed,
                    "Failure Reasons": result.get("reasoning", ""),
                    "RAG Similarity": rag_score,
                    "Non-RAG Similarity": no_rag_score,
                    "RAG Value Rating": rag_value_rating,
                    "Value Assessment": value_assessment,
                    "Text Similarity": rag_test.get("basic_similarity", 0),
                    "Word Overlap": rag_test.get("jaccard_similarity", 0),
                    "Bigram Overlap": rag_test.get("bigram_similarity", 0),
                    "Trigram Overlap": rag_test.get("trigram_similarity", 0),
                    "Concept Coverage": rag_test.get("concept_coverage", 0),
                    "Weighted Similarity": result.get("similarity_score", rag_score),
                    "Missing Key Concepts": ", ".join(rag_test.get("concepts_missing", []))[:100],
                    "Missing Numerical Values": ""  # Not available in current API response
                }
        except Exception as e:
            # Handle any errors gracefully
            self.logger.error(f"Error processing test for prompt '{prompt[:50]}...': {str(e)}")
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

        high_value, medium_value, low_value, negative_value = (0, 0, 0, 0)
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

                print(f"   Expected: {test['Expected Result'][:50]}...")
                print(f"   RAG Response: {test.get('RAG Response', '')[:50]}...")
                print(f"   Non-RAG Response: {test.get('Non-RAG Response', '')[:50]}...")
                print()

        return results_df

    async def start_batch_test(self) -> Dict[str, Any]:
        """
        Start a batch test job using the batch test API endpoint.
        
        Returns:
            Dictionary with job information
        """
        session = await self._get_session()
        
        try:
            # Prepare the file for upload
            with open(self.csv_path, 'rb') as f:
                files = {'csv_file': (os.path.basename(self.csv_path), f, 'text/csv')}
                
                # Make the request with both form data and query params
                async with session.post(
                    f"{self.api_url}/test/batch/start",
                    params={"similarity_threshold": self.similarity_threshold},
                    data=files
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"API request failed with status {response.status}: {error_text}")
                    
                    return await response.json()
        except Exception as e:
            self.logger.error(f"Error starting batch test: {str(e)}")
            raise

    async def check_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Check the status of a batch test job.
        
        Args:
            job_id: The ID of the job to check
            
        Returns:
            Dictionary with job status information
        """
        session = await self._get_session()
        
        try:
            async with session.get(f"{self.api_url}/test/jobs/{job_id}") as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API request failed with status {response.status}: {error_text}")
                
                return await response.json()
        except Exception as e:
            self.logger.error(f"Error checking job status: {str(e)}")
            raise

    async def run_batch_test(self) -> None:
        """Run batch test using the API's batch test functionality and wait for completion."""
        try:
            # Start the batch test job
            job_response = await self.start_batch_test()
            job_id = job_response.get("job_id")
            
            if not job_id:
                raise Exception("No job ID received from batch test request")
            
            print(f"Batch test job started with ID: {job_id}")
            print(f"Status: {job_response.get('status', 'unknown')}")
            print("Waiting for job to complete...")
            
            # Poll for job completion
            completed = False
            start_time = time.time()
            last_progress = -1
            last_message = ""
            
            while not completed and (time.time() - start_time) < 3600:  # 1-hour timeout
                # Wait before polling again
                await asyncio.sleep(2)
                
                # Check job status
                job_status = await self.check_job_status(job_id)
                status = job_status.get("status")
                progress = job_status.get("progress", 0)
                message = job_status.get("message", "")
                
                # Only print if there's a change in progress or message
                if progress != last_progress or message != last_message:
                    # If this is a progress update, show it on the same line
                    if progress > 0 and progress < 100 and last_progress > 0:
                        print(f"\rProgress: {progress}% - {message}", end="", flush=True)
                    else:
                        # For status changes, print on a new line
                        print(f"\nStatus: {status} - Progress: {progress}%")
                        if message:
                            print(f"Message: {message}")
                    
                    last_progress = progress
                    last_message = message
                
                # Check if job is completed
                if status in ["completed", "failed"]:
                    completed = True
                    print("\n")  # Ensure we're on a new line
                    
                    if status == "completed":
                        print("Batch test completed successfully!")
                    else:
                        print(f"Batch test failed: {message}")
                    
                    # Print results location if available
                    result = job_status.get("result", {})
                    if isinstance(result, dict):
                        if "output_file" in result:
                            print(f"Results saved to: {result['output_file']}")
                        if "rag_report_file" in result:
                            print(f"RAG comparison report saved to: {result['rag_report_file']}")
            
            if not completed:
                print("Batch test timed out after 1 hour")
            
        except Exception as e:
            print(f"Error running batch test: {str(e)}")
            
    async def cleanup(self):
        """Clean up resources."""
        if self.session and not self.session.closed:
            try:
                await self.session.close()
            except Exception as e:
                self.logger.error(f"Error closing session: {str(e)}")
            finally:
                self.session = None


async def main():
    # Check command line arguments
    if len(sys.argv) < 3:
        print("Usage: python -m app.utils.chat_tester_cli <api_url> <csv_file> [similarity_threshold]")
        print("Example: python -m app.utils.chat_tester_cli http://localhost:8005 test_cases.csv 0.7")
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

    # Check if we should use batch mode (using API's batch functionality)
    use_batch_mode = False
    if len(sys.argv) > 4 and sys.argv[4].lower() == 'batch':
        use_batch_mode = True

    # Get the full path assuming the file is in the current working directory
    full_path = os.path.join(os.getcwd(), csv_path)
    if not os.path.exists(full_path):
        # Try the absolute path directly
        if not os.path.exists(csv_path):
            print(f"Error: CSV file '{csv_path}' not found in the current working directory: {os.getcwd()}")
            sys.exit(1)
        full_path = csv_path

    print(f"Full path to the CSV file: {full_path}")
    print(f"Using similarity threshold: {similarity_threshold}")
    print(f"Mode: {'Batch' if use_batch_mode else 'Sequential'}")
    print(f"API endpoint: {api_url}")
    
    # Test API connectivity
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{api_url}/test/status") as response:
                if response.status != 200:
                    print(f"Warning: API status check returned {response.status}. The API might not be available.")
                else:
                    status_data = await response.json()
                    print(f"API status: {status_data.get('status', 'unknown')}")
                    print(f"API message: {status_data.get('message', 'No message')}")
    except Exception as e:
        print(f"Warning: Could not connect to API at {api_url}. Error: {str(e)}")
        print("Please ensure the server is running before continuing.")
        # Ask if the user wants to continue despite the connection error
        user_input = input("Continue anyway? (y/n): ")
        if user_input.lower() != 'y':
            sys.exit(1)

    # Create the tester
    tester = ChatTesterCLI(api_url, full_path, similarity_threshold)

    try:
        if use_batch_mode:
            # Run batch test using API's batch functionality
            await tester.run_batch_test()
        else:
            # Run tests in sequence
            results = await tester.run_all_tests()

            # Save results to CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Get the absolute path to the test_results directory
            # Use project root instead of current working directory
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            results_dir = os.path.join(project_root, "test_results")
            
            # Create results directory if it doesn't exist
            os.makedirs(results_dir, exist_ok=True)
            
            output_file = os.path.join(results_dir, f"semantic_test_results_{timestamp}.csv")
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
                rag_report_file = os.path.join(results_dir, f"rag_comparison_{timestamp}.csv")
                rag_report.to_csv(rag_report_file, index=False)
                print(f"RAG comparison report saved to {rag_report_file}")

    finally:
        await tester.cleanup()


if __name__ == "__main__":
    asyncio.run(main())