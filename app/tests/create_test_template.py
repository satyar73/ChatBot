import pandas as pd
import sys

def create_test_template(filename, num_examples=5):
    """
    Create a test template CSV file with example rows.
    
    Args:
        filename: Output CSV filename
        num_examples: Number of example rows to include
    """
    # Example test cases
    examples = [
        {
            "Prompt": "What is the weather like today?",
            "Expected Result": "I don't have access to real-time weather information"
        },
        {
            "Prompt": "Tell me about yourself",
            "Expected Result": "I am an AI assistant"
        },
        {
            "Prompt": "What is 2 + 2?",
            "Expected Result": "4"
        },
        {
            "Prompt": "Explain quantum computing",
            "Expected Result": "quantum computing uses quantum phenomena"
        },
        {
            "Prompt": "Generate a poem about nature",
            "Expected Result": "trees"
        }
    ]
    
    # Create DataFrame with examples (limited by num_examples)
    df = pd.DataFrame(examples[:min(num_examples, len(examples))])
    
    # Add empty rows for user to fill
    empty_rows = pd.DataFrame([{"Prompt": "", "Expected Result": ""}] * (10 - num_examples))
    df = pd.concat([df, empty_rows], ignore_index=True)
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Template created: {filename}")
    print(f"Added {num_examples} example rows and {10 - num_examples} empty rows.")
    print("Edit the CSV with your test cases and run the test script.")

if __name__ == "__main__":
    output_file = "test_cases.csv"
    
    # Check if filename was provided as command line argument
    if len(sys.argv) > 1:
        output_file = sys.argv[1]
    
    create_test_template(output_file)
