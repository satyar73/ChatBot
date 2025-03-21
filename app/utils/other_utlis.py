import json

# Load feature flags from a configuration file
def load_feature_flags():
    try:
        with open('feature_flags.json', 'r') as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        # return nothing
        return {}

def load_json(json_file):
    """
    Load QA pairs from a JSON file and cache them in memory.

    Args:
        json_file (str): Path to the JSON file
    """
    qa_data = {}
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Build a dictionary with questions as keys and answers as values
        for item in data:
            qa_data[item['Prompt']] = item['Expected Result']

        print(f"Loaded {len(qa_data)} QA pairs into cache")
    except Exception as e:
        print(f"Error loading JSON file: {e}")

    return qa_data

def write_data_logfile (context, data, log_file):
    json_data = json.dumps(data)
    print(f"DEBUG: Writing LLM {context} log to {log_file}, data length: {len(json_data)}")

    # Direct file write as a fallback mechanism
    with open(log_file, "a") as f:
        f.write(json_data + "\n")

    return json_data